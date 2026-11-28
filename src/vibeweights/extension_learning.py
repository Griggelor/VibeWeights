"""Additive single-fact learning from a recorded abstention. Trusted host only."""
from dataclasses import asdict
import json
from .abi import ContractError
from .bootstrap import BootstrapExpert
from .environment import identifier
from .knowledge_learning import Domain, _hash
from .population import RESIDENT
from .task_learning import TaskLearning


def _parse(raw, entity, domain):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj: raise ContractError('duplicate teacher key')
            obj[k]=v
        return obj
    if type(raw) is not str or len(raw)>1024: raise ContractError('response bounds')
    try: data=json.loads(raw,object_pairs_hook=unique)
    except (ValueError,TypeError,RecursionError) as exc: raise ContractError('extension JSON') from exc
    if (type(data) is not dict or set(data)!={'entity','label'} or type(data['entity']) is not int
        or data['entity']!=entity or type(data['label']) is not int): raise ContractError('extension fact contract')
    domain.target(data['label']); return data['label']


class ExtensionLearning:
    def __init__(self,tasks):
        if not isinstance(tasks,TaskLearning): raise ContractError('task learning required')
        self.tasks=tasks; self.learning=tasks.learning
        self.learning.db.execute('CREATE TABLE IF NOT EXISTS extension_reviews_v08(review TEXT PRIMARY KEY, receipt TEXT NOT NULL)')
        self.learning.db.commit()

    def _outputs(self,d,entity):
        l=self.learning; outputs={}
        for m in l.catalog.list():
            if m['family']!=d.topic: continue
            if 'learning_job' not in m: raise ContractError('untyped domain expert')
            j=l.job(m['learning_job'])
            if j['spec']['domain']!=asdict(d): raise ContractError('extension domain mismatch')
            if m['state'] in RESIDENT:
                if not j['report'] or not j['report']['accepted']: raise ContractError('unapproved expert')
                outputs[m['id']]=d.decode(l.population.compute(m['id'],d.encode(entity)))
        return outputs

    def stage(self,task,teacher,*,request,job_id,expert_id,oracle,oracle_id,method='TRAINED',steps=80):
        """Store a verified fact and durable intake atomically, then stage only."""
        l=self.learning
        for value in (task,job_id,expert_id,oracle_id): identifier(value)
        source=identifier('teacher:'+job_id)
        if not isinstance(teacher,BootstrapExpert) or not callable(oracle): raise ContractError('teacher/oracle required')
        if type(request) is not str or not request.strip() or len(request)>8000: raise ContractError('request bounds')
        if method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256: raise ContractError('learning budget')
        obs=self.tasks.observation(task)
        if obs['outcome']!='ABSTAINED': raise ContractError('recorded abstention required')
        d=Domain(**obs['domain']); entity=obs['entity']; key=f'{d.topic}:entity:{entity}'
        policy={'contract':'task-extension-v1','task_sha256':_hash(obs),'entity':entity,'request_sha256':_hash(request),
                'teacher_id':teacher.spec.id,'model':teacher.spec.model,'oracle_id':oracle_id}
        def preflight():
            pending=l.knowledge.db.execute('SELECT expert_id,spec FROM learning_intakes_v08 WHERE job_id=?',(job_id,)).fetchone()
            if pending:
                spec=json.loads(pending[1]); derivation=spec['derivation']
                if (pending[0]!=expert_id or spec['domain']!=asdict(d) or spec['method']!=method or spec['steps']!=steps
                    or not isinstance(derivation,dict) or derivation.get('policy')!=policy): raise ContractError('extension intake mismatch')
                return l.resume_intake(job_id)
            if (l.db.execute('SELECT 1 FROM experts WHERE id=?',(expert_id,)).fetchone() or
                l.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job_id,expert_id)).fetchone() or
                l.knowledge.db.execute('SELECT 1 FROM learning_intakes_v08 WHERE expert_id=?',(expert_id,)).fetchone()): raise ContractError('extension identity reserved')
            if l.knowledge.db.execute('SELECT 1 FROM knowledge_v1 WHERE scope=? AND key=? LIMIT 1',('global',key)).fetchone():
                raise ContractError('fact already reserved; use correction or reactivation')
            outputs=self._outputs(d,entity)
            if not outputs or any(v is not None for v in outputs.values()): raise ContractError('no current resident gap')
            return None
        with l._guard():
            old=preflight()
            if old is not None: return old
        result=teacher.propose(request+f'\nReturn only JSON with entity {entity} and integer label 0..{d.labels-1}.')
        if not isinstance(result,dict) or result.get('verified') is not False: raise ContractError('unverified teacher required')
        raw=result.get('proposal'); label=_parse(raw,entity,d)
        with l._guard():
            old=preflight()
            if old is not None: return old
            expected=oracle(entity); d.target(expected)
            if label!=expected: raise ContractError('independent extension verification failed')
            rec=l.knowledge._store_locked(l.access,scope='global',key=key,topic=d.topic,values=(label,),source=source,status='VERIFIED')
            l.record_intake(job_id=job_id,expert_id=expert_id,domain=d,ids=[rec.id],method=method,steps=steps,
                            derivation={'policy':policy,'proposal_sha256':_hash(raw)})
            l.knowledge.db.commit()
        return l.resume_intake(job_id)

    def promote(self,job_id,oracle,*,oracle_id):
        """Validate the new fact and abstention on all other IDs, separately."""
        l=self.learning; j=l.job(job_id); derivation=j['spec']['derivation']
        policy=derivation.get('policy',{}) if isinstance(derivation,dict) else {}
        if policy.get('contract')!='task-extension-v1' or policy.get('oracle_id')!=oracle_id or not callable(oracle): raise ContractError('extension promotion contract')
        return l.promote(job_id,lambda i: oracle(i) if i==policy['entity'] else None,oracle_id=oracle_id)

    def review_conflict(self,review,task,oracle,*,oracle_id):
        """Record independent evidence about a current disagreement; no state change."""
        for value in (review,task,oracle_id): identifier(value)
        if not callable(oracle): raise ContractError('oracle required')
        l=self.learning; obs=self.tasks.observation(task)
        if obs['outcome']!='CONFLICT': raise ContractError('recorded conflict required')
        d=Domain(**obs['domain']); entity=obs['entity']
        with l._guard():
            outputs=self._outputs(d,entity)
            if len({v for v in outputs.values() if v is not None})<2: raise ContractError('conflict no longer present')
            expected=oracle(entity);d.target(expected)
            receipt={'review':review,'task_sha256':_hash(obs),'oracle_id':oracle_id,'expected':expected,'outputs':outputs,
                     'disagreeing':sorted(k for k,v in outputs.items() if v is not None and v!=expected) if expected is not None else [],
                     'outcome':'UNRESOLVED' if expected is None else 'EVIDENCE_RECORDED'}
            encoded=json.dumps(receipt,sort_keys=True)
            with l.db:
                prior=l.db.execute('SELECT receipt FROM extension_reviews_v08 WHERE review=?',(review,)).fetchone()
                if prior is not None and prior!=(encoded,): raise ContractError('conflict review changed')
                if prior is None: l.db.execute('INSERT INTO extension_reviews_v08 VALUES(?,?)',(review,encoded))
            return receipt
