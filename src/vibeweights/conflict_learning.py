"""Durable conflict hold and additive, independently verified relearning.

Trusted single-owner host; typed finite domains only. A case is not a new
residency state and never authorizes deleting or editing original weights.
"""
from dataclasses import asdict
import json
from .abi import ContractError
from .environment import identifier
from .knowledge_learning import Domain, _hash
from .population import RESIDENT


def _head(l, d, entity):
    row=l.knowledge.db.execute('SELECT id FROM knowledge_heads_v1 WHERE scope=? AND key=?',
        ('global',f'{d.topic}:entity:{entity}')).fetchone()
    return None if row is None else l.knowledge.read(row[0],l.access)


def open_case(l, d, entity):
    row=l.knowledge.db.execute('SELECT receipt FROM conflict_cases_v08 WHERE topic=? AND entity=? AND closed=0',
        (d.topic,entity)).fetchone()
    if row is None: return None
    case=json.loads(row[0])
    if case['domain']!=asdict(d): raise ContractError('conflict domain mismatch')
    return case


def record_conflict(l, task, d, entity, outputs):
    """Caller holds the source guard; head change and repair case commit together."""
    if not l.knowledge.db.in_transaction or not l._guard_depth:
        raise ContractError('conflict guard required')
    known={v for v in outputs.values() if v is not None}
    if len(known)<2 or open_case(l,d,entity) is not None:
        raise ContractError('new conflict required')
    prior=_head(l,d,entity)
    suspects=sorted(k for k,v in outputs.items() if v is not None)
    affected=set(suspects)
    if prior is not None: affected.update(l.dependencies.consumers(prior.id))
    affected=l.dependencies.descendants(affected)
    # Intents without an artifact are not operational reconstruction targets.
    affected={e for e in affected if l.db.execute('SELECT 1 FROM experts WHERE id=?',(e,)).fetchone()}
    # Preserve the old fact and all competing outputs for review. The head is
    # not verified and cannot be consumed by the ordinary memory/learning path.
    rec=l.knowledge._store_locked(l.access,scope='global',key=f'{d.topic}:entity:{entity}',topic=d.topic,
        values=prior.values if prior else tuple(sorted(known)),source='conflict:'+_hash(task),
        status='CONFLICTED',expected=None if prior is None else prior.id)
    case={'task':task,'domain':asdict(d),'entity':entity,'outcome':'CONFLICT','answer':None,
          'experts':sorted(outputs),'outputs':outputs,'suspects':suspects,'affected':sorted(affected),
          'prior_id':None if prior is None else prior.id,'uncertain_id':rec.id,'conflict_case':task}
    l.knowledge.db.execute('INSERT INTO conflict_cases_v08(task,topic,entity,receipt,uncertain_id) VALUES(?,?,?,?,?)',
        (task,d.topic,entity,json.dumps(case,sort_keys=True),rec.id))
    l.knowledge.db.commit()
    return case


class ConflictLearning:
    def __init__(self, tasks):
        from .task_learning import TaskLearning
        if not isinstance(tasks,TaskLearning): raise ContractError('task learning required')
        self.tasks=tasks; self.learning=tasks.learning

    def case(self, task):
        identifier(task)
        row=self.learning.knowledge.db.execute('SELECT receipt,resolution,closed FROM conflict_cases_v08 WHERE task=?',(task,)).fetchone()
        if row is None: raise ContractError('unknown conflict case')
        return {'receipt':json.loads(row[0]),'resolution':None if row[1] is None else json.loads(row[1]),'closed':bool(row[2])}

    def pending(self):
        return tuple(self.case(t) for (t,) in self.learning.knowledge.db.execute('SELECT task FROM conflict_cases_v08 WHERE closed=0 ORDER BY task'))

    def reconstruction_pending(self):
        return tuple((task,tuple(json.loads(raw)['pending_reconstruction']))
                     for task,raw in self.learning.knowledge.db.execute(
                         'SELECT task,resolution FROM conflict_cases_v08 WHERE resolution IS NOT NULL ORDER BY task')
                     if json.loads(raw)['pending_reconstruction'])

    def stage(self, task, *, job_id, expert_id, oracle, oracle_id, evidence_id, method='TRAINED', steps=80):
        """Resolve using independent evidence and stage a NEW candidate.

        Also recover independently confirmed unaffected facts of same-domain
        quarantined experts. Unresolved dependents stay recorded, never silently
        released. No teacher answer or model consensus is evidence by itself.
        """
        l=self.learning
        for x in (task,job_id,expert_id,oracle_id,evidence_id): identifier(x)
        if not callable(oracle) or method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256:
            raise ContractError('correction policy')
        policy={'contract':'conflict-correction-v1','job_id':job_id,'expert_id':expert_id,
                'oracle_id':oracle_id,'evidence_id':evidence_id,'method':method,'steps':steps}
        with l._guard():
            c=self.case(task); obs=c['receipt']; d=Domain(**obs['domain']); entity=obs['entity']
            if c['resolution'] is not None:
                if c['resolution']['policy']!=policy: raise ContractError('correction policy changed')
            else:
                if c['closed']: raise ContractError('closed conflict')
                current=_head(l,d,entity)
                if current is None or current.id!=obs['uncertain_id'] or current.status!='CONFLICTED':
                    raise ContractError('conflict head changed; review required')
                if (l.db.execute('SELECT 1 FROM experts WHERE id=?',(expert_id,)).fetchone() or
                    l.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job_id,expert_id)).fetchone() or
                    l.knowledge.db.execute('SELECT 1 FROM learning_intakes_v08 WHERE job_id=? OR expert_id=?',(job_id,expert_id)).fetchone()):
                    raise ContractError('correction identity reserved')
                expected=oracle(entity); d.target(expected)
                if expected is None: raise ContractError('conflict remains unresolved')
                # Preserve other competencies when they can be independently
                # recovered from current verified roots. Do not copy stale data.
                restore={}; pending=set(); roots=set()
                for eid in obs['affected']:
                    m=l.catalog.manifest(eid)
                    if 'learning_job' not in m: pending.add(eid); continue
                    j=l.job(m['learning_job'])
                    if j['spec']['domain']!=asdict(d): pending.add(eid); continue
                    for rid in j['spec']['roots']:
                        old=l.knowledge.read(rid,l.access)
                        i=int(old.key.rsplit(':',1)[1])
                        if i==entity: continue
                        if open_case(l,d,i) is not None: pending.add(eid); continue
                        cur=_head(l,d,i)
                        if cur is None: pending.add(eid); continue
                        try: examples,_=l._snapshot([cur.id],d)
                        except ContractError: pending.add(eid); continue
                        value=oracle(i); d.target(value)
                        if value is None or value!=examples[i]: pending.add(eid); continue
                        restore[i]=value; roots.add(cur.id)
                rec=l.knowledge._store_locked(l.access,scope='global',key=current.key,topic=d.topic,values=(expected,),
                    source='correction:'+_hash(policy),status='VERIFIED',expected=current.id)
                restore[entity]=expected; roots.add(rec.id)
                # Source validation catches any roots invalidated by the change.
                resolution={'policy':policy,'new_head':rec.id,'roots':sorted(roots),
                            'expected':{str(k):v for k,v in sorted(restore.items())},
                            'pending_reconstruction':sorted(pending),'case_sha256':_hash(obs)}
                l.record_intake(job_id=job_id,expert_id=expert_id,domain=d,ids=sorted(roots),method=method,steps=steps,
                    derivation={'correction':resolution})
                l.knowledge.db.execute('UPDATE conflict_cases_v08 SET resolution=? WHERE task=? AND resolution IS NULL',
                    (json.dumps(resolution,sort_keys=True),task))
                l.knowledge.db.commit()
        # New guarded transaction; durable intent survives a build interruption.
        return l.resume_intake(job_id)

    def promote(self,task,oracle,*,oracle_id):
        """Independent candidate check. Close the case only after admission."""
        identifier(oracle_id)
        if not callable(oracle): raise ContractError('correction oracle required')
        l=self.learning
        with l._guard():
            c=self.case(task); r=c['resolution']
            if r is None or r['policy']['oracle_id']!=oracle_id: raise ContractError('correction promotion contract')
            d=Domain(**c['receipt']['domain']); job_id=r['policy']['job_id']; j=l.job(job_id)
            if j['spec']['derivation']!={'correction':r}: raise ContractError('correction job mismatch')
            l._check_sources(j)
            # A correction cannot be closed with different evidence on retry.
            expected={int(i):v for i,v in r['expected'].items()}
            for i,v in expected.items():
                actual=oracle(i)
                if type(actual) is not int or actual!=v: raise ContractError('correction verification changed')
            if j['state']=='STAGED':
                if not l.promote(job_id,lambda i: expected.get(i),oracle_id=oracle_id): return False
            elif j['state'] not in RESIDENT or not j['report'] or not j['report']['accepted']:
                raise ContractError('correction not admitted')
            l.knowledge.db.execute('UPDATE conflict_cases_v08 SET closed=1 WHERE task=?',(task,))
            l.knowledge.db.commit()
            return True
