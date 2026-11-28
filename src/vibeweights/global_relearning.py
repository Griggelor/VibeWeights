"""Global repair impulses and additive cross-domain reconstruction.

The durable conflict case is the broadcast source, not a cognitive router.
Only trusted host-authorized independent evidence may rebuild a candidate.
"""
from dataclasses import asdict
import json
from .abi import ContractError
from .environment import identifier
from .knowledge_learning import Domain, _hash
from .conflict_learning import ConflictLearning, open_case, _head
from .population import RESIDENT


class GlobalRelearning:
    def __init__(self, tasks):
        self.conflicts=ConflictLearning(tasks)
        self.learning=tasks.learning
        self.db=self.learning.knowledge.db
        self.db.execute('''CREATE TABLE IF NOT EXISTS global_repair_attempts_v08(
          task TEXT NOT NULL, predecessor TEXT NOT NULL, job_id TEXT PRIMARY KEY,
          expert_id TEXT NOT NULL UNIQUE, receipt TEXT NOT NULL,
          completed INTEGER NOT NULL DEFAULT 0)''')
        self.db.execute('CREATE INDEX IF NOT EXISTS global_repair_case_v08 ON global_repair_attempts_v08(task,predecessor)')
        self.db.commit()

    def attempts(self, task, predecessor):
        return tuple({'receipt':json.loads(raw),'completed':bool(done)} for raw,done in self.db.execute(
            'SELECT receipt,completed FROM global_repair_attempts_v08 WHERE task=? AND predecessor=? ORDER BY rowid',
            (task,predecessor)))

    def _live(self, attempt):
        l=self.learning; r=attempt['receipt']; job=r['policy']['job_id']
        if not attempt['completed']: return False
        j=l.job(job)
        if j['state'] not in RESIDENT|{'ARCHIVED'} or not j['report'] or not j['report']['accepted']: return False
        try: l._check_sources(j)
        except ContractError: return False
        return True

    def impulses(self):
        """Persistent global notification view; includes open and downstream work.

        Not an event delivery guarantee, and not a model authorization. Consumers
        may poll the same impulse repeatedly and use case/attempt IDs for dedup.
        """
        l=self.learning; result=[]
        with l._guard():
            for task, in self.db.execute('SELECT task FROM conflict_cases_v08 ORDER BY task').fetchall():
                c=self.conflicts.case(task); obs=c['receipt']; resolution=c['resolution']
                pending=[] if resolution is None else [e for e in resolution['pending_reconstruction']
                    if not any(self._live(a) for a in self.attempts(task,e))]
                if not c['closed'] or pending:
                    result.append({'scope':'global','case':task,'domain':obs['domain'],'entity':obs['entity'],
                        'affected':obs['affected'],'local_open':not c['closed'],'pending_reconstruction':pending})
        return tuple(result)

    def stage(self, task, predecessor, *, job_id, expert_id, oracle, oracle_id,
              evidence_id, method='DIRECT', steps=80):
        """Re-establish old root competencies using independent current evidence.

        This is independent fact reconstruction, not replay of the old program.
        Never take a stale label as truth or declare a quarantined parent live.
        """
        l=self.learning
        for x in (task,predecessor,job_id,expert_id,oracle_id,evidence_id): identifier(x)
        if not callable(oracle) or method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256:
            raise ContractError('global repair policy')
        policy={'contract':'global-repair-v1','task':task,'predecessor':predecessor,'job_id':job_id,
                'expert_id':expert_id,'oracle_id':oracle_id,'evidence_id':evidence_id,'method':method,'steps':steps}
        with l._guard():
            c=self.conflicts.case(task); resolution=c['resolution']
            if not c['closed'] or resolution is None or predecessor not in resolution['pending_reconstruction']:
                raise ContractError('global repair requires closed local case and pending predecessor')
            prior=self.db.execute('SELECT receipt FROM global_repair_attempts_v08 WHERE job_id=?',(job_id,)).fetchone()
            if prior:
                if json.loads(prior[0])['policy']!=policy: raise ContractError('global repair policy changed')
            else:
                for attempt in self.attempts(task,predecessor):
                    if self._live(attempt): raise ContractError('global repair already complete')
                    prev=attempt['receipt']['policy']['job_id']
                    row=l.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=?',(prev,)).fetchone()
                    if row is None: raise ContractError('resume existing global repair')
                    j=l.job(prev)
                    if j['state']!='QUARANTINED' and (not j['report'] or j['report']['accepted']):
                        raise ContractError('resume existing global repair')
                if (l.db.execute('SELECT 1 FROM experts WHERE id=?',(expert_id,)).fetchone() or
                    l.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job_id,expert_id)).fetchone() or
                    self.db.execute('SELECT 1 FROM learning_intakes_v08 WHERE job_id=? OR expert_id=?',(job_id,expert_id)).fetchone()):
                    raise ContractError('global repair identity reserved')
                m=l.catalog.manifest(predecessor)
                if m['state']!='QUARANTINED' or 'learning_job' not in m: raise ContractError('unsupported predecessor')
                old=l.job(m['learning_job']); d=Domain(**old['spec']['domain']); roots=[]; expected={}; history=[]
                for rid in old['spec']['roots']:
                    historic=l.knowledge.read(rid,l.access); entity=int(historic.key.rsplit(':',1)[1])
                    if open_case(l,d,entity): raise ContractError('root has open conflict')
                    current=_head(l,d,entity)
                    if current is None or current.status!='VERIFIED': raise ContractError('root needs separate resolution')
                    value=oracle(entity); d.target(value)
                    if value is None: raise ContractError('independent evidence incomplete')
                    try: l.knowledge.current_verified(current.id,l.access)
                    except ContractError:
                        # Independent evidence re-establishes the fact without the
                        # invalid derivation. Historical linkage stays in receipt.
                        current=l.knowledge._store_locked(l.access,scope='global',key=current.key,topic=d.topic,
                            values=(value,),source='global-repair:'+_hash(policy),status='VERIFIED',expected=current.id)
                    else:
                        if current.values!=(value,): raise ContractError('new contradiction requires separate resolution')
                    roots.append(current.id); expected[str(entity)]=value; history.append(rid)
                receipt={'policy':policy,'domain':asdict(d),'old_artifact':m['sha256'],'historical_roots':history,
                         'roots':sorted(roots),'expected':expected,'case_sha256':_hash(c['receipt'])}
                l.record_intake(job_id=job_id,expert_id=expert_id,domain=d,ids=sorted(roots),method=method,steps=steps,
                                derivation={'global_repair':receipt})
                self.db.execute('INSERT INTO global_repair_attempts_v08(task,predecessor,job_id,expert_id,receipt) VALUES(?,?,?,?,?)',
                    (task,predecessor,job_id,expert_id,json.dumps(receipt,sort_keys=True)))
                self.db.commit()
        return l.resume_intake(job_id)

    def promote(self, job_id, oracle, *, oracle_id):
        """Independent acceptance; keep impulse pending until admission succeeds."""
        l=self.learning; identifier(job_id); identifier(oracle_id)
        if not callable(oracle): raise ContractError('global repair oracle required')
        with l._guard():
            row=self.db.execute('SELECT receipt FROM global_repair_attempts_v08 WHERE job_id=?',(job_id,)).fetchone()
            if row is None: raise ContractError('unknown global repair')
            r=json.loads(row[0]); j=l.job(job_id)
            if r['policy']['oracle_id']!=oracle_id or j['spec']['derivation']!={'global_repair':r}:
                raise ContractError('global repair promotion contract')
            l._check_sources(j)
            expected={int(i):v for i,v in r['expected'].items()}
            for i,v in expected.items():
                actual=oracle(i)
                if type(actual) is not int or actual!=v: raise ContractError('global repair evidence changed')
            if j['state']=='STAGED':
                if not l.promote(job_id,expected.get,oracle_id=oracle_id): return False
            elif j['state'] not in RESIDENT|{'ARCHIVED'} or not j['report'] or not j['report']['accepted']:
                raise ContractError('global repair not admitted')
            self.db.execute('UPDATE global_repair_attempts_v08 SET completed=1 WHERE job_id=?',(job_id,))
            self.db.commit()
            return True
