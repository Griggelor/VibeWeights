"""Bounded knowledge-first vertical slice: associative DIRECT and SGD weights.

Trusted host supplies current verified records and an independently defined oracle.
No external model, natural-language semantics or autonomous authority is implied.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
import math
import time
import torch
from .abi import ContractError
from .environment import identifier
from .factory import case_digest
from .dependency_index import DependencyIndex


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Domain:
    topic: str
    entities: int
    labels: int

    def __post_init__(self):
        identifier(self.topic)
        if type(self.entities) is not int or not 1<=self.entities<=128:
            raise ContractError('entity bound')
        if type(self.labels) is not int or not 2<=self.labels<=32:
            raise ContractError('label bound')
        identifier(self.topic+':query'); identifier(self.topic+':answer'); identifier(self.topic+':entity:127')

    def encode(self, entity):
        # Identity only; no answer table or retrieval. Out-of-domain => zero.
        if type(entity) is not int: raise ContractError('entity identity')
        return tuple(float(entity==i) for i in range(self.entities))

    def target(self, label):
        if label is not None and (type(label) is not int or not 0<=label<self.labels):
            raise ContractError('label identity')
        return tuple(float(label==i) for i in range(self.labels))

    def decode(self, scores):
        if len(scores)!=self.labels or any(not math.isfinite(x) for x in scores):
            raise ContractError('invalid scores')
        # No extrapolation/argmax guess on ambiguous outputs.
        for i in range(self.labels):
            if all(abs(a-b)<=1e-6 for a,b in zip(scores,self.target(i))): return i
        return None


class KnowledgeLearning:
    """Persistent job/evaluation ledger in the population catalog DB.

    One trusted owner. The separate knowledge DB is write-locked during source
    validation, live inference, and selection. Offline population instances remain
    historical snapshots; no sandbox for a malicious host.
    """
    def __init__(self, population, knowledge, access):
        if population._source_owner is not None:
            raise ContractError('population already has a source owner')
        self.population=population; self.catalog=population.catalog
        self.knowledge=knowledge; self.access=access; self.db=self.catalog.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS knowledge_jobs_v08(
          id TEXT PRIMARY KEY, expert_id TEXT NOT NULL UNIQUE, method TEXT NOT NULL,
          spec TEXT NOT NULL, created_ns INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_reports_v08(
          job_id TEXT PRIMARY KEY, passed INTEGER NOT NULL, case_sha TEXT NOT NULL,
          artifact_sha TEXT NOT NULL, report TEXT NOT NULL, created_ns INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_builds_v08(
          job_id TEXT PRIMARY KEY, stats TEXT NOT NULL);
        ''')
        self.knowledge.db.executescript('''
        CREATE TABLE IF NOT EXISTS learning_intakes_v08(
          job_id TEXT PRIMARY KEY, expert_id TEXT NOT NULL UNIQUE,
          spec TEXT NOT NULL, created_ns INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS conflict_cases_v08(
          task TEXT PRIMARY KEY, topic TEXT NOT NULL, entity INTEGER NOT NULL,
          receipt TEXT NOT NULL, uncertain_id INTEGER NOT NULL,
          resolution TEXT, closed INTEGER NOT NULL DEFAULT 0);
        CREATE UNIQUE INDEX IF NOT EXISTS conflict_open_v08 ON conflict_cases_v08(topic,entity) WHERE closed=0;
        ''')
        self.dependencies=DependencyIndex(self)
        self._locked_depth=0
        self._guard_depth=0
        population._source_owner=self
        population._source_guard=self._guard

    @contextmanager
    def _locked(self):
        db=self.knowledge.db
        outer=self._locked_depth==0
        if outer: db.execute('BEGIN IMMEDIATE')
        self._locked_depth+=1
        try: yield
        finally:
            self._locked_depth-=1
            if outer: db.rollback()

    @contextmanager
    def _guard(self):
        with self._locked():
            if self._guard_depth==0: self._audit_locked()
            self._guard_depth+=1
            try: yield
            finally: self._guard_depth-=1

    def _snapshot(self, roots, domain):
        roots=tuple(roots)
        if not roots or len(roots)>domain.entities or len(set(roots))!=len(roots):
            raise ContractError('learning record count')
        records={}; examples={}
        pending=list(roots)
        while pending:
            rid=pending.pop()
            if rid in records: continue
            if len(records)>=512: raise ContractError('source dependency bound')
            rec=self.knowledge.read(rid,self.access)
            head=self.knowledge.db.execute('SELECT id FROM knowledge_heads_v1 WHERE scope=? AND key=?',(rec.scope,rec.key)).fetchone()
            if rec.scope!='global' or rec.status!='VERIFIED' or head!=(rid,):
                raise ContractError('current verified global knowledge required')
            records[rid]=rec
            pending.extend(rec.parents)
        for rid in roots:
            rec=records[rid]
            if rec.topic!=domain.topic or rec.kind!='semantic' or not rec.key.startswith(domain.topic+':entity:') or len(rec.values)!=1:
                raise ContractError('association record contract')
            try: entity=int(rec.key.rsplit(':',1)[1])
            except (ValueError, IndexError): raise ContractError('entity key') from None
            label=rec.values[0]
            if rec.key!=f'{domain.topic}:entity:{entity}' or not 0<=entity<domain.entities or type(label) is not int or not 0<=label<domain.labels or entity in examples:
                raise ContractError('association value contract')
            examples[entity]=label
        refs=[{'id':i,'sha256':_hash(asdict(records[i])),'source':records[i].source} for i in sorted(records)]
        return examples,refs

    def prepare(self, job_id, expert_id, domain, record_ids, *, method, steps=80, parents=(), derivation=None):
        """Stage only; record exact source revisions and dataset hash, not answers."""
        identifier(job_id); identifier(expert_id)
        if method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256:
            raise ContractError('learning method/budget')
        ids=tuple(record_ids)
        parents=tuple(parents)
        if len(set(parents))!=len(parents): raise ContractError('duplicate expert parents')
        for parent in parents:
            if self.catalog.manifest(parent)['state'] not in ('ACTIVE','LOW_ACTIVE_IMPORTANT'):
                raise ContractError('learning parent not resident')
        examples,refs=self._snapshot(ids,domain)
        spec={'domain':asdict(domain),'roots':ids,'sources':refs,
              'dataset_sha256':_hash(sorted(examples.items())),
              'steps':steps if method=='TRAINED' else 0,'optimizer':'SGD' if method=='TRAINED' else None,
              'learning_rate':0.25 if method=='TRAINED' else None,
              'torch_version':str(torch.__version__),'initialization':'zero','contract':'association-v1',
              'expert_parents':parents,'derivation':derivation}
        pending=self.knowledge.db.execute('SELECT job_id,expert_id,spec FROM learning_intakes_v08 WHERE job_id=? OR expert_id=?',(job_id,expert_id)).fetchall()
        if pending:
            expected={'domain':asdict(domain),'roots':list(ids),'sources':refs,
                      'dataset_sha256':spec['dataset_sha256'],'method':method,'steps':steps,
                      'expert_parents':list(parents),'derivation':derivation}
            if pending!=[(job_id,expert_id,json.dumps(expected,sort_keys=True))]:
                raise ContractError('learning identity reserved by intake')
        if self.db.execute('SELECT 1 FROM experts WHERE id=?',(expert_id,)).fetchone() or self.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job_id,expert_id)).fetchone():
            raise ContractError('duplicate learning identity')
        with self.db:
            self.db.execute('INSERT INTO knowledge_jobs_v08 VALUES(?,?,?,?,?)',(job_id,expert_id,method,json.dumps(spec,sort_keys=True),time.time_ns()))
        return self.resume_build(job_id)

    def resume_build(self, job_id):
        """Explicit restart of an incomplete build; never promotes a candidate."""
        with self._guard():
            job=self.job(job_id)
            if job['state']!='INCOMPLETE' or job['report'] is not None:
                raise ContractError('job is not an incomplete build')
            domain=self._check_sources(job)
            spec=job['spec']; method=job['method']; expert_id=job['expert_id']
            steps=spec['steps']; parents=tuple(spec['expert_parents'])
            if spec['torch_version']!=str(torch.__version__):
                raise ContractError('recovery requires original torch version')
            for parent in parents:
                if self.catalog.manifest(parent)['state'] not in ('ACTIVE','LOW_ACTIVE_IMPORTANT'):
                    raise ContractError('learning parent not resident')
            examples,_=self._snapshot(spec['roots'],domain)
            return self._build(job_id,expert_id,domain,examples,spec,method,steps,parents)

    def _build(self, job_id, expert_id, domain, examples, spec, method, steps, parents):
        w=torch.zeros((domain.labels,domain.entities),dtype=torch.float32)
        b=torch.zeros(domain.labels,dtype=torch.float32)
        # Sorted independent identity vectors, target labels only in this learning process.
        x=torch.tensor([domain.encode(i) for i in sorted(examples)],dtype=torch.float32)
        y=torch.tensor([domain.target(examples[i]) for i in sorted(examples)],dtype=torch.float32)
        initial=float(y.square().sum()/2)
        if method=='DIRECT':
            for entity,label in examples.items(): w[label,entity]=1
        else:
            w.requires_grad_(True)
            opt=torch.optim.SGD([w],lr=0.25)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss=(x@w.T-y).square().sum()/2
                loss.backward(); opt.step()
        w=w.detach()
        final=float((x@w.T-y).square().sum()/2)
        stats={'initial_loss':initial,'final_loss':final,'optimizer_steps':spec['steps'],'examples':len(examples)}
        encoded=json.dumps(stats,sort_keys=True)
        prior=self.db.execute('SELECT stats FROM knowledge_builds_v08 WHERE job_id=?',(job_id,)).fetchone()
        if prior is not None and prior!=(encoded,):
            raise ContractError('recovered build statistics differ')
        if prior is None:
            with self.db:
                self.db.execute('INSERT INTO knowledge_builds_v08 VALUES(?,?)',(job_id,encoded))
        return self.catalog.stage_affine(expert_id,domain.topic,w.tolist(),b.tolist(),
            source_topic=domain.topic+':query',target_scope='global',target_topic=domain.topic+':answer',
            origin=method,learning_job=job_id,parents=parents)


    def record_intake(self, *, job_id, expert_id, domain, ids, method, steps, parents=(), derivation=None):
        """Append a durable intent in the SAME transaction as verified facts."""
        if not self.knowledge.db.in_transaction or not self._locked_depth:
            raise ContractError('intake requires guarded source transaction')
        identifier(job_id); identifier(expert_id)
        if method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256:
            raise ContractError('learning method/budget')
        examples,refs=self._snapshot(ids,domain)
        spec={'domain':asdict(domain),'roots':list(ids),'sources':refs,
              'dataset_sha256':_hash(sorted(examples.items())), 'method':method,
              'steps':steps,'expert_parents':list(parents),'derivation':derivation}
        self.knowledge.db.execute('INSERT INTO learning_intakes_v08 VALUES(?,?,?,?)',
            (job_id,expert_id,json.dumps(spec,sort_keys=True),time.time_ns()))

    def intakes(self):
        """List persisted intake intents with current catalog state (no mutation)."""
        result=[]
        for job,expert,spec in self.knowledge.db.execute('SELECT job_id,expert_id,spec FROM learning_intakes_v08 ORDER BY job_id'):
            exists=self.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=?',(job,)).fetchone()
            result.append({'job_id':job,'expert_id':expert,'spec':json.loads(spec),
                           'state':self.job(job)['state'] if exists else 'PENDING'})
        return result

    def resume_intake(self, job_id):
        """Resume a committed fact batch; idempotent after staging, no promotion."""
        with self._guard():
            row=self.knowledge.db.execute('SELECT expert_id,spec FROM learning_intakes_v08 WHERE job_id=?',(job_id,)).fetchone()
            if row is None: raise ContractError('unknown intake')
            expert,raw=row; spec=json.loads(raw); d=Domain(**spec['domain'])
            examples,refs=self._snapshot(spec['roots'],d)
            if refs!=spec['sources'] or _hash(sorted(examples.items()))!=spec['dataset_sha256']:
                raise ContractError('intake sources changed')
            exists=self.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=?',(job_id,)).fetchone()
            if not exists:
                return self.prepare(job_id,expert,d,spec['roots'],method=spec['method'],
                    steps=spec['steps'],parents=tuple(spec['expert_parents']),derivation=spec['derivation'])
            j=self.job(job_id); js=j['spec']
            if (j['expert_id']!=expert or j['method']!=spec['method'] or
                js['steps']!=(spec['steps'] if spec['method']=='TRAINED' else 0) or
                any(js[key]!=spec[key] for key in ('domain','roots','sources','dataset_sha256','expert_parents','derivation'))):
                raise ContractError('intake/job mismatch')
            if j['state']=='INCOMPLETE': return self.resume_build(job_id)
            if j['state']=='QUARANTINED': raise ContractError('intake quarantined')
            self.catalog.load(expert,experimental=True,byte_budget=self.population.artifact_budget)
            return self.catalog.manifest(expert)

    def job(self, job_id):
        row=self.db.execute('SELECT expert_id,method,spec,created_ns FROM knowledge_jobs_v08 WHERE id=?',(job_id,)).fetchone()
        if row is None: raise ContractError('unknown learning job')
        expert,method,spec,created=row
        build=self.db.execute('SELECT stats FROM knowledge_builds_v08 WHERE job_id=?',(job_id,)).fetchone()
        report=self.db.execute('SELECT report FROM knowledge_reports_v08 WHERE job_id=?',(job_id,)).fetchone()
        m=self.db.execute('SELECT state FROM experts WHERE id=?',(expert,)).fetchone()
        return {'id':job_id,'expert_id':expert,'method':method,'spec':json.loads(spec),'created_ns':created,
                'build':None if build is None else json.loads(build[0]),
                'report':None if report is None else json.loads(report[0]),
                'state':'INCOMPLETE' if m is None else m[0]}

    def _check_sources(self, job):
        spec=job['spec']; domain=Domain(**spec['domain'])
        examples,refs=self._snapshot(spec['roots'],domain)
        if refs!=spec['sources'] or _hash(sorted(examples.items()))!=spec['dataset_sha256']:
            raise ContractError('learning source changed')
        return domain

    def promote(self, job_id, oracle, *, oracle_id):
        """Independent exhaustive bounded-domain evaluation; add only on success.

        Oracle returns an integer label or None for unknown; no output is passed
        into the forward call. Repeated evaluation must reproduce its receipt.
        """
        identifier(oracle_id)
        job=self.job(job_id)
        if job['state']!='STAGED' or job['build'] is None: raise ContractError('job not staged')
        with self._locked():
            d=self._check_sources(job)
            m=self.catalog.manifest(job['expert_id'])
            e=self.catalog.load(m['id'],experimental=True,byte_budget=self.population.artifact_budget)
            cases=[]; passed=0; known=unknown=0
            for i in range(-1,d.entities+1):
                expected=oracle(i) if 0<=i<d.entities else None
                x=d.encode(i); y=d.target(expected); actual=e.compute(x)
                cases.append((x,y)); known+=expected is not None; unknown+=expected is None
                passed+=int(all(math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6) for a,b in zip(actual,y)))
            report={'oracle_id':oracle_id,'case_sha256':case_digest(cases),'artifact_sha256':m['sha256'],
                    'cases':len(cases),'passed':passed,'known':known,'unknown':unknown,'accepted':passed==len(cases)}
            receipt=(int(report['accepted']),report['case_sha256'],m['sha256'],json.dumps(report,sort_keys=True))
            prior=self.db.execute('SELECT passed,case_sha,artifact_sha,report FROM knowledge_reports_v08 WHERE job_id=?',(job_id,)).fetchone()
            if prior is not None and prior!=receipt: raise ContractError('evaluation receipt changed; new job required')
            if prior is None:
                with self.db: self.db.execute('INSERT INTO knowledge_reports_v08 VALUES(?,?,?,?,?,?)',(job_id,*receipt,time.time_ns()))
            if not report['accepted']: return False
            return self.population.admit(m['id'],cases,learning_job=job_id)

    def audit_sources(self, *, full=False):
        """Explicit safety audit; quarantine invalid sources and all descendants.

        No evolutionary deletion/retirement. The live population automatically
        invokes this check before compute/run/admit/select. Offline artifacts cannot know
        about later corrections until synchronized; this is not automatic unlearning.
        """
        if type(full) is not bool: raise ContractError('full audit policy')
        with self._locked():
            return self.dependencies.audit(full=True) if full else self._audit_locked()

    def _audit_locked(self):
        return self.dependencies.audit()
