"""Evidence-bound additive DIRECT composition of a two-member affine group."""
import json
import math
from .abi import ContractError
from .environment import Signal, identifier
from .factory import case_digest
from .group_work import GroupWork
from .knowledge_learning import _hash
from .population import RESIDENT


def _same(a,b):
    return len(a)==len(b) and all(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-6) for x,y in zip(a,b))


class GroupLearning:
    def __init__(self, work):
        if not isinstance(work,GroupWork): raise ContractError('group work required')
        self.work=work; self.learning=work.learning; self.db=work.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS group_learning_jobs_v08(
          id TEXT PRIMARY KEY, expert_id TEXT UNIQUE NOT NULL, spec TEXT NOT NULL, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS group_learning_reports_v08(
          job_id TEXT PRIMARY KEY, passed INTEGER NOT NULL, case_sha TEXT NOT NULL,
          artifact_sha TEXT NOT NULL, receipt TEXT NOT NULL);
        ''')

    def job(self, job):
        identifier(job)
        row=self.db.execute('SELECT expert_id,spec,sha FROM group_learning_jobs_v08 WHERE id=?',(job,)).fetchone()
        if row is None: raise ContractError('unknown group learning job')
        spec=json.loads(row[1])
        if _hash(spec)!=row[2] or spec['expert']!=row[0]: raise ContractError('group job integrity')
        report=self.db.execute('SELECT receipt FROM group_learning_reports_v08 WHERE job_id=?',(job,)).fetchone()
        return {'spec':spec,'report':None if report is None else json.loads(report[0])}

    def _live(self,spec):
        w=self.work; l=self.learning
        event=w.bus.event(spec['event'])
        if not w.bus._live(event,w.bus._conflicts()): raise ContractError('stale group event')
        members=next((g['experts'] for g in w.bus.groups(spec['event'])
                    if (g['target'],g['group'])==(spec['target'],spec['group'])),())
        if set(members)!=set(spec['parents']): raise ContractError('group changed')
        for b in spec['members']:
            m=l.catalog.manifest(b['expert'])
            row=self.db.execute('''SELECT h.id,h.sha FROM perception_bindings_v08 b
                JOIN perception_heads_v08 h ON h.id=b.head WHERE b.expert=?''',(b['expert'],)).fetchone()
            if m['state'] not in RESIDENT or m['sha256']!=b['core_sha256'] or row!=(b['head'],b['head_sha256']) or (spec['target'],spec['group']) not in w.heads.propose(b['expert'],event):
                raise ContractError('group member changed')
        for ref in spec['runs']:
            if _hash(w.receipt(ref['id']))!=ref['sha256']: raise ContractError('group evidence changed')

    def stage(self,job,expert,runs,*,family):
        """Require multiple successful runs of the same unambiguous affine chain."""
        l=self.learning; w=self.work
        for x in (job,expert,family): identifier(x)
        if type(runs) not in (tuple,list) or not 2<=len(runs)<=64 or len(set(runs))!=len(runs): raise ContractError('group evidence count')
        with l._guard():
            receipts=[w.receipt(r) for r in sorted(runs)]
            base=receipts[0]; members=base['members']; inputs=base['inputs']; goal=base['goal']
            if len(members)!=2 or len(inputs)!=1: raise ContractError('two-member single-input group required')
            source=next(iter(inputs)); ms=[l.catalog.manifest(b['expert']) for b in members]
            chains=[(a,b) for a,b in (ms,ms[::-1]) if a['source_topic']==source and a['target_topic']==b['source_topic'] and b['target_topic']==goal and len({source,a['target_topic'],goal})==3 and a['shape'][0]==b['shape'][1]]
            if len(chains)!=1: raise ContractError('unambiguous affine chain required')
            a,b=chains[0]; parents=[a['id'],b['id']]; seen=[]
            for r in receipts:
                if any(r[k]!=base[k] for k in ('event','target','group','members','goal','oracle_id','validator_topics')) or set(r['inputs'])!={source}: raise ContractError('incompatible group evidence')
                out=r['result']; x=r['inputs'][source]
                if out['status']!='OK' or out['faults'] or out['value'] is None or {t[1] for t in out['trace']}!=set(parents): raise ContractError('successful full group evidence required')
                facts={f['topic']:f['values'] for f in out['facts'] if f['scope']==r['scope']}
                mid=l.population.compute(a['id'],x); y=l.population.compute(b['id'],mid)
                if not _same(mid,facts.get(a['target_topic'],[])) or not _same(y,out['value']): raise ContractError('group evidence replay mismatch')
                if tuple(x) in [tuple(v[0]) for v in seen]: raise ContractError('distinct evidence inputs required')
                seen.append([x,out['value']])
            spec={'contract':'group-direct-v1','expert':expert,'family':family,'parents':parents,'members':members,
                  'event':base['event'],'target':base['target'],'group':base['group'],'source':source,'goal':goal,
                  'runs':[{'id':r['run'],'sha256':_hash(r)} for r in receipts], 'observations':seen,
                  'input_width':a['shape'][1],'output_width':b['shape'][0]}
            self._live(spec)
            raw=json.dumps(spec,sort_keys=True)
            prior=self.db.execute('SELECT expert_id,spec,sha FROM group_learning_jobs_v08 WHERE id=? OR expert_id=?',(job,expert)).fetchall()
            if prior:
                if prior!=[(expert,raw,_hash(spec))]: raise ContractError('group job identity reserved')
            else:
                if any(m['id']==expert for m in l.catalog.list()) or self.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job,expert)).fetchone() or l.knowledge.db.execute('SELECT 1 FROM learning_intakes_v08 WHERE job_id=? OR expert_id=?',(job,expert)).fetchone(): raise ContractError('group identity reserved')
                with self.db: self.db.execute('INSERT INTO group_learning_jobs_v08 VALUES(?,?,?,?)',(job,expert,raw,_hash(spec)))
            if any(m['id']==expert for m in l.catalog.list()):
                m=l.catalog.manifest(expert)
                if m.get('group_job')!=job or m['parents']!=parents: raise ContractError('group artifact mismatch')
                return m
            return l.catalog.compose_affine(expert,family,*parents,byte_budget=l.population.artifact_budget,group_job=job)

    def promote(self,job,cases,*,oracle_id):
        """Separate independent holdout evaluation; no seen input may be reused."""
        identifier(oracle_id); l=self.learning
        with l._guard():
            spec=self.job(job)['spec']; self._live(spec)
            m=l.catalog.manifest(spec['expert'])
            if m.get('group_job')!=job or m['parents']!=spec['parents']: raise ContractError('group artifact mismatch')
            if type(cases) not in (list,tuple) or not 3<=len(cases)<=256: raise ContractError('group holdout count')
            seen={tuple(x) for x,_ in spec['observations']}; normalized=[]
            for x,y in cases:
                x=Signal('global',spec['source'],tuple(x)).values; y=Signal('global',spec['goal'],tuple(y)).values
                if len(x)!=spec['input_width'] or len(y)!=spec['output_width'] or x in seen: raise ContractError('distinct unseen holdout required')
                seen.add(x); normalized.append((x,y))
            e=l.catalog.load(m['id'],experimental=True,byte_budget=l.population.artifact_budget)
            observed=sum(_same(e.compute(x),y) for x,y in spec['observations'])
            heldout=sum(_same(e.compute(x),y) for x,y in normalized)
            accepted=observed==len(spec['observations']) and heldout==len(normalized)
            report={'oracle_id':oracle_id,'case_sha256':case_digest(normalized),'artifact_sha256':m['sha256'],
                    'observed_passed':observed,'observed_cases':len(spec['observations']),'holdout_passed':heldout,'holdout_cases':len(normalized),'accepted':accepted}
            row=(int(accepted),report['case_sha256'],m['sha256'],json.dumps(report,sort_keys=True))
            prior=self.db.execute('SELECT passed,case_sha,artifact_sha,receipt FROM group_learning_reports_v08 WHERE job_id=?',(job,)).fetchone()
            if prior is not None and prior!=row: raise ContractError('group evaluation changed; new job required')
            if prior is None:
                with self.db: self.db.execute('INSERT INTO group_learning_reports_v08 VALUES(?,?,?,?,?)',(job,*row))
            if not accepted: return False
            if m['state'] in RESIDENT: return True
            return l.population.admit(m['id'],normalized,group_job=job)
