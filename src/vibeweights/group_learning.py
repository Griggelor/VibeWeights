"""Evidence-bound additive DIRECT composition or TRAINED behavioral transfer."""
import json
import math
import torch
from .abi import ContractError
from .environment import Signal, identifier
from .factory import case_digest, quadratic_width, quadratic_features
from .group_work import GroupWork
from .knowledge_learning import _hash
from .population import RESIDENT


def _same(a,b):
    return len(a)==len(b) and all(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-6) for x,y in zip(a,b))


def train_affine(observations, steps):
    """Fit from input/output pairs alone, without parent tensors or holdouts.

    Float64 CPU SGD on standardized inputs; export an independent FP32 affine
    map. Full affine rank is required to avoid unidentifiable coefficients.
    """
    x=torch.tensor([v[0] for v in observations],dtype=torch.float64,device='cpu')
    y=torch.tensor([v[1] for v in observations],dtype=torch.float64,device='cpu')
    if x.shape[1]+1>len(x) or (x.shape[1]+1)*y.shape[1]>65536:
        raise ContractError('trained group dimension budget')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or max(x.abs().max().item(),y.abs().max().item())>1e6:
        raise ContractError('trained group sample bounds')
    center=x.mean(0); scale=(x-center).abs().amax(0)
    if (scale==0).any(): raise ContractError('trained group rank')
    z=(x-center)/scale
    design=torch.cat((z,torch.ones((len(z),1),dtype=z.dtype)),dim=1)
    if int(torch.linalg.matrix_rank(design))!=design.shape[1]: raise ContractError('trained group rank')
    # Fixed zero initialization; a bounded optimization run, not a closed-form fit.
    weights=torch.zeros((design.shape[1],y.shape[1]),dtype=z.dtype,requires_grad=True)
    lr=0.5/float(design.square().sum()/len(z))
    optimizer=torch.optim.SGD([weights],lr=lr)
    def loss(): return (design@weights-y).square().sum()/(2*len(z))
    initial=float(loss().detach())
    for _ in range(steps):
        optimizer.zero_grad(); value=loss(); value.backward(); optimizer.step()
    final=float(loss().detach())
    if not math.isfinite(final): raise ContractError('trained group nonfinite loss')
    with torch.no_grad():
        w=(weights[:-1]/scale[:,None]).T; b=weights[-1]-w@center
        matrix=w.float(); bias=b.float()
    return matrix.tolist(),bias.tolist(),{'initial_loss':initial,'final_loss':final,'steps':steps,
        'learning_rate':lr,'optimizer':'SGD','initialization':'zero','training_dtype':'float64',
        'export_dtype':'float32','dataset_sha256':_hash(observations),'torch_version':str(torch.__version__)}


def train_quadratic(observations,steps):
    """Train readout from observations only; no parent weights or holdout labels."""
    x=torch.tensor([v[0] for v in observations],dtype=torch.float64,device='cpu')
    if x.ndim!=2 or not torch.isfinite(x).all() or x.abs().max().item()>1000:
        raise ContractError('quadratic training sample bounds')
    width=quadratic_width(x.shape[1])
    if width+1>len(x):raise ContractError('quadratic training rank')
    z=quadratic_features(x).tolist()
    w,b,stats=train_affine([[v,y] for v,(_,y) in zip(z,observations)],steps)
    stats.update(dataset_sha256=_hash(observations),feature_contract='quadratic-upper-v1',input_width=x.shape[1],feature_width=width)
    return w,b,stats


def train_mlp(observations,steps,hidden_width):
    """Bounded full-layer optimization, deterministic CPU initialization, no teacher tensors."""
    x=torch.tensor([v[0] for v in observations],dtype=torch.float64,device='cpu')
    y=torch.tensor([v[1] for v in observations],dtype=torch.float64,device='cpu')
    if (x.ndim!=2 or y.ndim!=2 or len(x)<4 or not 1<=x.shape[1]<=64 or not 1<=y.shape[1]<=64 or
        type(hidden_width) is not int or not 1<=hidden_width<=64 or type(steps) is not int or not 1<=steps<=256):
        raise ContractError('MLP training dimensions/budget')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or max(x.abs().max().item(),y.abs().max().item())>1000:
        raise ContractError('MLP training sample bounds')
    g=torch.Generator(device='cpu').manual_seed(7)
    def initial(shape):return (2*torch.rand(shape,generator=g,dtype=torch.float64)-1).requires_grad_()
    hw=initial((hidden_width,x.shape[1]));hb=initial((hidden_width,))
    w=initial((y.shape[1],hidden_width));b=torch.zeros(y.shape[1],dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([hw,hb,w,b],lr=1,max_iter=1,max_eval=10,tolerance_grad=1e-14,
                          tolerance_change=1e-18,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        pred=torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x,hw,hb)),w,b)
        loss=(pred-y).square().mean()
        loss.backward()
        return loss
    before=float(closure().detach())
    for _ in range(steps):opt.step(closure)
    after=float(closure().detach())
    if not math.isfinite(after) or any(not torch.isfinite(t).all() for t in (w,b,hw,hb)):
        raise ContractError('MLP nonfinite training')
    stats={'initial_loss':before,'final_loss':after,'steps':steps,'optimizer':'LBFGS',
           'initialization':'uniform-seed-7','hidden_width':hidden_width,'activation':'relu',
           'learning_rate':1.0,'training_dtype':'float64','export_dtype':'float32',
           'dataset_sha256':_hash(observations),'torch_version':str(torch.__version__)}
    return w.detach().float().tolist(),b.detach().float().tolist(),hw.detach().float().tolist(),hb.detach().float().tolist(),stats


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
        CREATE TABLE IF NOT EXISTS group_training_builds_v08(
          job_id TEXT PRIMARY KEY, stats TEXT NOT NULL);
        ''')

    def job(self, job):
        identifier(job)
        row=self.db.execute('SELECT expert_id,spec,sha FROM group_learning_jobs_v08 WHERE id=?',(job,)).fetchone()
        if row is None: raise ContractError('unknown group learning job')
        spec=json.loads(row[1])
        if _hash(spec)!=row[2] or spec['expert']!=row[0]: raise ContractError('group job integrity')
        report=self.db.execute('SELECT receipt FROM group_learning_reports_v08 WHERE job_id=?',(job,)).fetchone()
        result={'spec':spec,'report':None if report is None else json.loads(report[0])}
        if spec['contract'] in ('group-trained-v1','group-quadratic-v1','group-mlp-v1'):
            stats=self.db.execute('SELECT stats FROM group_training_builds_v08 WHERE job_id=?',(job,)).fetchone()
            result['build']=None if stats is None else json.loads(stats[0])
        return result

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

    def stage(self,job,expert,runs,*,family,method='DIRECT',steps=256,hidden_width=None):
        """Require multiple successful runs of the same unambiguous affine chain."""
        l=self.learning; w=self.work
        for x in (job,expert,family): identifier(x)
        if method not in ('DIRECT','TRAINED','TRAINED_QUADRATIC','TRAINED_MLP') or type(steps) is not int or not 1<=steps<=1024:
            raise ContractError('group learning method/budget')
        if method=='TRAINED_MLP':
            if type(hidden_width) is not int or not 1<=hidden_width<=64 or steps>256:raise ContractError('MLP training dimensions/budget')
        elif hidden_width is not None:raise ContractError('hidden width requires MLP transfer')
        if type(runs) not in (tuple,list) or not 2<=len(runs)<=64 or len(set(runs))!=len(runs): raise ContractError('group evidence count')
        with l._guard():
            receipts=[w.receipt(r) for r in sorted(runs)]
            base=receipts[0]; members=base['members']; inputs=base['inputs']; goal=base['goal']
            if len(members)!=2 or len(inputs)!=1: raise ContractError('two-member single-input group required')
            source=next(iter(inputs)); ms=[l.catalog.manifest(b['expert']) for b in members]
            chains=[(a,b) for a,b in (ms,ms[::-1]) if a['source_topic']==source and a['target_topic']==b['source_topic'] and b['target_topic']==goal and len({source,a['target_topic'],goal})==3 and a['shape'][0]==b['shape'][1]]
            if len(chains)!=1: raise ContractError('unambiguous affine chain required')
            a,b=chains[0]; parents=[a['id'],b['id']]; seen=[]
            if method not in ('TRAINED_QUADRATIC','TRAINED_MLP') and any(m['op']!='affine' for m in (a,b)):
                raise ContractError('nonlinear chain requires nonlinear transfer')
            if method=='TRAINED_QUADRATIC':
                quadratic_width(a['shape'][1])
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
            if method!='DIRECT':
                spec.update(contract={'TRAINED_QUADRATIC':'group-quadratic-v1','TRAINED_MLP':'group-mlp-v1','TRAINED':'group-trained-v1'}[method],steps=steps,torch_version=str(torch.__version__),
                            training_oracle=base['oracle_id'],target_scope=b['target_scope'])
            if method=='TRAINED_MLP':spec['hidden_width']=hidden_width
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
                if m.get('group_job')!=job or m['parents']!=parents or m['origin']!=('DIRECT' if method=='DIRECT' else 'TRAINED') or m['op']!=({'TRAINED_QUADRATIC':'quadratic','TRAINED_MLP':'mlp'}.get(method,'affine')): raise ContractError('group artifact mismatch')
                return m
            if method=='DIRECT':
                return l.catalog.compose_affine(expert,family,*parents,byte_budget=l.population.artifact_budget,group_job=job)
            trainer=train_quadratic if method=='TRAINED_QUADRATIC' else train_affine
            if method=='TRAINED_MLP':matrix,bias,hw,hb,stats=train_mlp(spec['observations'],steps,hidden_width)
            else:matrix,bias,stats=trainer(spec['observations'],steps)
            encoded=json.dumps(stats,sort_keys=True)
            prior=self.db.execute('SELECT stats FROM group_training_builds_v08 WHERE job_id=?',(job,)).fetchone()
            if prior is not None and prior!=(encoded,): raise ContractError('group training build changed')
            if prior is None:
                with self.db: self.db.execute('INSERT INTO group_training_builds_v08 VALUES(?,?)',(job,encoded))
            if method=='TRAINED_MLP':
                return l.catalog.stage_mlp(expert,family,matrix,bias,hidden_matrix=hw,hidden_bias=hb,
                    source_topic=source,target_scope=spec['target_scope'],target_topic=goal,parents=parents,origin='TRAINED',group_job=job)
            if method=='TRAINED_QUADRATIC':
                return l.catalog.stage_quadratic(expert,family,matrix,bias,input_width=spec['input_width'],
                    source_topic=source,target_scope=spec['target_scope'],target_topic=goal,parents=parents,origin='TRAINED',group_job=job)
            return l.catalog.stage_affine(expert,family,matrix,bias,source_topic=source,target_scope=spec['target_scope'],
                target_topic=goal,parents=parents,origin='TRAINED',group_job=job)

    def promote(self,job,cases,*,oracle_id):
        """Separate independent holdout evaluation; no seen input may be reused."""
        identifier(oracle_id); l=self.learning
        with l._guard():
            spec=self.job(job)['spec']; self._live(spec)
            m=l.catalog.manifest(spec['expert'])
            if spec['contract'] in ('group-trained-v1','group-quadratic-v1','group-mlp-v1'):
                if oracle_id==spec['training_oracle']: raise ContractError('independent training holdout oracle required')
                if self.job(job)['build'] is None or m['origin']!='TRAINED': raise ContractError('trained group build required')
            if m.get('group_job')!=job or m['parents']!=spec['parents'] or m['op']!=({'group-quadratic-v1':'quadratic','group-mlp-v1':'mlp'}.get(spec['contract'],'affine')): raise ContractError('group artifact mismatch')
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
