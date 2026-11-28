"""Bounded analytical affine compiler and immutable, lazy expert catalog.
No executable manifests, untrusted generator execution or automatic promotion.
"""
from pathlib import Path
import hashlib, json, sqlite3, math, uuid
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from .abi import ContractError
from .environment import identifier, Signal


def case_digest(cases):
    return hashlib.sha256(json.dumps(cases, separators=(',',':'), allow_nan=False).encode()).hexdigest()

def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else _digest(f)
def _digest(f):
    h=hashlib.sha256()
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def quadratic_width(n):
    if type(n) is not int or not 1<=n<=16:raise ContractError('quadratic input width')
    return n+n*(n+1)//2


def quadratic_features(x):
    """Fixed numeric ABI: linear terms followed by upper-triangle products."""
    n=x.shape[-1];quadratic_width(n)
    return torch.cat((x,torch.stack([x[...,i]*x[...,j] for i in range(n) for j in range(i,n)],dim=-1)),dim=-1)


class Catalog:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'catalog.sqlite',timeout=2)
        self.db.execute('CREATE TABLE IF NOT EXISTS experts(id TEXT PRIMARY KEY, family TEXT NOT NULL, manifest TEXT NOT NULL, state TEXT NOT NULL)')
    def manifest(self,id):
        row=self.db.execute('SELECT manifest,state FROM experts WHERE id=?',(id,)).fetchone()
        if row is None: raise ContractError('unknown instance')
        return dict(json.loads(row[0]),state=row[1])
    def list(self): return [dict(json.loads(m),state=s) for m,s in self.db.execute('SELECT manifest,state FROM experts ORDER BY id')]
    def stage_affine(self,id,family,matrix,bias,*,source_topic,target_scope,target_topic,parents=(),origin="DIRECT",learning_job=None,group_job=None):
        return self._stage_numeric(id,family,matrix,bias,source_topic=source_topic,target_scope=target_scope,
            target_topic=target_topic,parents=parents,origin=origin,learning_job=learning_job,group_job=group_job)

    def stage_quadratic(self,id,family,matrix,bias,*,input_width,source_topic,target_scope,target_topic,
                        parents=(),origin="DIRECT",group_job=None):
        quadratic_width(input_width)
        return self._stage_numeric(id,family,matrix,bias,source_topic=source_topic,target_scope=target_scope,
            target_topic=target_topic,parents=parents,origin=origin,group_job=group_job,op='quadratic',input_width=input_width)

    def _stage_numeric(self,id,family,matrix,bias,*,source_topic,target_scope,target_topic,parents=(),
                       origin="DIRECT",learning_job=None,group_job=None,op='affine',input_width=None):
        if op not in ('affine','quadratic') or (op=='quadratic' and learning_job is not None):
            raise ContractError('numeric core operation')
        for s in (id,family,source_topic,target_scope,target_topic): identifier(s)
        if origin not in ("DIRECT","TRAINED") or (origin=="TRAINED" and learning_job is None and group_job is None): raise ContractError("learning origin")
        if learning_job is not None:
            identifier(learning_job)
            row=self.db.execute("SELECT expert_id,method FROM knowledge_jobs_v08 WHERE id=?",(learning_job,)).fetchone()
            if row!=(id,origin): raise ContractError("learning job mismatch")
        if group_job is not None:
            identifier(group_job)
            if learning_job is not None: raise ContractError('group origin')
            row=self.db.execute('SELECT expert_id,spec FROM group_learning_jobs_v08 WHERE id=?',(group_job,)).fetchone()
            if row is None or row[0]!=id: raise ContractError('group learning job mismatch')
            spec=json.loads(row[1])
            contract=('group-quadratic-v1' if op=='quadratic' and origin=='TRAINED' else
                      {'DIRECT':'group-direct-v1','TRAINED':'group-trained-v1'}[origin])
            if op=='quadratic' and (origin!='TRAINED' or spec.get('input_width')!=input_width):
                raise ContractError('quadratic group job mismatch')
            if spec['parents']!=list(parents) or spec['contract']!=contract:
                raise ContractError('group learning job mismatch')
            if origin=='TRAINED' and self.db.execute('SELECT 1 FROM group_training_builds_v08 WHERE job_id=?',(group_job,)).fetchone() is None:
                raise ContractError('trained group build required')
        if type(parents) not in (tuple,list) or len(set(parents))!=len(parents): raise ContractError('parents')
        for p in parents: self.manifest(p)
        w=torch.tensor(matrix,dtype=torch.float32); b=torch.tensor(bias,dtype=torch.float32)
        if w.ndim!=2 or b.shape!=(w.shape[0],) or not w.numel() or max(w.shape)>4096 or not torch.isfinite(w).all() or not torch.isfinite(b).all(): raise ContractError('affine shape or values')
        if op=='quadratic' and (w.shape[1]!=quadratic_width(input_width) or w.numel()+b.numel()>65536):
            raise ContractError('quadratic parameter contract')
        if self.db.execute('SELECT 1 FROM experts WHERE id=?',(id,)).fetchone(): raise ContractError('duplicate instance')
        # Unique staging path; artifacts named by content digest, not by arbitrary manifest paths.
        temp=self.root/(uuid.uuid4().hex+'.tmp')
        try:
            metadata={'vibeweights_contract':f'quadratic-upper-v1:{input_width}'} if op=='quadratic' else None
            save_file({'w':w,'b':b},str(temp),metadata=metadata); sha=digest(temp); path=self.root/(sha+'.safetensors')
            if path.exists():
                if digest(path)!=sha: raise ContractError('corrupt existing artifact')
            else: temp.replace(path)
            m={'id':id,'family':family,'parents':list(parents),'origin':origin,'abi':'numeric-events/0.1','op':op,'shape':list(w.shape) if op=='affine' else [w.shape[0],input_width],'parameters':w.numel()+b.numel(),'sha256':sha,'bytes':path.stat().st_size,'source_topic':source_topic,'target_scope':target_scope,'target_topic':target_topic}
            if learning_job is not None: m["learning_job"]=learning_job
            if group_job is not None: m["group_job"]=group_job
            with self.db: self.db.execute('INSERT INTO experts VALUES(?,?,?,?)',(id,family,json.dumps(m,sort_keys=True),'STAGED'))
            return self.manifest(id)
        finally: temp.unlink(missing_ok=True)
    def load(self,id,*,byte_budget=64*1024*1024,experimental=False,device="cpu"):
        m=self.manifest(id)
        if m['state'] not in ('ACTIVE','LOW_ACTIVE_IMPORTANT') and not experimental: raise ContractError('candidate not admitted')
        if type(byte_budget) is not int or byte_budget<1: raise ContractError('byte budget')
        path=self.root/(m['sha256']+'.safetensors')
        if not path.is_file() or path.stat().st_size!=m['bytes'] or m['bytes']>byte_budget or digest(path)!=m['sha256']: raise ContractError('artifact budget or integrity')
        shape=m['shape'];op=m['op']
        if m['abi']!='numeric-events/0.1' or op not in ('affine','quadratic') or type(shape) is not list or len(shape)!=2 or any(type(n) is not int or not 1<=n<=4096 for n in shape):
            raise ContractError('numeric core ABI')
        expected=shape if op=='affine' else [shape[0],quadratic_width(shape[1])]
        with safe_open(str(path),framework='pt',device='cpu') as artifact:
            metadata=artifact.metadata() or {}
            expected_metadata={'vibeweights_contract':f'quadratic-upper-v1:{shape[1]}'} if op=='quadratic' else {}
            if metadata!=expected_metadata:raise ContractError('numeric artifact operation binding')
            if set(artifact.keys())!={'w','b'}:raise ContractError('tensor keys')
            w,b=artifact.get_tensor('w'),artifact.get_tensor('b')
        if (list(w.shape)!=expected or b.shape!=(shape[0],) or m['parameters']!=w.numel()+b.numel() or
            (op=='quadratic' and m['parameters']>65536) or
            any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in (w,b))):
            raise ContractError('tensor contract')
        cls=AffineExpert if op=='affine' else QuadraticExpert
        return cls(m,w.to(device),b.to(device))
    def admit(self,id,cases,*,learning_job=None,group_job=None):
        """Host-supplied independent expected pairs. Limited, not proof of general competence."""
        m=self.manifest(id)
        if m['state']!='STAGED': raise ContractError('not staged')
        cases=tuple(cases)
        if not cases: raise ContractError('empty evaluation')
        if "learning_job" in m:
            if learning_job!=m["learning_job"]: raise ContractError("use knowledge learning promotion")
            row=self.db.execute("SELECT passed,case_sha,artifact_sha FROM knowledge_reports_v08 WHERE job_id=?",(m["learning_job"],)).fetchone()
            if row!=(1,case_digest(cases),m["sha256"]): raise ContractError("learning evaluation receipt required")
        if 'group_job' in m:
            if group_job!=m['group_job']: raise ContractError('use group learning promotion')
            row=self.db.execute('SELECT passed,case_sha,artifact_sha FROM group_learning_reports_v08 WHERE job_id=?',(group_job,)).fetchone()
            if row!=(1,case_digest(cases),m['sha256']): raise ContractError('group evaluation receipt required')
        e=self.load(id,experimental=True)
        for values,expected in cases:
            actual=e.compute(tuple(values)); expected=tuple(expected)
            if len(actual)!=len(expected) or any(not math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6) for a,b in zip(actual,expected)): return False
        with self.db:
            if self.db.execute('UPDATE experts SET state=? WHERE id=? AND state=?',('ACTIVE',id,'STAGED')).rowcount!=1:
                raise ContractError('stale admission')
        return True
    def compose_affine(self,id,family,first,second,*,byte_budget=64*1024*1024,group_job=None):
        """Analytic fusion of compatible affine maps; candidate still requires admission.
        Original experts remain untouched. Composition is not arbitrary weight merging.
        """
        a=self.load(first,byte_budget=byte_budget); b=self.load(second,byte_budget=byte_budget)
        if a.m['op']!='affine' or b.m['op']!='affine':raise ContractError('affine composition requires affine cores')
        if a.m['target_topic']!=b.m['source_topic'] or a.w.shape[0]!=b.w.shape[1]: raise ContractError('incompatible composition')
        if a.m['bytes']+b.m['bytes']>byte_budget: raise ContractError('composition input budget')
        w=b.w@a.w; bias=b.w@a.b+b.b
        return self.stage_affine(id,family,w.tolist(),bias.tolist(),source_topic=a.m['source_topic'],target_scope=b.m['target_scope'],target_topic=b.m['target_topic'],parents=(first,second),group_job=group_job)
    def close(self): self.db.close()

class AffineExpert:
    def __init__(self,m,w,b): self.id=m['id']; self.m=m; self.w=w; self.b=b
    def relevant(self,snapshot): return any(s.topic==self.m['source_topic'] for s in snapshot)
    @torch.inference_mode()
    def compute(self,values):
        if len(values)!=self.w.shape[1]: raise ContractError('input width')
        v=torch.tensor(values,dtype=torch.float32,device=self.w.device); y=torch.nn.functional.linear(v,self.w,self.b)
        if not torch.isfinite(y).all(): raise ContractError('nonfinite output')
        return tuple(y.tolist())
    def forward(self,snapshot):
        for s in snapshot:
            if s.topic==self.m['source_topic']:
                yield Signal(self.m['target_scope'],self.m['target_topic'],self.compute(s.values))

class QuadraticExpert(AffineExpert):
    """Independent nonlinear numeric core with fixed feature basis and learned readout."""
    @torch.inference_mode()
    def compute(self,values):
        if len(values)!=self.m['shape'][1]:raise ContractError('input width')
        x=torch.tensor(values,dtype=torch.float32,device=self.w.device)
        y=torch.nn.functional.linear(quadratic_features(x),self.w,self.b)
        if not torch.isfinite(y).all():raise ContractError('nonfinite output')
        return tuple(y.tolist())
