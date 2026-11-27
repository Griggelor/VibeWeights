"""Bounded analytical affine compiler and immutable, lazy expert catalog.
No executable manifests, untrusted generator execution or automatic promotion.
"""
from pathlib import Path
import hashlib, json, sqlite3, math, uuid
import torch
from safetensors.torch import save_file, load_file
from .abi import ContractError
from .environment import identifier, Signal


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else _digest(f)
def _digest(f):
    h=hashlib.sha256()
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

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
    def stage_affine(self,id,family,matrix,bias,*,source_topic,target_scope,target_topic,parents=()):
        for s in (id,family,source_topic,target_scope,target_topic): identifier(s)
        if type(parents) not in (tuple,list) or len(set(parents))!=len(parents): raise ContractError('parents')
        for p in parents: self.manifest(p)
        w=torch.tensor(matrix,dtype=torch.float32); b=torch.tensor(bias,dtype=torch.float32)
        if w.ndim!=2 or b.shape!=(w.shape[0],) or not w.numel() or max(w.shape)>4096 or not torch.isfinite(w).all() or not torch.isfinite(b).all(): raise ContractError('affine shape or values')
        if self.db.execute('SELECT 1 FROM experts WHERE id=?',(id,)).fetchone(): raise ContractError('duplicate instance')
        # Unique staging path; artifacts named by content digest, not by arbitrary manifest paths.
        temp=self.root/(uuid.uuid4().hex+'.tmp')
        try:
            save_file({'w':w,'b':b},str(temp)); sha=digest(temp); path=self.root/(sha+'.safetensors')
            if path.exists():
                if digest(path)!=sha: raise ContractError('corrupt existing artifact')
            else: temp.replace(path)
            m={'id':id,'family':family,'parents':list(parents),'origin':'DIRECT','abi':'numeric-events/0.1','op':'affine','shape':list(w.shape),'parameters':w.numel()+b.numel(),'sha256':sha,'bytes':path.stat().st_size,'source_topic':source_topic,'target_scope':target_scope,'target_topic':target_topic}
            with self.db: self.db.execute('INSERT INTO experts VALUES(?,?,?,?)',(id,family,json.dumps(m,sort_keys=True),'STAGED'))
            return self.manifest(id)
        finally: temp.unlink(missing_ok=True)
    def load(self,id,*,byte_budget=64*1024*1024,experimental=False):
        m=self.manifest(id)
        if m['state']!='ACTIVE' and not experimental: raise ContractError('candidate not admitted')
        if type(byte_budget) is not int or byte_budget<1: raise ContractError('byte budget')
        path=self.root/(m['sha256']+'.safetensors')
        if path.stat().st_size!=m['bytes'] or m['bytes']>byte_budget or digest(path)!=m['sha256']: raise ContractError('artifact budget or integrity')
        tensors=load_file(str(path)); w,b=tensors['w'],tensors['b']
        if list(w.shape)!=m['shape'] or b.shape!=(w.shape[0],) or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in (w,b)): raise ContractError('tensor contract')
        return AffineExpert(m,w,b)
    def admit(self,id,cases):
        """Host-supplied independent expected pairs. Limited, not proof of general competence."""
        m=self.manifest(id)
        if m['state']!='STAGED': raise ContractError('not staged')
        cases=tuple(cases)
        if not cases: raise ContractError('empty evaluation')
        e=self.load(id,experimental=True)
        for values,expected in cases:
            actual=e.compute(tuple(values)); expected=tuple(expected)
            if len(actual)!=len(expected) or any(not math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6) for a,b in zip(actual,expected)): return False
        with self.db: self.db.execute('UPDATE experts SET state=? WHERE id=?',('ACTIVE',id))
        return True
    def compose_affine(self,id,family,first,second,*,byte_budget=64*1024*1024):
        """Analytic fusion of compatible affine maps; candidate still requires admission.
        Original experts remain untouched. Composition is not arbitrary weight merging.
        """
        a=self.load(first,byte_budget=byte_budget); b=self.load(second,byte_budget=byte_budget)
        if a.m['target_topic']!=b.m['source_topic'] or a.w.shape[0]!=b.w.shape[1]: raise ContractError('incompatible composition')
        if a.m['bytes']+b.m['bytes']>byte_budget: raise ContractError('composition input budget')
        w=b.w@a.w; bias=b.w@a.b+b.b
        return self.stage_affine(id,family,w.tolist(),bias.tolist(),source_topic=a.m['source_topic'],target_scope=b.m['target_scope'],target_topic=b.m['target_topic'],parents=(first,second))
    def close(self): self.db.close()

class AffineExpert:
    def __init__(self,m,w,b): self.id=m['id']; self.m=m; self.w=w; self.b=b
    def relevant(self,snapshot): return any(s.topic==self.m['source_topic'] for s in snapshot)
    @torch.inference_mode()
    def compute(self,values):
        if len(values)!=self.w.shape[1]: raise ContractError('input width')
        v=torch.tensor(values,dtype=torch.float32); y=torch.nn.functional.linear(v,self.w,self.b)
        if not torch.isfinite(y).all(): raise ContractError('nonfinite output')
        return tuple(y.tolist())
    def forward(self,snapshot):
        for s in snapshot:
            if s.topic==self.m['source_topic']:
                yield Signal(self.m['target_scope'],self.m['target_topic'],self.compute(s.values))
