"""Analytic construction only: no examples, RNG, training or optimization."""
from pathlib import Path
import hashlib, json
import torch
from safetensors.torch import save_file, load_file
from .abi import ABI, Slot, ContractError

# (input slots, output slots, uses mask, expected layer dimensions)
S=Slot
SPECS={
 'arith.init': ((S.Y,), (S.ACC,S.REMAIN),False,(1,2)),
 'arith.step': ((S.ACC,S.X,S.REMAIN),(S.ACC,S.REMAIN),False,(3,2)),
 'arith.finish': ((S.ACC,),(S.PRODUCT,),False,(1,1)),
 'logic.ge': ((S.PRODUCT,S.THRESHOLD),(S.GE,),False,(2,2,1)),
 'memory.action': ((S.RECORD,S.Y,S.GE),(),True,(3,5)),
}

def t(x): return torch.tensor(x,dtype=torch.float32)
def layer(w,b): return {'w0':t(w),'b0':t(b)}
def generate():
    # multiplication by repeated addition; controller supplies bounded recurrence.
    ge=layer([[1,-1],[1,-1]],[1,0]); ge.update(w1=t([[1,-1]]),b1=t([0]))
    # logits for SEARCH, READ, READY, EVICT, HALT on reachable validity masks.
    return {
      'arith.init':layer([[0],[1]],[0,0]),
      'arith.step':layer([[1,1,0],[0,0,1]],[0,-1]),
      'arith.finish':layer([[1]],[0]),
      'logic.ge':ge,
      'memory.action':layer([[-1,0,-1],[1,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]],[1,0,0,-1,0]),
    }
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def build(root):
    root=Path(root); wp=root/'artifacts/weights'; mp=root/'catalog/manifests'; wp.mkdir(parents=True,exist_ok=True); mp.mkdir(parents=True,exist_ok=True)
    for name,weights in generate().items():
        path=wp/(name+'.safetensors'); save_file(weights,str(path)); ins,outs,mask,dims=SPECS[name]
        manifest=dict(id=name,version='0.1.0',category=name.split('.')[0],status='experimental',weight_origin='DIRECT',latent_abi=ABI,architecture='affine_relu',parameters=sum(x.numel() for x in weights.values()),input_contract=[s.name for s in ins],output_contract=[s.name for s in outs] or ['ACTION_SCORES_5'],mask_input=mask,layer_dimensions=dims,weight_generator='vibeweights.weights.generate',weights_sha256=sha(path),loop_policy='recurrent_shared' if name=='arith.step' else 'single',memory_permissions=['SEARCH','READ','EVICT'] if mask else [],verification_suite='tests/',tolerance=0,limitations='bounded integer ABI; no language understanding')
        (mp/(name+'.json')).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

class Expert(torch.nn.Module):
    def __init__(self,name,weights):
        super().__init__(); self.name=name; ins,outs,mask,dims=SPECS[name]; self.ins=ins; self.outs=outs; self.mask=mask; self.depth=len(dims)-1
        expected={k: shape for i in range(self.depth) for k,shape in [(f'w{i}',(dims[i+1],dims[i])),(f'b{i}',(dims[i+1],))]}
        if set(weights)!=set(expected): raise ContractError('weight keys')
        for k,shape in expected.items():
            x=weights[k]
            if tuple(x.shape)!=shape or x.dtype!=torch.float32 or not torch.isfinite(x).all(): raise ContractError('weight shape/dtype/finite')
            self.register_parameter(k,torch.nn.Parameter(x.clone(),requires_grad=False))
    @torch.inference_mode()
    def forward(self,x):
        for i in range(self.depth): x=torch.relu(torch.nn.functional.linear(x,getattr(self,f'w{i}'),getattr(self,f'b{i}')))
        return x
    def run(self,state):
        out=self(state.tensor(self.ins,self.mask))
        return state.apply(self.outs,out) if self.outs else out

def load(root):
    root=Path(root); pool={}
    for name,(ins,outs,mask,dims) in SPECS.items():
        path=root/'artifacts/weights'/(name+'.safetensors'); m=json.loads((root/'catalog/manifests'/(name+'.json')).read_text(encoding='utf-8'))
        if (m['id'],m['version'],m['latent_abi'],m['weight_origin'],m['weights_sha256'])!=(name,'0.1.0',ABI,'DIRECT',sha(path)) or m['input_contract']!=[s.name for s in ins] or m['output_contract']!=([s.name for s in outs] or ['ACTION_SCORES_5']) or m['layer_dimensions']!=list(dims) or m['mask_input'] is not mask: raise ContractError('manifest mismatch')
        e=Expert(name,load_file(str(path)))
        if sum(p.numel() for p in e.parameters())!=m['parameters']: raise ContractError('parameter count')
        pool[name]=e
    return pool
