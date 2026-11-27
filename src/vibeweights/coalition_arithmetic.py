"""Arithmetic participants for the decentralized reference. No ordered rules or TaskRouter.
Local gates are DIRECT; feature extraction, action adapters and verifier are SYMBOLIC.
"""
from pathlib import Path
import hashlib,json
import torch
from safetensors.torch import save_file,load_file
from .coalition import Gate,construct_gate,Participant,CoalitionEngine,Goal
from .arithmetic_plan import SCHEMA
from .platform import Registry,Delta,ExecutionFault
from .abi import ContractError

FEATURES=SCHEMA.names+('REMAIN_POSITIVE',)
F={n:i for i,n in enumerate(FEATURES)}
def clause(**bits): return tuple((F[k],v) for k,v in sorted(bits.items()))
def features(f): return torch.tensor([int(f.has(n)) for n in SCHEMA.names]+[int(f.has('REMAIN') and f.get('REMAIN')>0)],dtype=torch.float32)
# Per-module contracts, not an ordered plan. Heads are independently constructed.
HEADS={
'arith.init':((clause(X=1,Y=1,REMAIN=0,PRODUCT=0,KEY=0),clause(X=1,Y=1,REMAIN=0,PRODUCT=0,KEY=1,READY=1)),(clause(X=1),)),
'arith.step':((clause(X=1,ACC=1,REMAIN=1,REMAIN_POSITIVE=1,PRODUCT=0),),(clause(X=1),)),
'arith.finish':((clause(ACC=1,REMAIN=1,REMAIN_POSITIVE=0,PRODUCT=0),),(clause(X=1),)),
'logic.ge':((clause(PRODUCT=1,THRESHOLD=1,GE=0),),(clause(THRESHOLD=1),)),
'memory.action':((clause(KEY=1,RECORD=0,HALTED=0),clause(KEY=1,RECORD=1,Y=0,READY=0,HALTED=0),clause(KEY=1,Y=1,READY=0,GE=0,HALTED=0),clause(KEY=1,Y=1,GE=1,HALTED=0),clause(KEY=1,READY=1,Y=0,GE=1,HALTED=0)),(clause(KEY=1),)),
}

def build(root):
    d=Path(root)/'artifacts/coalition'; d.mkdir(parents=True,exist_ok=True)
    for name,(work,join) in HEADS.items():
        data={f'{kind}.{k}':v for kind,clauses in [('work',work),('join',join)] for k,v in construct_gate(len(FEATURES),clauses).items()}
        p=d/f'{name}.safetensors'; save_file(data,str(p))
        m={'id':name,'version':'0.4.0','abi':'coalition.scalar.1','features':list(FEATURES),'weight_origin':'DIRECT','parameters':sum(t.numel() for t in data.values()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        (d/f'{name}.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')

def load_heads(root,name):
    d=Path(root)/'artifacts/coalition'; p=d/f'{name}.safetensors'; m=json.loads((d/f'{name}.json').read_text(encoding='utf-8'))
    if m['id']!=name or m['version']!='0.4.0' or m['abi']!='coalition.scalar.1' or m['features']!=list(FEATURES) or m['weight_origin']!='DIRECT' or m['sha256']!=hashlib.sha256(p.read_bytes()).hexdigest(): raise ContractError('head manifest')
    data=load_file(str(p));
    if set(data)!={f'{kind}.{k}' for kind in ('work','join') for k in ('w','b','out')}: raise ContractError('head tensor keys')
    heads=tuple(Gate({k.split('.')[1]:v for k,v in data.items() if k.startswith(kind+'.')}) for kind in ('work','join'))
    if sum(h.count for h in heads)!=m['parameters'] or any(h.w.shape[1]!=len(FEATURES) for h in heads): raise ContractError('head count')
    return heads

class ArithmeticCoalitions:
    def __init__(self,root,memory,cooperation):
        self.registry=Registry.discover(root,SCHEMA); self.memory=memory; self.cooperation=cooperation; participants=[]
        for name,expert in self.registry.experts.items():
            work,join=load_heads(root,name)
            if name=='memory.action':
                allowed=('RECORD','Y','READY','HALTED'); mutable=('Y',); predecessors=(); apply=self.memory_delta
            else:
                allowed=expert.spec.outputs; mutable=allowed if name=='arith.step' else (); predecessors=(('ACC','arith.init'),('REMAIN','arith.init')) if name=='arith.step' else ()
                def apply(f,vals,ports=allowed): return Delta(tuple(zip(ports,vals))),None
            tag='+heads:'+hashlib.sha256((Path(root)/'artifacts/coalition'/f'{name}.safetensors').read_bytes()).hexdigest()
            tag+='+core:'+hashlib.sha256((Path(root)/'artifacts/weights'/f'{name}.safetensors').read_bytes()).hexdigest()
            participants.append(Participant(expert,work,join,features,apply,allowed,mutable,predecessors,tag))
        self.engine=CoalitionEngine(SCHEMA,participants,cooperation)
    def memory_delta(self,f,vals):
        if sorted(vals)!=[0,0,0,0,1]: raise ContractError('non-onehot memory action')
        code=vals.index(1)
        try:
            if code==0 and f.has('KEY') and not f.has('RECORD'): return Delta((('RECORD',self.memory.search(f.get('KEY'))),)),'SEARCH'
            if code==1 and f.has('RECORD') and not f.has('Y') and not f.has('READY'): return Delta((('Y',self.memory.read(f.get('RECORD'),f.get('KEY'))),)),'READ'
            if code==2 and f.has('Y') and not f.has('READY'): return Delta((('READY',1),)),'READY'
            if code==3 and f.has('Y') and f.has('GE'): return Delta(clears=('Y',)),'EVICT'
            if code==4 and f.has('GE') and not f.has('Y') and not f.has('HALTED'): return Delta((('HALTED',1),)),'HALT'
        except ContractError as e: raise ExecutionFault('MEMORY',str(e)) from e
        raise ContractError('memory action denied')
    def goal(self,request):
        inputs=dict(request.inputs); fields=set(inputs)
        if fields=={'PRODUCT','THRESHOLD'}: context=1
        elif fields=={'X','Y','THRESHOLD'}: context=2
        elif fields=={'X','KEY','THRESHOLD'}: context=3
        else: raise ContractError('unsupported boundary input')
        # Context identifies cooperation episodes only; does not determine expert selection.
        def done(f): return f.has('PRODUCT') and f.has('GE') and (not f.has('KEY') or (f.has('HALTED') and not f.has('Y')))
        def verify(a,b):
            if a.has('PRODUCT'): expected=a.get('PRODUCT')
            else:
                operand=self.memory.read(b.get('RECORD'),a.get('KEY')) if a.has('KEY') else a.get('Y')
                expected=a.get('X')*operand
                if not b.has('REMAIN') or b.get('REMAIN')!=0: return False
            return b.get('PRODUCT')==expected and b.get('GE')==int(expected>=a.get('THRESHOLD'))
        return Goal(inputs,context,verify,done)
    def run(self,request,**kwargs):
        return self.run_groups((request,),**kwargs)[0]
    def run_groups(self,requests,**kwargs):
        return self.engine.run_groups(tuple(self.goal(request) for request in requests),**kwargs)
