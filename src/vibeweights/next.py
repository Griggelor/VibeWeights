"""Executable v0.6 vertical slice; legacy arithmetic remains a separate reference."""
import argparse,json
from pathlib import Path
from .paths import project_root,data_root
from .factory import Catalog
from .environment import Signal,Environment,EventLedger
from .bootstrap import BootstrapSpec,BootstrapExpert

def demo(root,x):
    data=data_root(root); cat=Catalog(data/'experts'); ledger=EventLedger(data/'environment.sqlite')
    try:
        definitions=[('demo.scale@1','demo.scale',[[2]],[0],'input','work','scaled',()),('demo.scale@2','demo.scale',[[3]],[0],'input','work','scaled',('demo.scale@1',)),('demo.shift@1','demo.shift',[[1]],[1],'scaled','global','result',())]
        for id,f,w,b,src,scope,dst,parents in definitions:
            if not any(m['id']==id for m in cat.list()):
                cat.stage_affine(id,f,w,b,source_topic=src,target_scope=scope,target_topic=dst,parents=parents)
            if cat.manifest(id)['state']=='STAGED':
                # Independently supplied fixture: both scale variants have distinct validated functions.
                expected={'demo.scale@1':(4,6),'demo.scale@2':(6,9),'demo.shift@1':(3,4)}[id]
                if not cat.admit(id,[((2,),(expected[0],)),((3,),(expected[1],))]): raise RuntimeError('fixture admission failed')
        experts=[cat.load(id) for id,*_ in definitions]
        # Task-specific oracle belongs to demo, not core. All proposals retained; only 2x passes.
        validators={('work','scaled'):lambda s,v:v==(2*x,),('global','result'):lambda s,v:v==(2*x+1,)}
        return Environment(experts,validators,ledger).run([Signal('global','input',(x,))],('global','result'))
    finally: ledger.close();cat.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path); sub=p.add_subparsers(dest='mode',required=True)
    d=sub.add_parser('demo');d.add_argument('x',type=int)
    m=sub.add_parser('memory-demo');m.add_argument('x',type=int);m.add_argument('--initialize',action='store_true');m.add_argument('--remember',action='store_true')
    b=sub.add_parser('bootstrap');b.add_argument('--model',required=True);b.add_argument('--port',type=int,default=1234);b.add_argument('--text',required=True)
    a=p.parse_args()
    if a.mode=='demo': result=demo(a.root or project_root(),a.x)
    elif a.mode=='memory-demo':
        from .memory_demo import demo as memory_demo
        result=memory_demo(a.root or project_root(),a.x,initialize=a.initialize,remember=a.remember)
    else: result=BootstrapExpert(BootstrapSpec('bootstrap@1',a.model,a.port)).propose(a.text)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if a.mode!='memory-demo' or result['status']=='OK' else 1
if __name__=='__main__': raise SystemExit(main())
