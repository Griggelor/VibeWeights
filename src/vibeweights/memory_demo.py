"""Bounded proof: indexed recall requested by a directly constructed action head.
The query topic is fixed by the demonstrated competence; no general language grounding.
"""
import torch
from .environment import Signal,Environment,EventLedger
from .knowledge import KnowledgeLedger,Access
from .context import Action,Op
from .paths import data_root
from .factory import Catalog

class RecallExpert:
    id='demo.recall@1'
    def __init__(self,affine,remember=False):
        self.affine=affine; self.remember=remember
        # One-hot state [no hits/no records, hits/no records, records loaded].
        # Identity rows select SEARCH, READ or READY respectively.
        self.head=torch.eye(3)
    def relevant(self,snapshot): return any(s.topic=='input' for s in snapshot)
    @torch.inference_mode()
    def memory_action(self,snapshot,view):
        if any(s.topic=='result' for s in snapshot):
            if self.remember and not view.remember: return Action(Op.REMEMBER)
            return Action(Op.EVICT,record=view.records[0].id) if view.records else None
        state=torch.tensor([not view.hits and not view.records,bool(view.hits) and not view.records,bool(view.records)],dtype=torch.float32)
        action=int((self.head@state).argmax())
        if action==0: return Action(Op.SEARCH,topic='demo.offset',limit=1)
        if action==1: return Action(Op.READ,record=view.hits[0])
        return None
    def forward_with_memory(self,snapshot,view):
        if any(s.topic=='result' for s in snapshot): return
        x=next(s.values[0] for s in snapshot if s.topic=='input')
        if len(view.records)==1:
            yield Signal('global','result',self.affine.compute((x,view.records[0].values[0])))

def demo(root,x,*,initialize=False,remember=False):
    data=data_root(root); access=Access('demo.recall@1'); cat=Catalog(data/'experts'); events=EventLedger(data/'environment.sqlite')
    try:
        with KnowledgeLedger(data/'knowledge_ledger.sqlite') as ledger:
            if initialize:
                for key,topic,value in [('offset','demo.offset',6),('noise','demo.other',91)]:
                    if not ledger.search(access,topic=topic,status=None): ledger.store(access,scope='global',key=key,topic=topic,values=(value,),source='demo.fixture',status='VERIFIED')
            if not any(m['id']=='demo.recall.add@1' for m in cat.list()):
                cat.stage_affine('demo.recall.add@1','demo.recall.add',[[1,1]],[0],source_topic='pair',target_scope='global',target_topic='result')
                if not cat.admit('demo.recall.add@1',[((4,9),(13,)),((-2,5),(3,))]): raise RuntimeError('admission failed')
            # Demo verifier is independent of the forward; fixed fixture value, not general oracle.
            return Environment([RecallExpert(cat.load('demo.recall.add@1'),remember=remember)],{('global','result'):lambda s,v:v==(x+6,)},events).run([Signal('global','input',(x,))],('global','result'),knowledge=ledger,remember_experts=(access.expert,) if remember else ())
    finally: events.close(); cat.close()
