"""Per-run expert-owned working sets and bounded numeric memory actions."""
from dataclasses import dataclass
from enum import IntEnum
from .knowledge import Access,integer
from .abi import ContractError

class Op(IntEnum):
    SEARCH=1
    READ=2
    EVICT=3
    RELATED=4
    REMEMBER=5

@dataclass(frozen=True)
class Action:
    op: Op
    topic: str | None=None
    record: int | None=None
    relation: str | None=None
    space: str | None=None
    query: tuple | None=None
    limit: int=8

@dataclass(frozen=True)
class View:
    hits: tuple=()
    records: tuple=()
    remember: bool=False

@dataclass(frozen=True)
class ContextLimits:
    actions: int=128
    records: int=16
    components: int=4096
    hits: int=32
    scan: int=4096
    total_components: int=65536
    total_records: int=256
    def __post_init__(self):
        for v in (self.actions,self.records,self.components,self.hits,self.scan,self.total_components,self.total_records): integer(v,1,100_000)
        integer(self.hits,1,128)

class WorkingContext:
    def __init__(self,ledger,access,limits=ContextLimits()):
        self.ledger=ledger; self.access=dict(access); self.limits=limits
        if any(k!=v.expert for k,v in self.access.items()): raise ContractError('access identity mismatch')
        self.views={}; self.actions=0; self.trace=[]
    def view(self,expert): return self.views.get(expert,View())
    def apply(self,expert,action,*,allow_remember=False):
        if self.actions>=self.limits.actions: raise ContractError('memory action budget exhausted')
        self.actions+=1
        if not isinstance(action,Action) or not isinstance(action.op,Op): raise ContractError('invalid memory action')
        access=self.access.get(expert,Access(expert)); old=self.view(expert); hits=old.hits; records=old.records; remember=old.remember
        if action.op==Op.REMEMBER:
            if not allow_remember or any(v is not None for v in (action.topic,action.record,action.relation,action.space,action.query)):
                raise ContractError('remember not authorized for verified goal')
            remember=True
        elif action.op==Op.SEARCH:
            if action.record is not None or action.relation is not None: raise ContractError('search arguments')
            hits=self.ledger.search(access,topic=action.topic,space=action.space,query=action.query,limit=min(integer(action.limit,1,128),self.limits.hits),scan_limit=self.limits.scan)
        elif action.op==Op.RELATED:
            if any(v is not None for v in (action.topic,action.space,action.query)): raise ContractError('related arguments')
            hits=self.ledger.related(access,action.record,action.relation,limit=min(integer(action.limit,1,128),self.limits.hits))
        elif action.op in (Op.READ,Op.EVICT):
            if any(v is not None for v in (action.topic,action.relation,action.space,action.query)): raise ContractError('record arguments')
            rid=integer(action.record)
            if action.op==Op.READ:
                rec=self.ledger.read(rid,access)
                if rec.status!='VERIFIED': raise ContractError('unverified record is not a trusted context fact')
                records=tuple(r for r in records if r.id!=rid)+(rec,)
            else: records=tuple(r for r in records if r.id!=rid)
        if len(records)>self.limits.records or sum(len(r.values)+len(r.embedding or ()) for r in records)>self.limits.components: raise ContractError('working context budget exceeded')
        new=View(hits,records,remember)
        others=[v for k,v in self.views.items() if k!=expert]
        if sum(len(v.records) for v in others)+len(records)>self.limits.total_records or sum(len(r.values)+len(r.embedding or ()) for v in [*others,new] for r in v.records)>self.limits.total_components:
            raise ContractError('total working context budget exceeded')
        self.views[expert]=new
        self.trace.append((expert,int(action.op),action.record,tuple(hits),tuple(r.id for r in records)))
        return new!=old
