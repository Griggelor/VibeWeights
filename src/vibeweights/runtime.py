from dataclasses import dataclass
from enum import IntEnum
import torch
import sqlite3
from .abi import State, Slot as S, ContractError
class Action(IntEnum): SEARCH=0; READ=1; READY=2; EVICT=3; HALT=4
class Status(IntEnum): OK=0; BUDGET=1; STAGNATION=2; INVALID=3; MEMORY=4; ROUTING=5; VERIFICATION=6
@dataclass(frozen=True)
class Result:
    state: State
    status: Status
    trace: tuple
    steps: int
    error: str=''

class Runtime:
    """Fixed reference controller; weighted memory actions and arithmetic only."""
    def __init__(self,pool,memory=None): self.pool=pool; self.memory=memory
    def run(self,x,threshold,*,y=None,key=None,budget=48):
        state=State(); trace=[]; steps=0
        def tick(name):
            nonlocal steps
            if steps>=budget: raise Stop(Status.BUDGET,'step budget')
            steps+=1; trace.append(name)
        def expert(name):
            nonlocal state
            tick(name); state=self.pool[name].run(state)
        def memory_action(expected):
            nonlocal state
            tick('memory.action'); scores=self.pool['memory.action'].run(state)
            if scores.shape!=(5,) or scores.dtype!=torch.float32 or not torch.isfinite(scores).all() or sorted(scores.tolist())!=[0.,0.,0.,0.,1.]: raise Stop(Status.INVALID,'non-onehot action')
            a=Action(int(scores.argmax())); trace.append(a.name)
            if a not in expected: raise Stop(Status.INVALID,'action violates phase permissions')
            if a==Action.SEARCH:
                if state.valid[S.RECORD] or state.valid[S.Y]: raise ContractError('duplicate search')
                try: rid=self.memory.search(state.get(S.KEY))
                except (ContractError,sqlite3.Error) as e: raise Stop(Status.MEMORY,str(e)) from e
                state=state.change({S.RECORD:rid},[S.RECORD])
            elif a==Action.READ:
                if not state.valid[S.RECORD] or state.valid[S.Y]: raise ContractError('invalid read phase')
                try: value=self.memory.read(state.get(S.RECORD),state.get(S.KEY))
                except (ContractError,sqlite3.Error) as e: raise Stop(Status.MEMORY,str(e)) from e
                state=state.change({S.Y:value},[S.Y])
            elif a==Action.EVICT:
                if not state.valid[S.Y] or not state.valid[S.GE]: raise ContractError('invalid eviction')
                state=state.change({},[S.Y],clear=[S.Y])
            return a
        try:
            if type(budget) is not int or not 1<=budget<=128: raise ContractError('budget range')
            if (y is None)==(key is None): raise ContractError('exactly one of y/key required')
            state=state.change({S.X:x,S.THRESHOLD:threshold,**({S.Y:y} if key is None else {S.KEY:key})},[S.X,S.THRESHOLD,S.Y,S.KEY])
            if key is not None:
                if self.memory is None: raise Stop(Status.MEMORY,'no memory')
                # Action head selects among phase-permitted actions; runtime validates permissions and dependencies.
                while memory_action({Action.SEARCH,Action.READ,Action.READY})!=Action.READY:
                    pass
                if not state.valid[S.Y] or not state.valid[S.RECORD]: raise ContractError('working memory missing')
            expert('arith.init')
            while state.get(S.REMAIN)>0:
                prev=state.get(S.REMAIN); expert('arith.step')
                if state.get(S.REMAIN)>=prev: raise Stop(Status.STAGNATION,'counter did not decrease')
                if state.get(S.REMAIN)!=prev-1: raise ContractError('counter transition')
            expert('arith.finish'); expert('logic.ge')
            if key is not None:
                while memory_action({Action.EVICT,Action.HALT})!=Action.HALT: pass
                if state.valid[S.Y]: raise ContractError('eviction required')
            return Result(state,Status.OK,tuple(trace),steps)
        except Stop as e: return Result(state,e.status,tuple(trace),steps,str(e))
        except ContractError as e: return Result(state,Status.INVALID,tuple(trace),steps,str(e))
class Stop(Exception):
    def __init__(self,status,message): super().__init__(message); self.status=status

def verbalize(result):
    if result.status!=Status.OK: return f'Kein Ergebnis ({result.status.name}).'
    s=result.state; product=s.get(S.PRODUCT); ge=s.get(S.GE); threshold=s.get(S.THRESHOLD)
    return f'Das Produkt ist {product}. Es ist {"mindestens" if ge else "kleiner als"} {threshold}.'
