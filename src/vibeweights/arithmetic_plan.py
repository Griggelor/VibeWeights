"""Bounded arithmetic plugin for ABI 0.2. Rules/verification are SYMBOLIC.
The independent verifier compares results but never writes an answer.
"""
import sqlite3
from .abi import Slot, LIMITS, State, ContractError
from .runtime import Result, Status
from .platform import Field, Schema, Plan, Rule, ActionHandler, Delta, Engine, Registry, ExecutionFault, TaskRouter

SCHEMA=Schema('bounded.arithmetic',tuple(Field(s.name,1 if s in (Slot.KEY,Slot.RECORD) else 0,LIMITS[s]) for s in Slot)+(Field('READY',0,1),Field('HALTED',0,1)))

def make_plan(memory=None):
    def indexed(f): return f.has('KEY')
    def mem_call(fn):
        if memory is None: raise ExecutionFault('MEMORY','no memory')
        try: return fn()
        except (ContractError,sqlite3.Error) as e: raise ExecutionFault('MEMORY',str(e)) from e
    def search(f): return Delta((('RECORD',mem_call(lambda:memory.search(f.get('KEY')))),))
    def read(f): return Delta((('Y',mem_call(lambda:memory.read(f.get('RECORD'),f.get('KEY')))),))
    def verify(initial,f):
        operand=mem_call(lambda:memory.read(f.get('RECORD'),initial.get('KEY'))) if indexed(initial) else initial.get('Y')
        expected=initial.get('X')*operand
        return f.get('PRODUCT')==expected and f.get('GE')==int(expected>=initial.get('THRESHOLD')) and f.get('REMAIN')==0 and (not indexed(initial) or (not f.has('Y') and f.has('HALTED')))
    def handler(name,fields,fn): return ActionHandler(name,fields,fn)
    rules=(
        Rule('search','memory.action',lambda f:indexed(f) and not f.has('RECORD'),((0,handler('SEARCH',('RECORD',),search)),)),
        Rule('read','memory.action',lambda f:indexed(f) and f.has('RECORD') and not f.has('Y') and not f.has('READY'),((1,handler('READ',('Y',),read)),)),
        Rule('ready','memory.action',lambda f:indexed(f) and f.has('Y') and not f.has('READY'),((2,handler('READY',('READY',),lambda f:Delta((('READY',1),)))),)),
        Rule('init','arith.init',lambda f:f.has('Y') and (not indexed(f) or f.has('READY')) and not f.has('REMAIN')),
        Rule('step','arith.step',lambda f:f.has('REMAIN') and f.get('REMAIN')>0,progress=lambda a,b:b.get('REMAIN')==a.get('REMAIN')-1),
        Rule('finish','arith.finish',lambda f:f.has('REMAIN') and f.get('REMAIN')==0 and not f.has('PRODUCT')),
        Rule('compare','logic.ge',lambda f:f.has('PRODUCT') and not f.has('GE')),
        Rule('evict','memory.action',lambda f:indexed(f) and f.has('GE') and f.has('Y'),((3,handler('EVICT',('Y',),lambda f:Delta(clears=('Y',)))),)),
        Rule('halt','memory.action',lambda f:indexed(f) and f.has('GE') and not f.has('Y') and not f.has('HALTED'),((4,handler('HALT',('HALTED',),lambda f:Delta((('HALTED',1),)))),)),
    )
    contracts=(('arith.init',('Y',),('ACC','REMAIN'),False,0),('arith.step',('ACC','X','REMAIN'),('ACC','REMAIN'),False,0),('arith.finish',('ACC',),('PRODUCT',),False,0),('logic.ge',('PRODUCT','THRESHOLD'),('GE',),False,0),('memory.action',('RECORD','Y','GE'),(),True,5))
    return Plan('bounded.multiply_compare',SCHEMA,rules,('X','Y','KEY','THRESHOLD'),('PRODUCT','GE','THRESHOLD'),lambda f:f.has('GE') and (not indexed(f) or f.has('HALTED')),verify,contracts,(('X','Y','THRESHOLD'),('X','KEY','THRESHOLD')))

class PlatformRuntime:
    """Compatibility facade for existing structured/language clients."""
    def __init__(self,registry,memory=None): self.registry=registry; self.memory=memory; self.plan=make_plan(memory)
    @classmethod
    def load(cls,root,memory=None,**limits): return cls(Registry.discover(root,SCHEMA,**limits),memory)
    def task_router(self):
        router=TaskRouter(); router.register(self.plan,self.registry); router.register(make_compare_plan(),self.registry); return router
    def run(self,x,threshold,*,y=None,key=None,budget=48):
        if (y is None)==(key is None): return Result(State(),Status.INVALID,(),0,'exactly one operand source')
        r=Engine(self.registry).run(self.plan,{'X':x,'THRESHOLD':threshold,**({'KEY':key} if y is None else {'Y':y})},budget=budget)
        state=State().change({s:r.frame.get(s.name) for s in Slot if r.frame.has(s.name)},tuple(Slot))
        return Result(state,Status[r.status],r.trace,r.steps,r.error)


def make_compare_plan():
    # PRODUCT is the legacy scalar comparison operand port; no multiplication is needed.
    return Plan('bounded.compare',SCHEMA,(Rule('compare','logic.ge',lambda f:not f.has('GE')),),('PRODUCT','THRESHOLD'),('PRODUCT','THRESHOLD','GE'),lambda f:f.has('GE'),lambda a,b:b.get('GE')==int(a.get('PRODUCT')>=a.get('THRESHOLD')),(('logic.ge',('PRODUCT','THRESHOLD'),('GE',),False,0),))
