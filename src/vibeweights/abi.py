from dataclasses import dataclass
from enum import IntEnum
import torch

ABI = '0.1'
class Slot(IntEnum):
    X=0; Y=1; THRESHOLD=2; ACC=3; REMAIN=4; PRODUCT=5; GE=6; KEY=7; RECORD=8

LIMITS = (31,31,961,961,31,961,1,2**24-1,2**24-1)
N = len(Slot)

class ContractError(ValueError): pass

def checked(slot, value):
    if type(value) is not int or not 0 <= value <= LIMITS[slot]:
        raise ContractError(f'invalid {slot.name}: {value!r}')
    if slot in (Slot.KEY, Slot.RECORD) and value == 0:
        raise ContractError('IDs must be positive')
    return value

@dataclass(frozen=True)
class State:
    # Immutable tuple storage; experts receive isolated numeric tensor snapshots.
    values: tuple = (0,)*N
    valid: tuple = (False,)*N
    version: str = ABI
    def __post_init__(self):
        if self.version != ABI or type(self.values) is not tuple or type(self.valid) is not tuple or len(self.values)!=N or len(self.valid)!=N:
            raise ContractError('ABI mismatch')
        for s in Slot:
            if type(self.valid[s]) is not bool: raise ContractError('mask type')
            if self.valid[s]: checked(s,self.values[s])
            elif type(self.values[s]) is not int or self.values[s]!=0: raise ContractError('invalid slot must contain canonical zero')
    def get(self,s):
        if not self.valid[s]: raise ContractError(f'missing {s.name}')
        return self.values[s]
    def tensor(self,slots,mask=False):
        return torch.tensor([int(self.valid[s]) if mask else self.get(s) for s in slots],dtype=torch.float32)
    def change(self,delta,allowed,clear=()):
        if not set(delta).union(clear)<=set(allowed) or set(delta).intersection(clear): raise ContractError('write permission')
        v=list(self.values); m=list(self.valid)
        for s,x in delta.items(): v[s]=checked(s,x); m[s]=True
        for s in clear: v[s]=0; m[s]=False
        return State(tuple(v),tuple(m))
    def apply(self,slots,tensor):
        if tensor.dtype!=torch.float32 or tuple(tensor.shape)!=(len(slots),) or not torch.isfinite(tensor).all(): raise ContractError('output tensor contract')
        raw=tensor.tolist()
        if any(x!=int(x) for x in raw): raise ContractError('noninteger output')
        return self.change(dict(zip(slots,map(int,raw))),slots)
