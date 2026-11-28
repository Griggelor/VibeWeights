"""Explicit two-input/two-output numeric bridge for the first real social trial.

The port mapping is part of the task ABI, not a feature of a participation or
invitation head. This is a deliberately structured reference, not a universal
latent interface or evidence of autonomous competence discovery.
"""
from .abi import ContractError
from .qwen3_fragments import _hash
from .social_learning import SocialFragmentCore


def pack_tokens(first,second):
    a,b=tuple(first),tuple(second)
    if not 1<=len(a)<=63 or not 1<=len(b)<=63 or any(type(t) is not int or t<0 for t in (*a,*b)):
        raise ContractError('social fragment token ports')
    return (len(a),*a,len(b),*b)


def unpack_tokens(payload):
    if type(payload) not in (tuple,list) or not 4<=len(payload)<=128 or any(type(t) is not int or t<0 for t in payload):
        raise ContractError('social fragment packed payload')
    a=payload[0]
    if not 1<=a<=63 or a+1>=len(payload):raise ContractError('social fragment first port')
    b=payload[a+1]
    if not 1<=b<=63 or len(payload)!=a+b+2:raise ContractError('social fragment second port')
    return tuple(payload[1:a+1]),tuple(payload[a+2:])


class FragmentPortCore:
    """An admitted frozen fragment plus its signed binary readout port."""
    def __init__(self,core,*,port,scale):
        if not isinstance(core,SocialFragmentCore) or type(port) is not int or port not in (0,1) or type(scale) not in (int,float) or scale not in (.25,1.):
            raise ContractError('social fragment port contract')
        self._core=core
        self.id=core.id
        self.port=port
        self.scale=float(scale)
        self.output_width=2
        self.artifact_sha256=_hash({'contract':'fragment-port-v1','core':core.artifact_sha256,
                                   'port':port,'scale':self.scale})

    def contribute(self,payload):
        if _hash({'contract':'fragment-port-v1','core':self._core.artifact_sha256,
                  'port':self.port,'scale':self.scale})!=self.artifact_sha256:
            raise ContractError('social fragment port changed')
        tokens=unpack_tokens(payload)[self.port]
        bit,=self._core.contribute(tokens)
        if bit not in (0.,1.):raise ContractError('social fragment bit')
        v=[0.,0.]
        v[self.port]=self.scale*(2*bit-1)
        return tuple(v)


class OfflineMemo:
    """Bounded exact-input replay of immutable reference computations.

    Only used in the offline experiment to separate repeated social learning
    from repeated expensive frozen forwards. No target labels or head features
    in the key; first occurrence always executes the actual fragment. Do not
    use this wrapper as production residency or source-integrity infrastructure.
    """
    def __init__(self,core,*,limit=128):
        if not isinstance(core,FragmentPortCore) or type(limit) is not int or not 1<=limit<=128:
            raise ContractError('social memo contract')
        self._core=core
        self.id=core.id;self.output_width=core.output_width
        self.artifact_sha256=core.artifact_sha256
        self.limit=limit;self.cache={};self.calls=0;self.forwards=0

    def contribute(self,payload):
        if self._core.artifact_sha256!=self.artifact_sha256:
            raise ContractError('social memo source changed')
        # Validate both ports on every call; key only the actual core input.
        tokens=unpack_tokens(payload)[self._core.port]
        self.calls+=1
        if tokens not in self.cache:
            self.forwards+=1
            v=self._core.contribute(payload)
            if len(self.cache)<self.limit:self.cache[tokens]=v
            return v
        return self.cache[tokens]
