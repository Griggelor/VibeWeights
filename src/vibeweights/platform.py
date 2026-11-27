"""Task-independent numerical execution platform. No task arithmetic in scheduler.

ABI 0.2 is a schema-bound scalar tensor ABI; metadata names are not model messages.
Only the core commits deltas. Routing policies and action handlers are symbolic.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable
import hashlib
import json
import re
import torch
from safetensors.torch import load_file
from safetensors import SafetensorError
from .abi import ContractError, Slot, LIMITS

ABI_VERSION = '0.2'
ID = re.compile(r'[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z')

@dataclass(frozen=True)
class Field:
    name: str
    minimum: int
    maximum: int
    def __post_init__(self):
        if not isinstance(self.name,str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*',self.name):
            raise ContractError('invalid field name')
        if type(self.minimum) is not int or type(self.maximum) is not int or not -(2**24) <= self.minimum <= self.maximum <= 2**24:
            raise ContractError('field outside exact FP32 integer range')
    def check(self,value):
        if type(value) is not int or not self.minimum <= value <= self.maximum:
            raise ContractError(f'invalid value for {self.name}')
        return value

@dataclass(frozen=True)
class Schema:
    id: str
    fields: tuple[Field,...]
    version: str = ABI_VERSION
    def __post_init__(self):
        if not isinstance(self.id,str) or not ID.fullmatch(self.id) or self.version != ABI_VERSION or type(self.fields) is not tuple or not self.fields or any(not isinstance(f,Field) for f in self.fields):
            raise ContractError('invalid schema')
        if len(set(f.name for f in self.fields)) != len(self.fields): raise ContractError('duplicate field')
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps([self.id,self.version,[(f.name,f.minimum,f.maximum) for f in self.fields]],separators=(',',':')).encode()).hexdigest()
    @property
    def names(self): return tuple(f.name for f in self.fields)
    def field(self,name):
        for f in self.fields:
            if f.name == name: return f
        raise ContractError(f'unknown field {name}')

@dataclass(frozen=True)
class Frame:
    schema: Schema
    values: tuple
    def __post_init__(self):
        if not isinstance(self.schema,Schema) or type(self.values) is not tuple or len(self.values)!=len(self.schema.fields): raise ContractError('frame shape')
        for f,v in zip(self.schema.fields,self.values):
            if v is not None: f.check(v)
    @classmethod
    def empty(cls,schema): return cls(schema,(None,)*len(schema.fields))
    def has(self,name): return self.values[self.index(name)] is not None
    def index(self,name):
        self.schema.field(name); return self.schema.names.index(name)
    def get(self,name):
        v=self.values[self.index(name)]
        if v is None: raise ContractError(f'missing {name}')
        return v
    def update(self,delta,allowed,clear=()):
        if not set(delta).union(clear)<=set(allowed) or set(delta).intersection(clear): raise ContractError('write permission')
        values=list(self.values)
        for name,value in delta.items(): values[self.index(name)]=self.schema.field(name).check(value)
        for name in clear: values[self.index(name)]=None
        return Frame(self.schema,tuple(values))
    def tensor(self,names,mask=False):
        return torch.tensor([int(self.has(n)) if mask else self.get(n) for n in names],dtype=torch.float32)

@dataclass(frozen=True)
class ExpertSpec:
    id: str
    version: str
    origin: str
    inputs: tuple[str,...]
    outputs: tuple[str,...]
    mask: bool
    dimensions: tuple[int,...]
    action_count: int = 0
    def __post_init__(self):
        if not isinstance(self.id,str) or not ID.fullmatch(self.id) or not isinstance(self.version,str) or not re.fullmatch(r'\d+\.\d+\.\d+',self.version): raise ContractError('invalid expert identity')
        if self.origin not in {'DIRECT','SYNTHESIS','PRETRAINED','TRAINED'}: raise ContractError('invalid weight origin')
        if type(self.mask) is not bool or type(self.action_count) is not int or self.action_count<0: raise ContractError('invalid action contract')
        if not self.inputs or len(set(self.inputs))!=len(self.inputs) or len(set(self.outputs))!=len(self.outputs): raise ContractError('invalid ports')
        if bool(self.outputs)==bool(self.action_count): raise ContractError('exactly one output kind')
        if len(self.dimensions)<2 or any(type(d) is not int or d<=0 for d in self.dimensions) or self.dimensions[0]!=len(self.inputs) or self.dimensions[-1]!=(len(self.outputs) or self.action_count): raise ContractError('invalid dimensions')
    @property
    def parameters(self): return sum((a+1)*b for a,b in zip(self.dimensions,self.dimensions[1:]))

class DenseExpert(torch.nn.Module):
    """Versioned frozen affine/ReLU executor; no hidden task-specific logic."""
    def __init__(self,spec,weights):
        super().__init__(); self.spec=spec
        expected={k: shape for i,(a,b) in enumerate(zip(spec.dimensions,spec.dimensions[1:])) for k,shape in [(f'w{i}',(b,a)),(f'b{i}',(b,))]}
        if set(weights)!=set(expected): raise ContractError('weight keys')
        for k,shape in expected.items():
            v=weights[k]
            if v.dtype!=torch.float32 or tuple(v.shape)!=shape or not torch.isfinite(v).all(): raise ContractError('weight contract')
            self.register_parameter(k,torch.nn.Parameter(v.detach().clone().cpu(),requires_grad=False))
    @torch.inference_mode()
    def forward(self,x):
        for i in range(len(self.spec.dimensions)-1):
            x=torch.relu(torch.nn.functional.linear(x,getattr(self,f'w{i}'),getattr(self,f'b{i}')))
        return x

class Registry:
    def __init__(self,schema,*,max_parameters=1_000_000,max_experts=512):
        if type(max_parameters) is not int or max_parameters<1 or type(max_experts) is not int or not 1<=max_experts<=512: raise ContractError('registry limits')
        self.schema=schema; self.max_parameters=max_parameters; self.max_experts=max_experts; self._experts={}
    @property
    def experts(self): return MappingProxyType(self._experts)
    @property
    def parameter_count(self): return sum(e.spec.parameters for e in self._experts.values())
    def register(self,expert,*,replace=False):
        spec=expert.spec
        for name in spec.inputs+spec.outputs: self.schema.field(name)
        old=self._experts.get(spec.id)
        if old is not None:
            if not replace: raise ContractError('duplicate expert')
            # Compatible exchange; origin/version may change but interfaces may not.
            if (old.spec.inputs,old.spec.outputs,old.spec.mask,old.spec.action_count)!=(spec.inputs,spec.outputs,spec.mask,spec.action_count): raise ContractError('incompatible replacement')
        if len(self._experts)+int(old is None)>self.max_experts or self.parameter_count-(old.spec.parameters if old else 0)+spec.parameters>self.max_parameters: raise ContractError('registry capacity')
        if not isinstance(expert,DenseExpert): raise ContractError('unsupported backend')
        self._experts[spec.id]=expert
    @classmethod
    def discover(cls,root,schema,**limits):
        root=Path(root); reg=cls(schema,**limits)
        for path in sorted((root/'catalog/manifests').glob('*.json')):
            try:
                m=json.loads(path.read_text(encoding='utf-8')); name=m['id']
                if not isinstance(name,str) or not ID.fullmatch(name) or path.stem!=name: raise ContractError('manifest identity')
                # Explicit bridge for existing 0.1 weights. No automatic arbitrary schema mapping.
                if m['latent_abi']=='0.1':
                    if schema.id!='bounded.arithmetic' or tuple(schema.field(s.name) for s in Slot)!=tuple(Field(s.name,1 if s in (Slot.KEY,Slot.RECORD) else 0,LIMITS[s]) for s in Slot): raise ContractError('legacy ABI schema mismatch')
                elif m['latent_abi']!=ABI_VERSION or m.get('schema_id')!=schema.id or m.get('schema_sha256')!=schema.fingerprint: raise ContractError('manifest ABI mismatch')
                if m['architecture']!='affine_relu': raise ContractError('unsupported architecture')
                outputs=tuple(m['output_contract']); count=0
                if len(outputs)==1 and re.fullmatch(r'ACTION_SCORES_[1-9][0-9]*',outputs[0]): count=int(outputs[0].split('_')[-1]); outputs=()
                spec=ExpertSpec(name,m['version'],m['weight_origin'],tuple(m['input_contract']),outputs,m['mask_input'],tuple(m['layer_dimensions']),count)
                if type(m['parameters']) is not int or m['parameters']!=spec.parameters or reg.parameter_count+spec.parameters>reg.max_parameters or len(reg.experts)>=reg.max_experts: raise ContractError('parameter limit/count')
                # Bound serialized size before reading; no pickle or executable manifest content.
                wp=root/'artifacts/weights'/f'{name}.safetensors'
                if wp.stat().st_size>4*spec.parameters+65536: raise ContractError('artifact size')
                if hashlib.sha256(wp.read_bytes()).hexdigest()!=m['weights_sha256']: raise ContractError('weight hash')
                reg.register(DenseExpert(spec,load_file(str(wp))))
            except (KeyError,TypeError,ValueError,OSError,SafetensorError) as exc:
                raise ContractError(f'manifest {path.name}: {exc}') from exc
        if not reg.experts: raise ContractError('empty registry')
        return reg

@dataclass(frozen=True)
class Delta:
    writes: tuple[tuple[str,int],...] = ()
    clears: tuple[str,...] = ()
    def apply(self,frame,allowed):
        if len(dict(self.writes))!=len(self.writes) or len(set(self.clears))!=len(self.clears): raise ContractError('duplicate delta')
        return frame.update(dict(self.writes),allowed,self.clears)

@dataclass(frozen=True)
class ActionHandler:
    name: str
    allowed: tuple[str,...]
    execute: Callable[[Frame],Delta]

@dataclass(frozen=True)
class Rule:
    id: str
    expert: str
    ready: Callable[[Frame],bool]
    actions: tuple[tuple[int,ActionHandler],...] = ()
    progress: Callable[[Frame,Frame],bool] | None = None

@dataclass(frozen=True)
class Plan:
    id: str
    schema: Schema
    rules: tuple[Rule,...]
    input_fields: tuple[str,...]
    result_fields: tuple[str,...]
    done: Callable[[Frame],bool]
    verify: Callable[[Frame,Frame],bool]
    # Explicit, plan-owned ports; manifests cannot silently expand authority.
    requirements: tuple[tuple[str,tuple[str,...],tuple[str,...],bool,int],...]
    input_variants: tuple[tuple[str,...],...] = ()
    def validate(self,registry):
        if registry.schema!=self.schema or not self.rules or len({r.id for r in self.rules})!=len(self.rules): raise ContractError('plan/schema mismatch')
        if len(set(self.input_fields))!=len(self.input_fields) or len(set(self.result_fields))!=len(self.result_fields) or not self.result_fields: raise ContractError('invalid plan fields')
        for name in self.input_fields+self.result_fields: self.schema.field(name)
        variants=self.input_variants or (self.input_fields,)
        if set().union(*(set(v) for v in variants))!=set(self.input_fields) or any(len(set(v))!=len(v) for v in variants): raise ContractError('invalid input variants')
        contracts={x[0]:x[1:] for x in self.requirements}
        if len(contracts)!=len(self.requirements) or set(contracts)!={r.expert for r in self.rules}: raise ContractError('plan requirements mismatch')
        for r in self.rules:
            if r.expert not in registry.experts: raise ContractError(f'missing expert {r.expert}')
            spec=registry.experts[r.expert].spec
            if contracts[r.expert]!=(spec.inputs,spec.outputs,spec.mask,spec.action_count): raise ContractError('plan expert port mismatch')
            if set(spec.outputs).intersection(self.input_fields): raise ContractError('expert may not rewrite task inputs')
            if bool(r.actions)!=bool(spec.action_count): raise ContractError('action routing mismatch')
            if len(dict(r.actions))!=len(r.actions): raise ContractError('duplicate action code')
            for code,h in r.actions:
                if type(code) is not int or not 0<=code<spec.action_count: raise ContractError('action code range')
                for name in h.allowed: self.schema.field(name)

@dataclass(frozen=True)
class Execution:
    frame: Frame
    status: str
    trace: tuple[str,...]
    steps: int
    error: str = ''
    verified: bool = False
    parameter_visits: int = 0

class ExecutionFault(Exception):
    def __init__(self,status,message): super().__init__(message); self.status=status

class CPUBackend:
    """Reference backend; explicit device boundary for future implementations."""
    name='cpu'
    def forward(self,expert,tensor):
        if tensor.device.type!='cpu' or any(p.device.type!='cpu' for p in expert.parameters()): raise ContractError('CPU backend device mismatch')
        return expert(tensor)

class Engine:
    def __init__(self,registry,backend=None): self.registry=registry; self.backend=CPUBackend() if backend is None else backend
    def run(self,plan,inputs,*,budget=128,work_budget=100_000_000):
        frame=Frame.empty(plan.schema); trace=[]; steps=0; visits=0
        try:
            if type(budget) is not int or not 1<=budget<=4096: raise ContractError('step budget range')
            if type(work_budget) is not int or work_budget<1: raise ContractError('parameter-visit budget')
            plan.validate(self.registry)
            if not isinstance(inputs,dict) or not any(set(inputs)==set(v) for v in (plan.input_variants or (plan.input_fields,))): raise ContractError('task input fields')
            frame=frame.update(inputs,plan.input_fields); initial=frame; seen={frame.values}
            # Snapshot registry for the full execution; replacement applies to subsequent calls.
            experts=dict(self.registry.experts)
            while not plan.done(frame):
                if steps>=budget: raise ExecutionFault('BUDGET','step budget exhausted')
                candidates=[r for r in plan.rules if r.ready(frame)]
                if len(candidates)!=1: raise ExecutionFault('ROUTING',f'{len(candidates)} eligible rules')
                rule=candidates[0]; expert=experts[rule.expert]; spec=expert.spec; before=frame
                if visits+spec.parameters>work_budget: raise ExecutionFault('BUDGET','parameter-visit budget exhausted')
                steps+=1; visits+=spec.parameters; trace.append(spec.id)
                out=self.backend.forward(expert,frame.tensor(spec.inputs,spec.mask))
                if out.dtype!=torch.float32 or tuple(out.shape)!=(spec.dimensions[-1],) or not torch.isfinite(out).all(): raise ContractError('output tensor contract')
                vals=out.tolist()
                if any(v!=int(v) for v in vals): raise ContractError('noninteger tensor output')
                if spec.action_count:
                    if sorted(vals)!=[0.]*(spec.action_count-1)+[1.]: raise ContractError('non-onehot action')
                    code=vals.index(1.); handler=dict(rule.actions).get(code)
                    if handler is None: raise ContractError('action denied in current rule')
                    trace.append(handler.name); frame=handler.execute(frame).apply(frame,handler.allowed)
                else:
                    frame=Delta(tuple(zip(spec.outputs,map(int,vals)))).apply(frame,spec.outputs)
                if any(not frame.has(n) or frame.get(n)!=initial.get(n) for n in inputs): raise ContractError('initial input overwritten')
                if rule.progress is not None and not rule.progress(before,frame): raise ExecutionFault('STAGNATION','progress invariant')
                if frame.values in seen: raise ExecutionFault('STAGNATION','repeated state')
                seen.add(frame.values)
            if any(not frame.has(n) for n in plan.result_fields): raise ContractError('missing final fields')
            if not plan.verify(initial,frame): raise ExecutionFault('VERIFICATION','independent verification rejected result')
            return Execution(frame,'OK',tuple(trace),steps,verified=True,parameter_visits=visits)
        except ExecutionFault as exc: return Execution(frame,exc.status,tuple(trace),steps,str(exc),parameter_visits=visits)
        except ContractError as exc: return Execution(frame,'INVALID',tuple(trace),steps,str(exc),parameter_visits=visits)


@dataclass(frozen=True)
class TaskFrame:
    """External intent converted to a schema-specific numeric request."""
    task_id: str
    inputs: tuple[tuple[str,int],...]
    def __post_init__(self):
        if not isinstance(self.task_id,str) or not ID.fullmatch(self.task_id) or type(self.inputs) is not tuple or any(type(p) is not tuple or len(p)!=2 or not isinstance(p[0],str) or type(p[1]) is not int for p in self.inputs) or len(dict(self.inputs))!=len(self.inputs): raise ContractError('invalid task frame')

class TaskRouter:
    """Registration/dispatch of independent task plans, not a learned intent router."""
    def __init__(self): self._plans={}
    def register(self,plan,registry):
        if not isinstance(plan.id,str) or not ID.fullmatch(plan.id) or plan.id in self._plans: raise ContractError('duplicate/invalid task')
        plan.validate(registry); self._plans[plan.id]=(plan,registry)
    @property
    def task_ids(self): return tuple(sorted(self._plans))
    def run(self,task,*,budget=128,work_budget=100_000_000):
        if not isinstance(task,TaskFrame) or task.task_id not in self._plans: raise ContractError('unknown task')
        plan,registry=self._plans[task.task_id]
        return Engine(registry).run(plan,dict(task.inputs),budget=budget,work_budget=work_budget)
