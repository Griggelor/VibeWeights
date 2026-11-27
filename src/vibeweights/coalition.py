"""Deterministic event-round reference. No task plan, central competence ranking or top-k.
Scalar typed workspace; event transport and membership are symbolic infrastructure.
Each participant owns its learned/constructed activation and contribution functions.
"""
from dataclasses import dataclass
from enum import IntEnum
import sqlite3, uuid, random
import torch
from .abi import ContractError
from .platform import Frame, Delta, ExecutionFault, CPUBackend

class Op(IntEnum):
    NEED=1; OFFER=2; INVITE=3; JOIN=4; PROPOSE=5; LOOP=6; VERIFY=7; READY=8; HALT=9; LEAVE=10; CHALLENGE=11; SHOT=12; PROBE=13

@dataclass(frozen=True)
class Event:
    seq: int
    group: int
    op: int
    source: int
    target: int
    revision: int
    cause: int

class Gate(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        if set(tensors)!={'w','b','out'}: raise ContractError('gate keys')
        w,b,out=(tensors[k] for k in ('w','b','out'))
        if w.ndim!=2 or b.shape!=(w.shape[0],) or out.shape!=(1,w.shape[0]) or not w.numel(): raise ContractError('gate shape')
        for k,t in tensors.items():
            if t.dtype!=torch.float32 or not torch.isfinite(t).all(): raise ContractError('gate tensor')
            self.register_parameter(k,torch.nn.Parameter(t.clone(),requires_grad=False))
    @torch.inference_mode()
    def forward(self,bits):
        if bits.shape!=(self.w.shape[1],) or not torch.all((bits==0)|(bits==1)): raise ContractError('gate bits')
        score=torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(bits,self.w,self.b)),self.out).item()
        if not 0<=score<=self.w.shape[0] or score!=int(score): raise ContractError('gate score')
        return score>0
    @property
    def count(self): return sum(t.numel() for t in self.parameters())

def construct_gate(width,clauses):
    if type(width) is not int or width<1 or not clauses: raise ContractError('empty clauses/width')
    w=torch.zeros((len(clauses),width)); b=torch.ones(len(clauses))
    for i,clause in enumerate(clauses):
        for col,val in clause:
            if type(col) is not int or not 0<=col<width or val not in (0,1) or w[i,col]!=0: raise ContractError('invalid clause')
            w[i,col]=1 if val else -1; b[i]-=val
    return {'w':w,'b':b,'out':torch.ones((1,len(clauses)))}

class Cooperation:
    """Append-only run feedback; versioned, session-isolated, never stores answers."""
    def __init__(self,path=':memory:'):
        self.db=sqlite3.connect(path,timeout=2)
        self.db.executescript('''CREATE TABLE IF NOT EXISTS cooperation_v1(run TEXT, session TEXT, context INTEGER, owner TEXT, peer TEXT, ok INTEGER CHECK(ok IN (0,1)), PRIMARY KEY(run,owner,peer)); CREATE INDEX IF NOT EXISTS cooperation_lookup ON cooperation_v1(session,context,owner,peer);''')
    def peers(self,session,context,owner):
        return tuple(r[0] for r in self.db.execute('SELECT peer FROM cooperation_v1 WHERE session=? AND context=? AND owner=? GROUP BY peer HAVING SUM(CASE WHEN ok=1 THEN 1 ELSE -1 END)>0 ORDER BY peer',(session,context,owner)))
    def feedback(self,session,context,participants,ok):
        run=uuid.uuid4().hex; ids=sorted(set(participants))
        with self.db:
            self.db.executemany('INSERT INTO cooperation_v1 VALUES(?,?,?,?,?,?)',((run,session,context,a,b,int(ok)) for a in ids for b in ids if a!=b))
    def close(self): self.db.close()

@dataclass
class Participant:
    expert: object
    work: Gate
    join: Gate
    features: object
    apply: object
    allowed: tuple
    mutable: tuple=()
    predecessors: tuple=()
    identity_tag: str=''
    def remembered_partners(self,store,session,context):
        # Local symbolic invitation policy, not a neural recruitment head.
        return store.peers(session,context,self.identity)
    @property
    def id(self): return self.expert.spec.id
    @property
    def identity(self): return self.id+'@'+self.expert.spec.version+self.identity_tag

@dataclass(frozen=True)
class CoalitionResult:
    frame: Frame
    status: str
    trace: tuple
    steps: int
    parameter_visits: int
    verified: bool
    events: tuple
    members: tuple
    contributors: tuple
    error: str=''
    head_visits: int=0
    def event_dicts(self): return [dict(e.__dict__,kind=Op(e.op).name) for e in self.events]

@dataclass(frozen=True)
class Goal:
    inputs: dict
    context: int
    verify: object
    done: object

@dataclass
class SharedBudget:
    steps_left: int
    visits_left: int
    sequence: int=0
    shots_left: int=0
    shot_seed: int=0
    def __post_init__(self):
        if type(self.steps_left) is not int or not 1<=self.steps_left<=4096 or type(self.visits_left) is not int or self.visits_left<1: raise ContractError('budget')
    def charge(self,n):
        if n>self.visits_left: raise ExecutionFault('BUDGET','parameter-visit budget')
        self.visits_left-=n
    def step(self):
        if self.steps_left<=0: raise ExecutionFault('BUDGET','expert step budget')
        self.steps_left-=1
    def seq(self):
        self.sequence+=1; return self.sequence

class CoalitionEngine:
    def __init__(self,schema,participants,cooperation,*,backend=None,cache_heads=True):
        self.schema=schema; self.participants=tuple(participants); self.cooperation=cooperation; self.backend=backend or CPUBackend(); self.cache_heads=cache_heads
        if not self.participants or len({p.id for p in self.participants})!=len(self.participants): raise ContractError('participant identity')
        for p in self.participants:
            for field in p.allowed: schema.field(field)
            if not set(p.mutable)<=set(p.allowed): raise ContractError('mutation permission')
    def run(self,inputs,*,context,verify,done,session='default',budget=128,work_budget=100_000_000,recall=True,order=None,shots=0,shot_seed=0):
        return self.run_groups((Goal(inputs,context,verify,done),),session=session,budget=budget,work_budget=work_budget,recall=recall,order=order,shots=shots,shot_seed=shot_seed)[0]
    def run_groups(self,goals,*,session='default',budget=128,work_budget=100_000_000,recall=True,order=None,shots=0,shot_seed=0):
        """Fair cooperative interleaving of independent numeric goal scopes.
        Scope creation is external in this slice; member selection is local.
        Shared resource ledger; no group has a reserved top-k or step quota.
        """
        goals=tuple(goals)
        if not 1<=len(goals)<=64 or any(not isinstance(g,Goal) for g in goals): raise ContractError('goal scope count/type')
        if type(shots) is not int or not 0<=shots<=128 or type(shot_seed) is not int or not 0<=shot_seed<2**63: raise ContractError("shot configuration")
        try: shared=SharedBudget(budget,work_budget,shots_left=shots,shot_seed=shot_seed)
        except ContractError as e:
            return tuple(CoalitionResult(Frame.empty(self.schema),'INVALID',(),0,0,False,(),(),(),str(e)) for g in goals)
        runners={i:self._run(g.inputs,context=g.context,verify=g.verify,done=g.done,session=session,shared=shared,recall=recall,order=order,group=i) for i,g in enumerate(goals,1)}
        results={}
        while runners:
            for i,runner in tuple(runners.items()):
                try: next(runner)
                except StopIteration as stop: results[i]=stop.value; del runners[i]
        return tuple(results[i] for i in sorted(results))
    def _run(self,inputs,*,context,verify,done,session,shared,recall,order,group):
        # Every group has an isolated workspace. Membership starts at first JOIN.
        frame=Frame.empty(self.schema); initial=frame; trace=[]; events=[]; members=set(); contributed=set(); steps=visits=head_visits=rev=0
        names=sorted(p.id for p in self.participants); ids={name:i+1 for i,name in enumerate(names)}
        byid={p.id:p for p in self.participants}; byidentity={p.identity:p for p in self.participants}
        last_seen={}; invited=set(); seen=set(); writers={}; pending=[]; error=''; status='INVALID'
        def emit(op,source=0,target=0,cause=0):
            seq=shared.seq(); events.append(Event(seq,group,int(op),source,target,rev,cause)); return seq
        def charge(count,head=False):
            nonlocal visits,head_visits
            shared.charge(count)
            visits+=count
            if head: head_visits+=count
        # Request-local memoization of pure frozen heads. Dependencies come from weights,
        # not a central relevance rule. Same exact input subspace implies same gate result.
        feature_cache={}; head_cache={}; dependencies={}
        def activate(p,head):
            if not self.cache_heads:
                charge(head.count,True); return head(p.features(frame))
            fkey=(id(p.features),rev)
            if fkey not in feature_cache:
                bits=p.features(frame)
                if bits.dtype!=torch.float32 or bits.ndim!=1 or not torch.all((bits==0)|(bits==1)): raise ContractError('gate bits')
                feature_cache[fkey]=tuple(bits.tolist())
            bits=feature_cache[fkey]; hkey=id(head)
            if len(bits)!=head.w.shape[1]: raise ContractError('gate feature width')
            if hkey not in dependencies:
                dependencies[hkey]=tuple(torch.nonzero((head.w.detach()!=0).any(dim=0),as_tuple=True)[0].tolist())
            key=tuple(bits[i] for i in dependencies[hkey])
            # Keep one last value per head, bounded memory; no inference on a cache hit.
            if hkey in head_cache and head_cache[hkey][0]==key: return head_cache[hkey][1]
            charge(head.count,True); value=head(torch.tensor(bits,dtype=torch.float32)); head_cache[hkey]=(key,value); return value
        def join(p,cause):
            if p.id in members:return
            members.add(p.id); seq=emit(Op.JOIN,ids[p.id],cause=cause)
            if recall:
                for peer in p.remembered_partners(self.cooperation,session,context):
                    q=byidentity.get(peer)
                    if q is not None and q.id not in members and (p.id,q.id) not in invited:
                        invited.add((p.id,q.id)); pending.append((q,emit(Op.INVITE,ids[p.id],ids[q.id],seq)))
        try:
            if type(context) is not int or not isinstance(session,str) or not session: raise ContractError('context/session')
            if order is not None and (len(order)!=len(names) or set(order)!=set(names)): raise ContractError('dispatch order')
            dispatch=tuple(byid[n] for n in (order or names)); frame=initial=frame.update(inputs,inputs); seen.add(frame.values)
            cause=emit(Op.NEED)
            # Global first impression, before any competence changes the workspace.
            # Cheap heads only; their decisions remain local. The cache prevents repeat work.
            for participant in dispatch:
                activate(participant,participant.work)
            if shared.shots_left:
                rng=random.Random(shared.shot_seed + group)
            else: rng=None
            shot_seen=set()
            def explore(cause):
                nonlocal steps
                # Random invitation from inactive, previously unshot participants.
                # Full core is probed on a valid snapshot only; no action/delta is committed.
                if not shared.shots_left or rng is None: return
                candidates=[p for p in sorted(dispatch,key=lambda q:q.id) if p.id not in members and p.id not in shot_seen and not activate(p,p.work)]
                if not candidates: return
                p=rng.choice(candidates); shot_seen.add(p.id); shared.shots_left-=1
                seq=emit(Op.SHOT,target=ids[p.id],cause=cause)
                join(p,seq)
                spec=p.expert.spec
                if spec.mask or all(frame.has(n) for n in spec.inputs):
                    # No unbudgeted forward. Probes cannot execute memory actions.
                    if shared.steps_left>0 and shared.visits_left>=spec.parameters:
                        charge(spec.parameters); shared.step(); steps+=1; steps_probe=self.backend.forward(p.expert,frame.tensor(spec.inputs,spec.mask))
                        if steps_probe.dtype!=torch.float32 or steps_probe.shape!=(spec.dimensions[-1],) or not torch.isfinite(steps_probe).all():
                            raise ContractError('invalid shot probe')
                        emit(Op.PROBE,ids[p.id],cause=seq)
            # Start and on later rounds while budget allows. No fixed active top-k.
            explore(cause)
            while True:
                before_round=rev
                for p in dispatch:
                    # No task-specific condition in scheduler. Broad discovery on changed revision.
                    while pending:
                        q,seq=pending.pop(0)
                        if q.id not in members and activate(q,q.join): join(q,seq)
                    if last_seen.get(p.id)==rev: continue
                    last_seen[p.id]=rev
                    if not activate(p,p.work): continue
                    if shared.steps_left<=0: raise ExecutionFault('BUDGET','expert step budget')
                    charge(p.expert.spec.parameters); shared.step(); steps+=1
                    if p.id in contributed: emit(Op.LOOP,ids[p.id],cause=cause)
                    seq=emit(Op.OFFER,ids[p.id],cause=cause); join(p,seq); trace.append(p.id)
                    s=p.expert.spec; out=self.backend.forward(p.expert,frame.tensor(s.inputs,s.mask))
                    if out.dtype!=torch.float32 or out.shape!=(s.dimensions[-1],) or not torch.isfinite(out).all(): raise ContractError('output contract')
                    vals=out.tolist()
                    if any(v!=int(v) for v in vals): raise ContractError('noninteger contribution')
                    delta,action=p.apply(frame,tuple(map(int,vals)))
                    if not isinstance(delta,Delta): raise ContractError('delta contract')
                    for n in set(dict(delta.writes))|set(delta.clears):
                        if n in inputs: raise ContractError('immutable initial input')
                        if frame.has(n) and n not in p.mutable:
                            emit(Op.CHALLENGE,ids[p.id],cause=seq); raise ExecutionFault('CONFLICT','existing fact cannot be overwritten')
                        if n in writers and writers[n]!=p.id and (n,writers[n]) not in p.predecessors:
                            emit(Op.CHALLENGE,ids[p.id],cause=seq); raise ExecutionFault('CONFLICT','competing field writers')
                    new=delta.apply(frame,p.allowed)
                    if new.values in seen: raise ExecutionFault('STAGNATION','repeated workspace')
                    for n in set(dict(delta.writes))|set(delta.clears): writers[n]=p.id
                    frame=new; seen.add(frame.values); rev+=1; feature_cache.clear(); contributed.add(p.id)
                    if action: trace.append(action)
                    cause=emit(Op.PROPOSE,ids[p.id],cause=seq)
                    yield  # one actual contribution; other group scopes can now advance
                if rev==before_round and not pending: break
                explore(cause)
            if not done(frame): raise ExecutionFault('STAGNATION','quiescent with unresolved goal')
            seq=emit(Op.VERIFY,cause=cause)
            if not verify(initial,frame):
                emit(Op.CHALLENGE,cause=seq); raise ExecutionFault('VERIFICATION','independent verification rejected result')
            status='OK'
        except ExecutionFault as exc: status,error=exc.status,str(exc)
        except (ContractError,sqlite3.Error) as exc: status,error='INVALID',str(exc)
        if contributed and status in ('OK','VERIFICATION','CONFLICT','STAGNATION'):
            try: self.cooperation.feedback(session,context,[byid[n].identity for n in contributed],status=='OK')
            except sqlite3.Error as exc: status,error='MEMORY',str(exc)
        if status=='OK': emit(Op.READY,cause=events[-1].seq if events else 0)
        emit(Op.HALT,cause=events[-1].seq if events else 0)
        for name in sorted(members): emit(Op.LEAVE,ids[name],cause=events[-1].seq)
        return CoalitionResult(frame,status,tuple(trace),steps,visits,status=='OK',tuple(events),tuple(sorted(members)),tuple(sorted(contributed)),error,head_visits)
