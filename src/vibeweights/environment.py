"""Numerical event-contract slice. No task-specific router.
Synchronous snapshot rounds; immutable scalar/vector facts and separate proposals.
Only trusted host code supplies adapters and independent verifier functions.
"""
from dataclasses import dataclass
from collections import defaultdict
import math, sqlite3, json, uuid, random
from itertools import islice
from .abi import ContractError


def identifier(s):
    if not isinstance(s,str) or not 1<=len(s)<=128 or not all(c.isascii() and (c.isalnum() or c in '_.@:-') for c in s):
        raise ContractError('invalid identifier')
    return s

@dataclass(frozen=True)
class Signal:
    scope: str
    topic: str
    values: tuple
    def __post_init__(self):
        identifier(self.scope); identifier(self.topic)
        if type(self.values) is not tuple or not 1<=len(self.values)<=4096 or any(type(v) not in (int,float) or abs(v)>1e100 or not math.isfinite(v) for v in self.values):
            raise ContractError('finite bounded numeric vector required')

@dataclass(frozen=True)
class Proposal:
    source: str
    signal: Signal
    def __post_init__(self): identifier(self.source)

@dataclass(frozen=True)
class Limits:
    steps: int=128
    rounds: int=32
    signals: int=2048
    probes: int=0
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.steps,self.rounds,self.signals)) or type(self.probes) is not int or self.probes<0:
            raise ContractError('invalid resource limits')

class EventLedger:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=2)
        self.db.execute('CREATE TABLE IF NOT EXISTS environment_runs(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    def save(self,result,run=None):
        run=run or uuid.uuid4().hex
        with self.db: self.db.execute('INSERT INTO environment_runs VALUES(?,?)',(run,json.dumps(result,allow_nan=False)))
        return run
    def read(self,run):
        row=self.db.execute('SELECT payload FROM environment_runs WHERE id=?',(run,)).fetchone()
        if row is None: raise ContractError('unknown run')
        return json.loads(row[0])
    def close(self): self.db.close()

class Environment:
    """Experts see immutable snapshots; opt-in context adds logical scope filtering.
    Independent validators own release.
    A proposal does not become a fact until exactly one distinct candidate is verified.
    New scope identifiers are permitted; no fixed group membership or population limit.
    The synchronous engine bounds invocations, not time or memory of arbitrary callbacks.
    """
    def __init__(self,experts,validators,ledger=None):
        self.experts=tuple(sorted(experts,key=lambda e:e.id)); self.validators=dict(validators); self.ledger=ledger
        if len({identifier(e.id) for e in self.experts})!=len(self.experts): raise ContractError('duplicate instance')
    def run(self,inputs,goal,*,limits=Limits(),seed=0,knowledge=None,access=None,context_limits=None,remember_experts=(),_remember_writer=None,_run_id=None):
        from .context import WorkingContext, ContextLimits
        from .knowledge import Access
        context=WorkingContext(knowledge,access or {},context_limits or ContextLimits()) if knowledge is not None else None
        remember_experts=frozenset(identifier(x) for x in remember_experts)
        run_id=uuid.uuid4().hex if _run_id is None else identifier(_run_id); proposal_parents=defaultdict(set)
        if type(seed) is not int: raise ContractError('seed')
        identifier(goal[0]); identifier(goal[1]); facts={}; hypotheses=defaultdict(set); trace=[]; faults=[]; seen=set(); steps=probes=0; status='STAGNATION'; rng=random.Random(seed)
        for x in inputs:
            if not isinstance(x,Signal) or (x.scope,x.topic) in facts: raise ContractError('duplicate/invalid input')
            facts[x.scope,x.topic]=x.values
        if goal in facts: raise ContractError('goal must not be a supplied fact')
        if len(facts)>limits.signals: raise ContractError('signal budget')
        for round_no in range(limits.rounds):
            snapshot=tuple(Signal(s,t,v) for (s,t),v in sorted(facts.items())); fresh=[]; inactive=[]; memory_changed=False
            def expert_snapshot(expert):
                if context is None: return snapshot
                scopes=context.access.get(expert.id,Access(expert.id)).scopes
                return tuple(s for s in snapshot if not s.scope.startswith(('private:','group:')) or s.scope in scopes)
            for expert in self.experts:
                own_snapshot=expert_snapshot(expert)
                try: active=expert.relevant(own_snapshot)
                except Exception as exc:
                    faults.append((expert.id,'activation',type(exc).__name__)); continue
                if not active: inactive.append(expert); continue
                # Opt-in adapters ask for memory explicitly; actions are separate from fact publication.
                uses_memory=hasattr(expert,'memory_action')
                if uses_memory and context is None:
                    faults.append((expert.id,'memory','UNAVAILABLE')); continue
                if uses_memory:
                    try:
                        action=expert.memory_action(own_snapshot,context.view(expert.id))
                        if action is not None:
                            memory_changed |= context.apply(expert.id,action,allow_remember=expert.id in remember_experts and goal in facts and (expert.id,facts[goal]) in hypotheses[goal])
                            continue
                    except Exception as exc:
                        faults.append((expert.id,'memory',str(exc)[:160])); continue
                signature=(expert.id,own_snapshot,context.view(expert.id) if uses_memory else None)
                if signature in seen: continue
                if steps>=limits.steps: status='BUDGET'; break
                seen.add(signature); steps+=1
                try:
                    out=tuple(islice(expert.forward_with_memory(own_snapshot,context.view(expert.id)) if uses_memory else expert.forward(own_snapshot),limits.signals+1))
                    if len(out)>limits.signals or any(not isinstance(p,Signal) for p in out): raise ContractError('invalid proposals')
                    if context is not None:
                        scopes=context.access.get(expert.id,Access(expert.id)).scopes
                        for p in out:
                            if p.scope.startswith(('private:','group:')) and p.scope not in scopes: raise ContractError('unauthorized workspace scope')
                            if uses_memory and any(r.scope not in ('global',p.scope) for r in context.view(expert.id).records): raise ContractError('loaded source scope cannot be widened')
                    fresh.extend(Proposal(expert.id,p) for p in out)
                    if uses_memory:
                        for p in out:
                            proposal_parents[(p.scope,p.topic,expert.id,p.values)].update(r.id for r in context.view(expert.id).records)
                    trace.append((round_no,expert.id,'PROPOSE',len(out)))
                except Exception as exc: faults.append((expert.id,'forward',type(exc).__name__))
            if status=='BUDGET': break # incomplete round cannot release a result
            # Probe only after productive invocations. Separate optional quota, no writes.
            inactive=[e for e in inactive if not hasattr(e,'memory_action')]
            if inactive and probes<limits.probes:
                e=rng.choice(inactive); probes+=1
                try: tuple(islice(e.forward(expert_snapshot(e)),limits.signals+1)); trace.append((round_no,e.id,'PROBE',0))
                except Exception as exc: faults.append((e.id,'probe',type(exc).__name__))
            for p in fresh:
                key=(p.signal.scope,p.signal.topic)
                hypotheses[key].add((p.source,p.signal.values))
            if len(facts)+sum(map(len,hypotheses.values()))>limits.signals: status='BUDGET'; break
            changed=False; conflicts=[]; verifier_failed=False
            def verified(key,values):
                nonlocal verifier_failed
                try: return self.validators[key](snapshot,values) is True
                except Exception as exc:
                    verifier_failed=True; faults.append(('verifier',key,type(exc).__name__)); return False
            for key,options in sorted(hypotheses.items()):
                if key in facts: # immutable publication; retain and flag verified contradictions
                    if key in self.validators:
                        if any(v!=facts[key] and verified(key,v) for _,v in options): conflicts.append(key)
                    continue
                verifier=self.validators.get(key)
                if verifier is None: continue
                accepted={v for _,v in options if verified(key,v)}
                if len(accepted)>1: conflicts.append(key)
                elif len(accepted)==1: facts[key]=accepted.pop(); changed=True
            if verifier_failed: status='INVALID'; break
            if conflicts: status='CONFLICT'; break
            if goal in facts and not changed and not memory_changed: status='OK'; break
            if not changed and not memory_changed: break
        else: status='BUDGET'
        result={'status':status,'value':list(facts[goal]) if status=='OK' else None,'steps':steps,'probes':probes,'trace':trace,'faults':faults,
                'facts':[{'scope':s,'topic':t,'values':v} for (s,t),v in sorted(facts.items())],
                'hypotheses':[{'scope':s,'topic':t,'source':src,'values':v} for (s,t),opts in sorted(hypotheses.items()) for src,v in sorted(opts)]}
        if context is not None:
            result['memory_actions']=context.actions
            result['memory_trace']=context.trace
        if _remember_writer is not None and faults:
            status='INVALID'; result['status']=status; result['value']=None
        if context is not None and status=='OK':
            requested=tuple(e for e in sorted(remember_experts) if context.view(e).remember and (e,facts[goal]) in hypotheses[goal])
            if requested:
                # Store exactly the verified goal (never a free-form expert payload). Keep
                # loaded record revisions as conservative provenance, even after eviction.
                parents=tuple(sorted({rid for e in requested for rid in proposal_parents[(*goal,e,facts[goal])]}))
                try:
                    fields=dict(scope=goal[0],key='result:'+run_id,topic=goal[1],values=facts[goal],source='run:'+run_id,status='VERIFIED',parents=parents)
                    actor=context.access.get(requested[0],Access(requested[0]))
                    rec=knowledge.store(actor,**fields) if _remember_writer is None else _remember_writer(result,actor,**fields)
                    result['remembered']={'id':rec.id,'experts':requested,'parents':rec.parents}
                except Exception as exc:
                    if _remember_writer is not None: raise
                    result['memory_commit_error']=str(exc)[:160]
        if self.ledger: result['run_id']=self.ledger.save(result,run_id)
        return result
