"""Execute a voluntary event group using the existing numeric environment.

Trusted host supplies a typed task and independent validators, never a route.
All intermediate signals are isolated to the group run. This does not grant
learning rights or implement a general semantic task planner.
"""
from dataclasses import asdict
import json
from .abi import ContractError
from .environment import Environment, Limits, Signal, identifier
from .knowledge_learning import _hash
from .perception import PerceptionHeads


class _Member:
    def __init__(self, population, manifest, scope):
        self.id = manifest['id']
        self.population = population
        self.source = manifest['source_topic']
        self.target = manifest['target_topic']
        self.scope = scope

    def relevant(self, snapshot):
        return any(s.scope == self.scope and s.topic == self.source for s in snapshot)

    def forward(self, snapshot):
        for s in snapshot:
            if s.scope == self.scope and s.topic == self.source:
                yield Signal(self.scope, self.target, self.population.compute(self.id, s.values))

    def recruitment_request(self, snapshot):
        """Local typed dependency check; no access to other experts' state."""
        if not self.relevant(snapshot):
            return self.id
        return None


class GroupWork:
    def __init__(self, heads):
        if not isinstance(heads, PerceptionHeads): raise ContractError('perception heads required')
        self.heads = heads
        self.bus = heads.bus
        self.learning = heads.learning
        self.db = self.learning.db
        self.db.execute('''CREATE TABLE IF NOT EXISTS group_runs_v08(
          id TEXT PRIMARY KEY, receipt TEXT NOT NULL, sha TEXT NOT NULL)''')

    def receipt(self, run):
        """Historical evidence only; never a fresh verified answer."""
        identifier(run)
        row = self.db.execute('SELECT receipt,sha FROM group_runs_v08 WHERE id=?',(run,)).fetchone()
        if row is None: raise ContractError('unknown group run')
        data = json.loads(row[0])
        if _hash(data) != row[1]: raise ContractError('group receipt integrity')
        return data

    def execute(self, run, event, target, group, inputs, goal, validators, *, oracle_id,
                limits=Limits(steps=64,rounds=16,signals=256), recruitment_budget=0):
        """One group-local numeric task; no manual ordering or external experts.

        Validators map topic to independent callable(snapshot, values). The
        caller's oracle identity is a trusted attestation. IDs are immutable;
        retry a failed or historical execution with a new run ID.

        A positive recruitment_budget authorizes publication of missing-topic
        metadata proposed by local adapters after clean stagnation. All or none
        of the local requests are dispatched. Receipt first, source commit
        second; dispatch_recruitment recovers an interrupted publication.
        """
        if type(recruitment_budget) is not int or not 0<=recruitment_budget<=64:
            raise ContractError('local recruitment budget')
        for x in (run,event,group,goal,oracle_id): identifier(x)
        if type(target) is not str or len(target)>140: raise ContractError('group target')
        if (not isinstance(limits,Limits) or limits.steps>1024 or limits.rounds>64 or
            limits.signals>2048 or limits.probes): raise ContractError('group work limits')
        if type(inputs) is not dict or not 1<=len(inputs)<=64 or goal in inputs:
            raise ContractError('group inputs')
        if (type(validators) is not dict or not 1<=len(validators)<=64 or goal not in validators or
            any(not callable(v) for v in validators.values())): raise ContractError('group validators')
        for topic in (*inputs,*validators): identifier(topic)
        scope='group:'+_hash([event,target,group,run])
        signals=tuple(Signal(scope,k,v) for k,v in sorted(inputs.items()))
        l=self.learning
        with l._guard():
            if self.db.execute('SELECT 1 FROM group_runs_v08 WHERE id=?',(run,)).fetchone():
                raise ContractError('group run already recorded; use historical receipt')
            e=self.bus.event(event)
            if not self.bus._live(e,self.bus._conflicts()): raise ContractError('stale learning event')
            members=next((g['experts'] for g in self.bus.groups(event)
                          if g['target']==target and g['group']==group),())
            if not 1<=len(members)<=64: raise ContractError('live group membership bound')
            manifests=[]; bindings=[]
            for expert in members:
                # Revalidate the admitted head and its local decision. A legacy
                # Python-only response is not enough for this neural group path.
                if (target,group) not in self.heads.propose(expert,e):
                    raise ContractError('current neural participation required')
                m=l.catalog.manifest(expert)
                if m['abi']!='numeric-events/0.1' or m['op']!='affine' or m['target_scope']!='global':
                    raise ContractError('unsupported group core ABI')
                manifests.append(m)
                head=self.db.execute('SELECT head FROM perception_bindings_v08 WHERE expert=?',(expert,)).fetchone()[0]
                sha=self.db.execute('SELECT sha FROM perception_heads_v08 WHERE id=?',(head,)).fetchone()[0]
                bindings.append({'expert':expert,'core_sha256':m['sha256'],'head':head,'head_sha256':sha})
            adapters=tuple(_Member(l.population,m,scope) for m in manifests)
            result=Environment(adapters,{(scope,k):v for k,v in validators.items()}).run(
                signals,(scope,goal),limits=limits)
            # Even a correct goal is not returned as a successful group result
            # if a member or verifier failed during the run.
            if result['faults']:
                result['status']='INVALID'; result['value']=None
            # Host callbacks are trusted, but do not publish from an event or
            # membership that became stale during a callback.
            if not self.bus._live(e,self.bus._conflicts()): raise ContractError('group event changed during work')
            if not any(g['target']==target and g['group']==group and g['experts']==members for g in self.bus.groups(event)):
                raise ContractError('group membership changed during work')
            for b in bindings:
                m=l.catalog.manifest(b['expert'])
                if m['sha256']!=b['core_sha256'] or (target,group) not in self.heads.propose(b['expert'],e):
                    raise ContractError('group member changed during work')
            data={'contract':'group-work-v1','run':run,'event':event,'target':target,'group':group,
                  'scope':scope,'members':bindings,'inputs':inputs,'goal':goal,'oracle_id':oracle_id,
                  'validator_topics':sorted(validators),'limits':asdict(limits),'result':result}
            if recruitment_budget:
                requesters=[]
                state='NOT_APPLICABLE'
                if result['status']=='STAGNATION':
                    snapshot=tuple(Signal(f['scope'],f['topic'],tuple(f['values'])) for f in result['facts'])
                    requesters=[req for adapter in adapters
                                if (req:=adapter.recruitment_request(snapshot)) is not None]
                    if not requesters:
                        state='NO_REQUEST'
                    elif e['kind']=='RECRUIT' and e['payload']['depth']>=4:
                        state='DEPTH_LIMIT'
                    elif len(requesters)>recruitment_budget:
                        state='BUDGET'
                    else:
                        state='READY'
                data['recruitment']={'contract':'local-recruitment-v1','budget':recruitment_budget,
                                     'requesters':requesters,'state':state}
            # Normalize tuples to JSON lists so persisted and immediate views match.
            raw=json.dumps(data,sort_keys=True,allow_nan=False); data=json.loads(raw)
            with self.db:
                self.db.execute('INSERT INTO group_runs_v08 VALUES(?,?,?)',(run,raw,_hash(data)))
        if recruitment_budget:
            self.dispatch_recruitment(run)
        return data

    def dispatch_recruitment(self, run):
        """Resume opt-in local intents, never repeat the group calculation.

        The original receipt is immutable. A successful source-DB commit may
        be replayed safely by content hash. Stale ancestry prevents dispatch.
        The two databases are deliberately not claimed to commit atomically.
        """
        data=self.receipt(run)
        intent=data.get('recruitment')
        if not intent or intent['state']!='READY':
            return ()
        if (intent.get('contract')!='local-recruitment-v1' or
            type(intent['budget']) is not int or not 1<=len(intent['requesters'])<=intent['budget']<=64 or
            data['result']['status']!='STAGNATION' or data['result']['faults']):
            raise ContractError('invalid local recruitment intent')
        return self.bus.publish_recruitments(run,intent['requesters'])
