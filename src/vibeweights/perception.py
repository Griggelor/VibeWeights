"""Bounded trainable local perception heads for global learning events.

CPU reference: a head is bound to an immutable core artifact. It proposes
memberships only; the event bus owns delivery, budgets and source validation.
The host defines feature and output schemas and independent acceptance cases.
"""
import hashlib
import json
import torch
from safetensors.torch import save, load
from .abi import ContractError
from .environment import identifier
from .knowledge_learning import _hash
from .population import RESIDENT
from .learning_events import LearningEvents

FEATURES = ('knowledge', 'relearn', 'same_topic', 'affected', 'local', 'repair')


RECRUIT_FEATURES = FEATURES + ('recruit', 'provides_topic')


def _samples(cases, width, features=FEATURES):
    if type(cases) not in (tuple, list) or not 1 <= len(cases) <= 256:
        raise ContractError('perception case count')
    xs, ys, seen = [], [], set()
    for x, y in cases:
        if (type(x) not in (tuple, list) or len(x) != len(features) or
            type(y) not in (tuple, list) or len(y) != width or
            any(type(v) is not int or v not in (0, 1) for v in (*x, *y))):
            raise ContractError('binary perception case')
        if tuple(x) in seen: raise ContractError('duplicate perception input')
        seen.add(tuple(x)); xs.append(list(x)); ys.append(list(y))
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


class PerceptionHeads:
    def __init__(self, bus):
        if not isinstance(bus, LearningEvents): raise ContractError('learning event bus required')
        self.bus = bus; self.learning = bus.learning; self.db = self.learning.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS perception_heads_v08(
          id TEXT PRIMARY KEY, expert TEXT NOT NULL, core_sha TEXT NOT NULL,
          spec TEXT NOT NULL, weights BLOB NOT NULL, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS perception_reports_v08(
          id TEXT PRIMARY KEY REFERENCES perception_heads_v08(id), receipt TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS perception_bindings_v08(
          expert TEXT PRIMARY KEY, head TEXT UNIQUE NOT NULL REFERENCES perception_heads_v08(id));
        ''')

    def _core(self, expert):
        m = self.learning.catalog.manifest(expert)
        if m['state'] not in RESIDENT or expert not in self.learning.population.resident_ids():
            raise ContractError('resident perception core required')
        return m

    def stage(self, head, expert, groups, cases, *, steps=160, feature_version=1):
        """Train a small affine sigmoid head, preserving the original core."""
        identifier(head); identifier(expert)
        if (type(groups) not in (tuple, list) or not 1 <= len(groups) <= 8 or
            len(set(groups)) != len(groups)):
            raise ContractError('perception groups')
        groups = [identifier(g) for g in groups]
        if type(steps) is not int or not 1 <= steps <= 512: raise ContractError('perception training budget')
        if type(feature_version) is not int or feature_version not in (1,2):
            raise ContractError('perception feature version')
        features=FEATURES if feature_version==1 else RECRUIT_FEATURES
        x, y = _samples(cases, len(groups), features)
        with self.learning._guard():
            m = self._core(expert)
            if self.db.execute('SELECT 1 FROM perception_heads_v08 WHERE id=?',(head,)).fetchone():
                raise ContractError('perception identity reserved')
        w = torch.zeros((len(groups), len(features)), requires_grad=True)
        b = torch.zeros(len(groups), requires_grad=True)
        opt = torch.optim.SGD((w,b), lr=0.5)
        def loss(): return torch.nn.functional.binary_cross_entropy_with_logits(x @ w.T+b,y)
        before = float(loss().detach())
        for _ in range(steps):
            opt.zero_grad(); loss().backward(); opt.step()
        after = float(loss().detach())
        blob = save({'w':w.detach(), 'b':b.detach()})
        sha = hashlib.sha256(blob).hexdigest()
        spec = {'contract':f'perception-v{feature_version}','features':list(features),'groups':groups,
                'training_sha256':_hash(cases),'training_inputs':sorted(_hash(list(x)) for x,_ in cases),'steps':steps,'loss_before':before,'loss_after':after,
                'origin':'TRAINED','torch_version':str(torch.__version__)}
        with self.learning._guard():
            if self._core(expert)['sha256'] != m['sha256']: raise ContractError('perception core changed')
            with self.db:
                self.db.execute('INSERT INTO perception_heads_v08 VALUES(?,?,?,?,?,?)',
                    (head,expert,m['sha256'],json.dumps(spec,sort_keys=True),blob,sha))
        return {'head':head,'expert':expert,'sha256':sha,**spec}

    def _load(self, head):
        row=self.db.execute('SELECT expert,core_sha,spec,weights,sha FROM perception_heads_v08 WHERE id=?',(head,)).fetchone()
        if row is None: raise ContractError('unknown perception head')
        expert,core,raw,blob,sha=row; spec=json.loads(raw)
        if hashlib.sha256(blob).hexdigest()!=sha or len(blob)>16384: raise ContractError('perception artifact integrity')
        if self._core(expert)['sha256']!=core: raise ContractError('perception core changed')
        features={'perception-v1':FEATURES,'perception-v2':RECRUIT_FEATURES}.get(spec.get('contract'))
        if features is None or spec['features']!=list(features): raise ContractError('perception feature ABI')
        tensors=load(blob); n=len(spec['groups'])
        if (set(tensors)!={'w','b'} or tensors['w'].shape!=(n,len(features)) or tensors['b'].shape!=(n,) or
            any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in tensors.values())):
            raise ContractError('perception weight contract')
        return expert,spec,sha,tensors

    def admit(self, head, cases, *, oracle_id):
        """Independent host acceptance, immutable report; no implicit rebinding."""
        identifier(oracle_id)
        with self.learning._guard():
            expert,spec,sha,t=self._load(head)
            x,y=_samples(cases,len(spec['groups']),spec['features'])
            if set(spec['training_inputs']).intersection(_hash(list(v)) for v,_ in cases):
                raise ContractError('disjoint acceptance inputs required')
            if not ((y.sum(dim=0)>0)&(y.sum(dim=0)<len(y))).all():
                raise ContractError('acceptance requires positive and negative cases per group')
            with torch.no_grad():
                logits=x @ t['w'].T+t['b']; passed=int(((logits>0)==y.bool()).all(dim=1).sum())
            report={'oracle_id':oracle_id,'case_sha256':_hash(cases),'artifact_sha256':sha,
                    'cases':len(cases),'passed':passed,'accepted':passed==len(cases)}
            encoded=json.dumps(report,sort_keys=True)
            prior=self.db.execute('SELECT receipt FROM perception_reports_v08 WHERE id=?',(head,)).fetchone()
            if prior and prior!=(encoded,): raise ContractError('perception report changed')
            with self.db:
                if not prior: self.db.execute('INSERT INTO perception_reports_v08 VALUES(?,?)',(head,encoded))
                if report['accepted']:
                    current=self.db.execute('SELECT head FROM perception_bindings_v08 WHERE expert=?',(expert,)).fetchone()
                    if current and current!=(head,): raise ContractError('explicit head replacement not implemented')
                    self.db.execute('INSERT OR IGNORE INTO perception_bindings_v08 VALUES(?,?)',(expert,head))
            return report

    def _features(self, expert, e, target, *, version=1):
        m=self._core(expert); p=e['payload']; repair=target.startswith('repair/')
        impulse=p.get('impulse',{}); domain=p.get('domain',impulse.get('domain',{}))
        topic=domain.get('topic')
        if repair: topic=self.learning.catalog.manifest(target.split('/',1)[1])['family']
        values=[int(e['kind']=='KNOWLEDGE'),int(e['kind']=='RELEARN'),int(m['family']==topic),
                int(expert in impulse.get('affected',())),int(target=='local'),int(repair)]
        if version==2:
            recruit=e['kind']=='RECRUIT'
            values.extend((int(recruit),int(recruit and m['target_topic']==p['requested_topic'])))
        return values

    def propose(self, expert, event):
        """Only admitted local weights decide which group outputs activate."""
        with self.learning._guard():
            row=self.db.execute('SELECT head FROM perception_bindings_v08 WHERE expert=?',(expert,)).fetchone()
            if row is None: raise ContractError('no admitted perception head')
            e=self.bus.event(event['id'])
            if e!=event or not self.bus._live(e,self.bus._conflicts()): raise ContractError('stale perception event')
            owner,spec,sha,t=self._load(row[0])
            if owner!=expert: raise ContractError('perception binding mismatch')
            r=self.db.execute('SELECT receipt FROM perception_reports_v08 WHERE id=?',row).fetchone()
            if not r or not json.loads(r[0])['accepted'] or json.loads(r[0])['artifact_sha256']!=sha:
                raise ContractError('perception admission required')
            # Existing admitted heads never receive an out-of-contract event.
            version=2 if spec['contract']=='perception-v2' else 1
            if version==1 and e['kind']=='RECRUIT':
                return []
            choices=[]
            with torch.no_grad():
                for target in e['payload']['targets']:
                    x=torch.tensor(self._features(expert,e,target,version=version),dtype=torch.float32)
                    scores=t['w'] @ x+t['b']
                    if not torch.isfinite(scores).all(): raise ContractError('nonfinite perception output')
                    choices.extend((float(s),target,g) for s,g in zip(scores,spec['groups']) if float(s)>0)
            choices.sort(key=lambda c:(-c[0],c[1],c[2]))
            return [(target,g) for _,target,g in choices[:self.bus.memberships_per_expert]]

    def tick(self, *, delivery_budget=16):
        with self.learning._guard():
            residents=set(self.learning.population.resident_ids())
            experts=[e for (e,) in self.db.execute('SELECT expert FROM perception_bindings_v08') if e in residents]
        return self.bus.tick({e:lambda event,e=e:self.propose(e,event) for e in experts},delivery_budget=delivery_budget)
