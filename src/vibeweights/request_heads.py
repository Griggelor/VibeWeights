"""Independent trainable, core-bound local request heads (CPU reference)."""
import hashlib
import json
import torch
from safetensors.torch import save, load
from .abi import ContractError
from .environment import identifier
from .knowledge_learning import _hash
from .population import RESIDENT
from .perception import _samples

FEATURES=('input_missing','output_missing','recruit_event','goal_is_output')
CONTRACT='request-head-v1'


def _core(learning,expert):
    m=learning.catalog.manifest(expert)
    if m['state'] not in RESIDENT or expert not in learning.population.resident_ids():
        raise ContractError('resident request core required')
    if m['abi']!='numeric-events/0.1' or m['op'] not in ('affine','quadratic','mlp') or m['target_scope']!='global':
        raise ContractError('unsupported request core')
    return m


def _load(learning,head):
    row=learning.db.execute('SELECT expert,core_sha,spec,weights,sha FROM request_heads_v08 WHERE id=?',(head,)).fetchone()
    if row is None: raise ContractError('unknown request head')
    expert,core,raw,blob,sha=row
    spec=json.loads(raw)
    if len(blob)>16384 or _hash([expert,core,spec,hashlib.sha256(blob).hexdigest()])!=sha:
        raise ContractError('request artifact integrity')
    if _core(learning,expert)['sha256']!=core:
        raise ContractError('request core changed')
    if spec.get('contract')!=CONTRACT or spec.get('features')!=list(FEATURES):
        raise ContractError('request feature ABI')
    t=load(blob)
    if (set(t)!={'w','b'} or t['w'].shape!=(1,len(FEATURES)) or t['b'].shape!=(1,) or
        any(x.dtype!=torch.float32 or not torch.isfinite(x).all() for x in t.values())):
        raise ContractError('request weight contract')
    return expert,spec,sha,t


def admitted(learning,expert):
    """Read-only validation; missing binding is abstention, corruption is error."""
    db=learning.db
    row=db.execute('SELECT head FROM request_bindings_v08 WHERE expert=?',(expert,)).fetchone()
    if row is None:return None
    head=row[0]
    owner,spec,sha,t=_load(learning,head)
    if owner!=expert: raise ContractError('request binding mismatch')
    r=db.execute('SELECT receipt,sha FROM request_reports_v08 WHERE id=?',(head,)).fetchone()
    if r is None: raise ContractError('request admission required')
    report=json.loads(r[0])
    if (_hash(report)!=r[1] or report.get('artifact_sha256')!=sha or report.get('accepted') is not True or
        report.get('passed')!=report.get('cases') or not report.get('cases')):
        raise ContractError('request admission integrity')
    return {'expert':expert,'head':head,'sha256':sha},t


def decide(learning,expert,features):
    """Return local weight decision and immutable reference; never a route."""
    if type(features) not in (tuple,list) or len(features)!=len(FEATURES) or any(type(v) is not int or v not in (0,1) for v in features):
        raise ContractError('request input ABI')
    entry=admitted(learning,expert)
    if entry is None:return None
    ref,t=entry
    with torch.inference_mode():
        score=t['w'] @ torch.tensor(features,dtype=torch.float32)+t['b']
        if not torch.isfinite(score).all():raise ContractError('nonfinite request decision')
        return {**ref,'features':list(features),'request':bool(score.item()>0)}


def local_features(manifest,scope,facts,event_kind,goal):
    """Only port presence in this scope and two bits of task metadata."""
    topics={f['topic'] for f in facts if f['scope']==scope}
    return [int(manifest['source_topic'] not in topics),
            int(manifest['target_topic'] not in topics),
            int(event_kind=='RECRUIT'),int(goal==manifest['target_topic'])]


def validate_intent(learning,receipt,event_kind):
    """Recompute all bound decisions before dispatch/liveness; no fallback.

    Unbound experts abstain. A head added after the historical run is not
    retroactive: only original bindings are validated, not new decisions.
    """
    intent=receipt.get('recruitment',{})
    if intent.get('contract')!='learned-recruitment-v1':
        raise ContractError('learned recruitment contract required')
    refs=intent.get('decisions')
    if type(refs) is not list:
        raise ContractError('learned recruitment decisions')
    members=[m['expert'] for m in receipt['members']]
    if len(refs)>len(members) or len({d['expert'] for d in refs})!=len(refs):
        raise ContractError('duplicate or excess request decisions')
    requesters=[]
    for d in refs:
        expert=d['expert']
        if expert not in members:raise ContractError('request decision outside group')
        m=_core(learning,expert)
        features=local_features(m,receipt['scope'],receipt['result']['facts'],event_kind,receipt['goal'])
        actual=decide(learning,expert,features)
        if actual!=d:raise ContractError('stale learned request decision')
        if d['request'] and features[0]:requesters.append(expert)
    if requesters!=intent.get('requesters'):
        raise ContractError('learned request list mismatch')
    return tuple(requesters)


class RequestHeads:
    def __init__(self,learning):
        self.learning=learning;self.db=learning.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS request_heads_v08(
          id TEXT PRIMARY KEY, expert TEXT NOT NULL, core_sha TEXT NOT NULL,
          spec TEXT NOT NULL, weights BLOB NOT NULL, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS request_reports_v08(
          id TEXT PRIMARY KEY REFERENCES request_heads_v08(id), receipt TEXT NOT NULL, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS request_bindings_v08(
          expert TEXT PRIMARY KEY, head TEXT UNIQUE NOT NULL REFERENCES request_heads_v08(id));
        ''')

    def stage(self,head,expert,cases,*,steps=512):
        identifier(head);identifier(expert)
        if type(steps) is not int or not 1<=steps<=512:raise ContractError('request training budget')
        x,y=_samples(cases,1,FEATURES)
        if not 0<int(y.sum())<len(y):raise ContractError('request training requires positive and negative cases')
        with self.learning._guard():
            core=_core(self.learning,expert)['sha256']
            if self.db.execute('SELECT 1 FROM request_heads_v08 WHERE id=?',(head,)).fetchone():
                raise ContractError('request head identity reserved')
        w=torch.zeros((1,len(FEATURES)),requires_grad=True);b=torch.zeros(1,requires_grad=True)
        opt=torch.optim.SGD((w,b),lr=0.5)
        def loss():return torch.nn.functional.binary_cross_entropy_with_logits(x@w.T+b,y)
        before=float(loss().detach())
        for _ in range(steps):
            opt.zero_grad();loss().backward();opt.step()
        after=float(loss().detach())
        blob=save({'w':w.detach(),'b':b.detach()})
        spec={'contract':CONTRACT,'features':list(FEATURES),'origin':'TRAINED','steps':steps,
              'training_sha256':_hash(cases),'training_inputs':sorted(_hash(list(v)) for v,_ in cases),
              'loss_before':before,'loss_after':after,'optimizer':'SGD','learning_rate':0.5,
              'torch_version':str(torch.__version__)}
        sha=_hash([expert,core,spec,hashlib.sha256(blob).hexdigest()])
        with self.learning._guard():
            if _core(self.learning,expert)['sha256']!=core:raise ContractError('request core changed')
            with self.db:
                self.db.execute('INSERT INTO request_heads_v08 VALUES(?,?,?,?,?,?)',
                                (head,expert,core,json.dumps(spec,sort_keys=True),blob,sha))
        return {'head':head,'expert':expert,'sha256':sha,**spec}

    def admit(self,head,cases,*,oracle_id):
        identifier(oracle_id)
        with self.learning._guard():
            expert,spec,sha,t=_load(self.learning,head)
            x,y=_samples(cases,1,FEATURES)
            if set(spec['training_inputs']).intersection(_hash(list(v)) for v,_ in cases):
                raise ContractError('disjoint request acceptance inputs required')
            if not 0<int(y.sum())<len(y):raise ContractError('request acceptance requires positive and negative cases')
            # Metadata gate forbids learning to request an already present input.
            if any(not v[0] and label[0] for v,label in cases):
                raise ContractError('request acceptance contradicts input contract')
            with torch.inference_mode():
                passed=int(((x@t['w'].T+t['b']>0)==y.bool()).all(dim=1).sum())
            report={'oracle_id':oracle_id,'case_sha256':_hash(cases),'artifact_sha256':sha,
                    'cases':len(cases),'passed':passed,'accepted':passed==len(cases)}
            raw=json.dumps(report,sort_keys=True);digest=_hash(report)
            prior=self.db.execute('SELECT receipt,sha FROM request_reports_v08 WHERE id=?',(head,)).fetchone()
            if prior and prior!=(raw,digest):raise ContractError('request report changed')
            with self.db:
                if not prior:self.db.execute('INSERT INTO request_reports_v08 VALUES(?,?,?)',(head,raw,digest))
                if report['accepted']:
                    current=self.db.execute('SELECT head FROM request_bindings_v08 WHERE expert=?',(expert,)).fetchone()
                    if current and current!=(head,):raise ContractError('explicit request head replacement not implemented')
                    self.db.execute('INSERT OR IGNORE INTO request_bindings_v08 VALUES(?,?)',(expert,head))
            return report
