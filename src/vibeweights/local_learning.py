"""Local learnability probes over verified usage, not a population router.

A candidate fragment independently scans a bounded recent experience window
and probes each compatible port. Neither the host nor the final holdout
selects a port. The request gate is an internal calibration test, never
competence admission. Existing A1 admission remains unchanged.
"""
import json
import torch
from .abi import ContractError
from .coalition import Cooperation
from .environment import identifier
from .qwen3_fragments import _hash
from .fragment_social_ports import unpack_tokens
from .fragment_adapters import FrozenPrefix,_fit,_evaluate,condition_training,fold_conditioning
from .usage_adapters import collect_usage,validate_usage,fit_usage_adapter

CONTRACT='local-usage-request-v1'
MIN_PER_CLASS=8
CAL_PER_CLASS=4
MIN_CAL_PASSED=6


def _window(cooperation,session,oracle_id,scan_limit):
    if not isinstance(cooperation,Cooperation) or type(scan_limit) is not int or not 1<=scan_limit<=4096:
        raise ContractError('learning request scan budget')
    identifier(session);identifier(oracle_id)
    rows=cooperation.db.execute(
        'SELECT rowid,episode,spec_sha,receipt,receipt_sha FROM social_episodes_v1 '
        'WHERE session=? ORDER BY rowid DESC LIMIT ?', (session,scan_limit+1)).fetchall()
    truncated=len(rows)>scan_limit
    eligible=[];counts={'scanned':min(len(rows),scan_limit),'outside_window':truncated,
                       'other_oracle':0,'unverified':0,'other_impulse':0,'not_cooperative':0}
    for _,episode,sha,raw,digest in reversed(rows[:scan_limit]):
        v=json.loads(raw)
        if (_hash(v)!=digest or v.get('spec_sha256')!=sha or v.get('session')!=session
            or v.get('episode')!=episode or v.get('contract')!='social-episodes-v1'):
            raise ContractError('learning request source integrity')
        if v['oracle_id']!=oracle_id:counts['other_oracle']+=1;continue
        if v['verified'] is not True:counts['unverified']+=1;continue
        if v['impulse']!=[0.,0.,1.]:counts['other_impulse']+=1;continue
        if sum(x>0 for x in v['marginal'].values())<2:counts['not_cooperative']+=1;continue
        if (v['released']!=v['candidate'] or len(v['released'])!=2
            or any(v['released'][i] not in (-s,s) for i,s in enumerate((1.,.25)))):
            raise ContractError('learning request source output')
        eligible.append((episode,unpack_tokens(v['payload']),v['released']))
    counts['eligible']=len(eligible)
    return eligible,counts


def _select(eligible,port,case_budget):
    # Latest occurrence of each distinct token input. Conflicting verified
    # labels are not reconciled by guessing: fail closed.
    seen={}
    duplicates=0
    for ep,tokens,release in eligible:
        x=tokens[port];y=int(release[port]>0)
        if x in seen:
            if seen[x][1]!=y:raise ContractError('learning request conflicting labels')
            duplicates+=1
            del seen[x]
        seen[x]=(ep,y)
    selected=list(seen.values())[-case_budget:]
    return [v[0] for v in selected],{'unique':len(seen),'duplicates':duplicates,
        'excluded_by_case_budget':max(0,len(seen)-case_budget),
        'class_counts':[sum(y==k for _,y in selected) for k in (0,1)]}


def _probe(encoder,cases):
    train=[];cal=[]
    for label in (0,1):
        ids=[i for i,(_,y) in enumerate(cases) if y==label]
        train.extend(ids[:-CAL_PER_CLASS]);cal.extend(ids[-CAL_PER_CLASS:])
    train.sort();cal.sort()
    x=torch.stack([encoder.features(v) for v,_ in cases])
    y=torch.tensor([v for _,v in cases])
    normalized,center,gain,stats=condition_training(x[train],'train-centered-radius-v1')
    w,b,_=_fit(normalized,y[train],64,regularization=.001)
    w,b=fold_conditioning(w,b,center,gain)
    a=_evaluate(x[train],y[train],w,b);c=_evaluate(x[cal],y[cal],w,b)
    per_class=[sum(p==t==k for p,t in zip(c['predictions'],c['expected'])) for k in (0,1)]
    requested=(a['passed']==a['cases'] and c['passed']>=MIN_CAL_PASSED and min(per_class)>=2)
    return {'train_indices':train,'calibration_indices':cal,'training':a,'calibration':c,
            'calibration_correct_by_class':per_class,'conditioning':stats,'requested':requested}


def _table(db):
    db.execute('''CREATE TABLE IF NOT EXISTS local_usage_requests_v1(
        id TEXT PRIMARY KEY, receipt TEXT NOT NULL, sha TEXT NOT NULL)''')


def _persist(db,receipt):
    key=_hash(receipt);raw=json.dumps(receipt,sort_keys=True,separators=(',',':'))
    prior=db.execute('SELECT receipt,sha FROM local_usage_requests_v1 WHERE id=?',(key,)).fetchone()
    if prior is not None:
        if prior!=(raw,key):raise ContractError('learning request integrity')
    else:db.execute('INSERT INTO local_usage_requests_v1 VALUES(?,?,?)',(key,raw,key))
    return {'id':key,**receipt}


def propose_usage(cooperation,encoder,session,*,oracle_id,scan_limit=1024,case_budget=64):
    """No holdout inputs accepted. Every compatible port independently considered."""
    if not isinstance(encoder,FrozenPrefix) or type(case_budget) is not int or not 16<=case_budget<=128:
        raise ContractError('learning request candidate/budget')
    eligible,counts=_window(cooperation,session,oracle_id,scan_limit)
    results=[]
    for port in (0,1):
        ids,coverage=_select(eligible,port,case_budget)
        spec={'contract':CONTRACT,'core':encoder.ref,'session':session,'oracle_id':oracle_id,
              'port':port,'scan_limit':scan_limit,'case_budget':case_budget,
              'coverage':{**counts,**coverage},'policy':{'min_per_class':MIN_PER_CLASS,
                'cal_per_class':CAL_PER_CLASS,'min_cal_passed':MIN_CAL_PASSED}}
        if min(coverage['class_counts'])<MIN_PER_CLASS:
            results.append({**spec,'status':'INSUFFICIENT_EVIDENCE','requested':False})
            continue
        cases,proof=collect_usage(cooperation,session,ids,port=port,oracle_id=oracle_id)
        probe=_probe(encoder,cases)
        results.append({**spec,'status':'REQUESTED' if probe['requested'] else 'ABSTAINED',
                        'requested':probe['requested'],'evidence':proof,'probe':probe})
    db=cooperation.db
    db.execute('BEGIN IMMEDIATE')
    try:
        _table(db)
        for receipt in results:
            if 'evidence' in receipt:validate_usage(cooperation,receipt['evidence'])
        saved=[_persist(db,v) for v in results]
        db.commit()
        return saved
    except BaseException:
        db.rollback();raise


def fit_requested_adapter(cooperation,encoder,request_id,holdout,*,holdout_oracle,output):
    """Execute a locally requested learning job against a separate final oracle."""
    if not isinstance(cooperation,Cooperation) or not isinstance(encoder,FrozenPrefix):
        raise ContractError('learning request executor')
    identifier(request_id)
    db=cooperation.db
    _table(db)
    row=db.execute('SELECT receipt,sha FROM local_usage_requests_v1 WHERE id=?',(request_id,)).fetchone()
    if row is None:raise ContractError('learning request missing')
    receipt=json.loads(row[0])
    if _hash(receipt)!=row[1] or row[1]!=request_id or receipt.get('contract')!=CONTRACT:
        raise ContractError('learning request integrity')
    if receipt['status']!='REQUESTED' or receipt['requested'] is not True:
        raise ContractError('learning request not authorized by local probe')
    if receipt['core']!=encoder.ref:raise ContractError('learning request core mismatch')
    proof=receipt['evidence'];validate_usage(cooperation,proof)
    return fit_usage_adapter(cooperation,encoder,proof['session'],[v['episode'] for v in proof['sources']],
        holdout,port=proof['port'],oracle_id=proof['oracle_id'],holdout_oracle=holdout_oracle,output=output)
