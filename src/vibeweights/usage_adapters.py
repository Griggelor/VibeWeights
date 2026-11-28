"""Train a new fragment readout from committed verified social episodes.

Restricted two-port reference ABI. The trainer reads only released values and
native token payloads from hash-checked receipts, never parent readout weights
or unverified candidate outputs. Host selects evidence and port; not autonomous
discovery. Trusted local SQLite integrity, not a cryptographic signature.
"""
import json
import math
from .abi import ContractError
from .coalition import Cooperation
from .environment import identifier
from .qwen3_fragments import _hash
from .fragment_social_ports import unpack_tokens
from . import social_revisions

CONTRACT='usage-readout-evidence-v1'


def collect_usage(cooperation,session,episode_ids,*,port,oracle_id):
    """Return distinct binary training cases and an immutable evidence descriptor."""
    if (not isinstance(cooperation,Cooperation) or type(port) is not int or port not in (0,1)
        or type(episode_ids) not in (list,tuple) or not 4<=len(episode_ids)<=128
        or len(set(episode_ids))!=len(episode_ids)):
        raise ContractError('usage evidence request')
    for s in (session,oracle_id,*episode_ids):identifier(s)
    db=cooperation.db
    chain=social_revisions.history(db,session)
    if not chain:raise ContractError('usage source population missing')
    spec,population_sha=chain[0]
    if (_hash(spec)!=population_sha or spec.get('contract')!='social-episodes-v1'
        or spec.get('width')!=3 or spec.get('output_width')!=2
        or not isinstance(spec.get('cores'),dict)):
        raise ContractError('usage source population integrity')
    cases=[];refs=[];seen=set()
    for episode in episode_ids:
        raw=db.execute('SELECT spec_sha,receipt,receipt_sha FROM social_episodes_v1 WHERE session=? AND episode=?',(session,episode)).fetchone()
        if raw is None:raise ContractError('usage receipt missing')
        receipt=json.loads(raw[1])
        if (_hash(receipt)!=raw[2] or receipt.get('spec_sha256')!=raw[0]
            or receipt.get('contract')!='social-episodes-v1'
            or receipt.get('session')!=session or receipt.get('episode')!=episode
            or receipt.get('oracle_id')!=oracle_id):
            raise ContractError('usage receipt integrity')
        if (receipt.get('verified') is not True or receipt.get('impulse')!=[0.,0.,1.]
            or receipt.get('released')!=receipt.get('candidate')
            or type(receipt.get('released')) is not list or len(receipt['released'])!=2):
            raise ContractError('usage requires verified joint output')
        revision,revision_sha=social_revisions.historical(chain,receipt)
        released=receipt['released'];contrib=receipt.get('contributions');marginal=receipt.get('marginal')
        if (not isinstance(contrib,dict) or not isinstance(marginal,dict)
            or set(contrib)!=set(marginal) or set(contrib)!=set(receipt.get('members',[]))
            or len(contrib)<2 or not set(contrib)<=set(revision['cores'])
            or any(type(v) is not list or len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for v in contrib.values())):
            raise ContractError('usage contribution provenance')
        if ([math.fsum(v[i] for v in contrib.values()) for i in range(2)]!=released
            or any(released[i] not in (-s,s) for i,s in enumerate((1.,.25)))
            or sum(type(x) in (float,int) and math.isfinite(x) and x>0 for x in marginal.values())<2):
            raise ContractError('usage verified cooperation required')
        tokens=unpack_tokens(receipt['payload'])[port]
        if tokens in seen:raise ContractError('usage duplicate training input')
        seen.add(tokens);cases.append((tokens,int(released[port]>0)))
        source={'episode':episode,'receipt_sha256':raw[2],
                'contributors':{k:revision['cores'][k] for k in sorted(contrib)}}
        if revision_sha!=population_sha:source['population_sha256']=revision_sha
        refs.append(source)
    if {y for _,y in cases}!={0,1}:raise ContractError('usage both classes required')
    proof={'contract':CONTRACT,'session':session,'oracle_id':oracle_id,'port':port,
           'population_sha256':population_sha,'sources':refs,'training_sha256':_hash(cases)}
    return cases,proof


def validate_usage(cooperation,proof):
    if not isinstance(proof,dict) or proof.get('contract')!=CONTRACT or not isinstance(proof.get('sources'),list):
        raise ContractError('usage proof contract')
    cases,current=collect_usage(cooperation,proof['session'],[v['episode'] for v in proof['sources']],
                                port=proof['port'],oracle_id=proof['oracle_id'])
    if current!=proof:raise ContractError('usage evidence changed')
    return cases


def fit_usage_adapter(cooperation,encoder,session,episode_ids,holdout,*,port,oracle_id,
                      holdout_oracle,output,topic='usage-candidate',steps=64):
    from .fragment_adapters import fit_experiment
    training,proof=collect_usage(cooperation,session,episode_ids,port=port,oracle_id=oracle_id)
    return fit_experiment(encoder,training,holdout,topic=topic,training_oracle=oracle_id,
        holdout_oracle=holdout_oracle,output=output,steps=steps,
        conditioning='train-centered-radius-v1',usage_evidence=proof,evidence_store=cooperation)
