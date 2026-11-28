"""A1 experimental adapter admission, head independence and negative controls."""
import pytest
import torch
from vibeweights.abi import ContractError
from vibeweights.qwen3_fragments import FragmentStore
from vibeweights.fragment_adapters import FrozenPrefix, fit_experiment, FragmentCompetence
from test_qwen3_a0 import snapshot, build

TRAIN=[([7],0),([7,7],0),([9],1),([9,9],1)]
TEST=[([7]*3,0),([7]*4,0),([9]*3,1),([9]*4,1)]


def fit(encoder,root,training=TRAIN,holdout=TEST,**kw):
    return fit_experiment(encoder,training,holdout,topic='task',
        training_oracle='train-v1',holdout_oracle=kw.pop('holdout_oracle','holdout-v1'),
        output=root,**kw)


def test_frozen_prefix_admission_and_local_abstention(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    store=FragmentStore(root);enc=FrozenPrefix(store,2)
    weights=[p.detach().clone() for layer in enc.span.layers for p in layer.parameters()]
    spec=fit(enc,tmp_path/'adapter')
    assert spec['report']['admitted']
    assert spec['report']['holdout']['passed']==4
    assert spec['report']['head_holdout']['passed']==8
    assert spec['adapter_parameters']==38
    assert all(torch.equal(a,b) for a,b in zip(weights,[p for layer in enc.span.layers for p in layer.parameters()]))
    snapshot.rename(tmp_path/'unavailable-source')
    expert=FragmentCompetence(FrozenPrefix(FragmentStore(root),2),tmp_path/'adapter')
    assert [expert.predict('task',x,priority=True) for x,_ in TEST]==[0,0,1,1]
    assert expert.predict('wrong',TEST[0][0],priority=True) is None
    assert expert.predict('task',None,priority=True) is None
    with pytest.raises(ContractError,match='core integrity'):
        FragmentCompetence(FrozenPrefix(FragmentStore(root),1),tmp_path/'adapter')


def test_equal_budget_controls_and_rejection(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    store=FragmentStore(root);enc=FrozenPrefix(store,2)
    for mode in ('random','embedding-only'):
        other=FrozenPrefix(store,2,mode=mode)
        assert other.core_bytes==(enc.core_bytes if mode=='random' else 0)
        spec=fit(other,tmp_path/mode)
        assert spec['adapter_parameters']==38 and not spec['report']['admitted']
        with pytest.raises(ContractError,match='not admitted'):
            FragmentCompetence(other,tmp_path/mode)
    wrong=[(x,1-y) for x,y in TEST]
    rejected=fit(enc,tmp_path/'rejected',holdout=wrong)
    assert not rejected['report']['admitted']
    with pytest.raises(ContractError,match='not admitted'):
        FragmentCompetence(enc,tmp_path/'rejected')


def test_holdout_integrity_and_artifact_corruption(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    enc=FrozenPrefix(FragmentStore(root),2)
    with pytest.raises(ContractError,match='independent adapter oracle'):
        fit(enc,tmp_path/'bad-oracle',holdout_oracle='train-v1')
    with pytest.raises(ContractError,match='disjoint adapter inputs'):
        fit(enc,tmp_path/'bad-split',holdout=TRAIN)
    assert not (tmp_path/'bad-split').exists()
    fit(enc,tmp_path/'adapter')
    path=tmp_path/'adapter'/'adapter.safetensors'
    raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    with pytest.raises(ContractError,match='artifact integrity'):
        FragmentCompetence(enc,tmp_path/'adapter')
