"""Train-only conditioning, representation binding and unchanged admission gate."""
import pytest
import torch
from vibeweights.abi import ContractError
from vibeweights.qwen3_fragments import FragmentStore
from vibeweights.fragment_adapters import (FrozenPrefix, FragmentCompetence,
    pool_latent, condition_training, fold_conditioning)
from test_qwen3_a0 import snapshot,build
from test_fragment_adapters import fit,TRAIN,TEST


def test_token_normalization_prevents_norm_dominance():
    h=torch.tensor([[[1000.,0.],[0.,1.]]])
    a=pool_latent(h,'mean-l2-v1')
    b=pool_latent(h,'token-l2-mean-v1')
    assert a[0]>.999 and a[1]<.002
    assert torch.allclose(b,torch.tensor([2**-.5,2**-.5]))
    assert torch.equal(pool_latent(h,'last-l2-v1'),torch.tensor([0.,1.]))
    with pytest.raises(ContractError):pool_latent(h,'unknown')


def test_train_only_centered_radius_and_export_equivalence():
    x=torch.tensor([[1.,.001],[1.,.002],[1.,.003],[1.,.004]])
    z,c,g,stats=condition_training(x,'train-centered-radius-v1')
    assert torch.allclose(z.mean(0),torch.zeros(2,dtype=torch.float64),atol=1e-12)
    assert abs(float(z.square().sum(1).mean())-1)<1e-10
    assert 1<g<=1000 and stats['gain']==g
    w=torch.tensor([.1,2.]);b=torch.tensor([.5])
    fw,fb=fold_conditioning(w,b,c,g)
    probe=torch.tensor([[1.,.005],[1.,.006]])
    expected=((probe.double()-c)*g)@w.double()+b.double()
    assert torch.allclose((probe@fw+fb).double(),expected,atol=1e-4)
    _,_,gain,_=condition_training(torch.ones(4,2),'train-centered-radius-v1')
    assert gain==1000
    with pytest.raises(ContractError):condition_training(torch.tensor([[float('nan')]]),'none')


def test_conditioned_candidate_and_independent_holdout(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    store=FragmentStore(root);enc=FrozenPrefix(store,2,pooling='token-l2-mean-v1')
    a=fit(enc,tmp_path/'adapter',conditioning='train-centered-radius-v1')
    b=fit(enc,tmp_path/'wrong',holdout=[(x,1-y) for x,y in TEST],
          conditioning='train-centered-radius-v1')
    assert a['report']['adapter_fit']==b['report']['adapter_fit']
    assert a['adapter_sha256']==b['adapter_sha256']
    assert not b['report']['admitted']
    assert a['report']['admitted']
    assert a['adapter_parameters']==38
    expert=FragmentCompetence(enc,tmp_path/'adapter')
    assert [expert.predict('task',x,priority=True) for x,_ in TEST]==[0,0,1,1]
    with pytest.raises(ContractError,match='core integrity'):
        FragmentCompetence(FrozenPrefix(store,2),tmp_path/'adapter')
    with pytest.raises(ContractError,match='not admitted'):
        FragmentCompetence(enc,tmp_path/'wrong')
