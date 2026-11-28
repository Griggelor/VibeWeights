"""Actual tiny Qwen3 readout bridge, structured ports and evidence-gated shots."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.qwen3_fragments import FragmentStore
from vibeweights.fragment_adapters import FrozenPrefix,FragmentCompetence
from vibeweights.social_learning import SocialFragmentCore,SocialPopulation
from vibeweights.fragment_social_ports import pack_tokens,unpack_tokens,FragmentPortCore,OfflineMemo
from test_qwen3_a0 import snapshot,build
from test_fragment_adapters import fit,TEST


def test_packed_token_ports_and_invalid_inputs():
    assert unpack_tokens(pack_tokens([1,2],[3,4,5]))==((1,2),(3,4,5))
    for p in ([0,1,1,2],[1,1,3,2],[1,1,1,2,3],[1,1,1,True]):
        with pytest.raises(ContractError):unpack_tokens(p)
    with pytest.raises(ContractError):pack_tokens([1.],[2])
    with pytest.raises(ContractError):pack_tokens([1]*64,[2])


def test_actual_tiny_readouts_without_old_topic_gate(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    enc=FrozenPrefix(FragmentStore(root),2,pooling='token-l2-mean-v1')
    fit(enc,tmp_path/'adapter',conditioning='train-centered-radius-v1')
    expert=FragmentCompetence(enc,tmp_path/'adapter')
    first=FragmentPortCore(SocialFragmentCore('a',expert),port=0,scale=1)
    second=FragmentPortCore(SocialFragmentCore('b',expert),port=1,scale=.25)
    payload=pack_tokens(TEST[0][0],TEST[-1][0])
    assert first.contribute(payload)==(-1.,0.)
    assert second.contribute(payload)==(0.,.25)
    memo=OfflineMemo(first)
    assert memo.contribute(payload)==memo.contribute(pack_tokens(TEST[0][0],TEST[1][0]))
    assert (memo.calls,memo.forwards)==(2,1)
    store=Cooperation()
    try:
        pop=SocialPopulation(store,[first,second],width=2)
        r=pop.run('s','episode',[1.,0.],payload,
            lambda value:(value==(-1.,.25),1.-sum(abs(a-b) for a,b in zip(value,(-1.,.25)))/2),
            oracle_id='test',shots=2,seed=19)
        assert r['verified'] and r['released']==[-1.,.25]
        assert r['self_activated']==[] and set(r['shot_ids'])=={'a','b'}
        assert r['marginal']=={'a':.5,'b':.125}
        assert len(r['updates'])==4
    finally:store.close()
    first.port=1
    with pytest.raises(ContractError,match='port changed'):first.contribute(payload)
