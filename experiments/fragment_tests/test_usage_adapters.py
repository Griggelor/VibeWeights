"""Verified usage receipts, additive candidate training, and live evidence guards."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.fragment_adapters import FrozenPrefix,FragmentCompetence
from vibeweights.qwen3_fragments import FragmentStore
from vibeweights.social_learning import SocialFragmentCore,SocialPopulation
from vibeweights.fragment_social_ports import FragmentPortCore,pack_tokens
from vibeweights.usage_adapters import collect_usage,fit_usage_adapter
from test_qwen3_a0 import snapshot,build
from test_fragment_adapters import fit,TRAIN,TEST


def test_usage_training_reopen_and_source_invalidation(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    store=FragmentStore(root);enc=FrozenPrefix(store,2)
    fit(enc,tmp_path/'parent')
    parent=FragmentCompetence(enc,tmp_path/'parent')
    parent_sha=parent.m['adapter_sha256']
    cores=[FragmentPortCore(SocialFragmentCore(f'e{i}',parent),port=i,scale=s) for i,s in enumerate((1.,.25))]
    db=Cooperation(tmp_path/'usage.sqlite')
    ids=[]
    try:
        pop=SocialPopulation(db,cores,width=3,credit_policy='isolated-activation-v2')
        for i,(tokens,label) in enumerate(TRAIN):
            ep=f'usage-{i}';ids.append(ep);sign=2*label-1;target=(float(sign),.25*sign)
            def oracle(v):return v==target,max(-1.,1.-sum(abs(a-b) for a,b in zip(v,target))/2)
            receipt=pop.run('source',ep,[0.,0.,1.],pack_tokens(tokens,tokens),oracle,
                oracle_id='usage-oracle',shots=2,seed=i,learn=False)
            assert receipt['verified']
        bad=pop.run('source','bad',[0.,0.,1.],pack_tokens([7],[7]),lambda v:(False,0.),
            oracle_id='usage-oracle',shots=2,learn=False)
        assert not bad['verified']
    finally:db.close()
    db=Cooperation(tmp_path/'usage.sqlite')
    try:
        cases,proof=collect_usage(db,'source',ids,port=0,oracle_id='usage-oracle')
        assert cases==[(tuple(x),y) for x,y in TRAIN]
        with pytest.raises(ContractError,match='verified joint'):
            collect_usage(db,'source',ids[:3]+['bad'],port=0,oracle_id='usage-oracle')
        candidate=FrozenPrefix(store,2)
        out=tmp_path/'candidate'
        spec=fit_usage_adapter(db,candidate,'source',ids,TEST,port=0,oracle_id='usage-oracle',
            holdout_oracle='independent-test',output=out)
        assert spec['report']['admitted'] and spec['usage_evidence']==proof
        assert parent.m['adapter_sha256']==parent_sha
        with pytest.raises(ContractError,match='usage evidence request'):
            FragmentCompetence(candidate,out)
        loaded=FragmentCompetence(candidate,out,evidence_store=db)
        assert [loaded.predict('usage-candidate',x) for x,_ in TEST]==[0,0,1,1]
        with db.db:
            db.db.execute("UPDATE social_episodes_v1 SET receipt_sha='bad' WHERE session='source' AND episode='usage-0'")
        with pytest.raises(ContractError,match='receipt integrity'):
            FragmentCompetence(candidate,out,evidence_store=db)
        with pytest.raises(ContractError,match='receipt integrity'):
            loaded.predict('usage-candidate',TEST[0][0])
        assert parent.predict('task',TEST[0][0])==0
    finally:db.close()
