"""Local request discovery without caller-selected training cases or ports."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.fragment_adapters import FrozenPrefix,FragmentCompetence
from vibeweights.qwen3_fragments import FragmentStore,_hash
from vibeweights.fragment_social_ports import pack_tokens,unpack_tokens
from vibeweights.social_learning import SocialPopulation
from vibeweights.local_learning import propose_usage,fit_requested_adapter
from test_qwen3_a0 import snapshot,build


class TokenCore:
    def __init__(self,port):
        self.port=port;self.id=f'peer-{port}';self.output_width=2
        self.artifact_sha256=_hash({'synthetic-token-core':port})
    def contribute(self,payload):
        x=unpack_tokens(payload)[self.port]
        sign=1. if x[0]==9 else -1.
        return (sign,0.) if self.port==0 else (0.,.25*sign)


def observe(pop,ep,x,*,verified=True):
    label=x[0]==9;sign=1. if label else -1.;target=(sign,.25*sign)
    def oracle(v):return (verified and v==target),max(-1.,1.-sum(abs(a-b) for a,b in zip(v,target))/2)
    return pop.run('source',ep,[0.,0.,1.],pack_tokens(x,x),oracle,
        oracle_id='test-oracle',shots=2,seed=1,learn=False)


def test_discovery_calibration_and_restart(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    enc=FrozenPrefix(FragmentStore(root),2,pooling='token-l2-mean-v1')
    path=tmp_path/'usage.sqlite';db=Cooperation(path)
    try:
        pop=SocialPopulation(db,[TokenCore(0),TokenCore(1)],width=3,credit_policy='isolated-activation-v2')
        observe(pop,'first',[7])
        r=propose_usage(db,enc,'source',oracle_id='test-oracle')
        assert [x['status'] for x in r]==['INSUFFICIENT_EVIDENCE']*2
        for token in (7,9):
            for n in range(1,9):
                assert observe(pop,f'{token}-{n}',[token]*n)['verified']
        observe(pop,'rejected',[9]*9,verified=False)
        r=propose_usage(db,enc,'source',oracle_id='test-oracle')
        assert len(r)==2 and all(x['coverage']['class_counts']==[8,8] for x in r)
        assert all(x['coverage']['duplicates']==1 and x['coverage']['unverified']==1 for x in r)
        assert all(x['requested'] for x in r)
        assert all(len(x['probe']['calibration_indices'])==8 for x in r)
        assert propose_usage(db,enc,'source',oracle_id='test-oracle')==r
    finally:db.close()
    db=Cooperation(path)
    try:
        hold=[([7]*9,0),([7]*10,0),([9]*9,1),([9]*10,1)]
        out=tmp_path/'candidate'
        spec=fit_requested_adapter(db,enc,r[0]['id'],hold,holdout_oracle='fresh',output=out)
        assert spec['report']['admitted']
        e=FragmentCompetence(enc,out,evidence_store=db)
        assert [e.predict('usage-candidate',x) for x,_ in hold]==[0,0,1,1]
        with db.db:
            db.db.execute("UPDATE social_episodes_v1 SET receipt_sha='bad' WHERE session='source' AND episode='7-1'")
        with pytest.raises(ContractError,match='receipt integrity'):
            fit_requested_adapter(db,enc,r[0]['id'],hold,holdout_oracle='fresh',output=tmp_path/'invalid')
        with pytest.raises(ContractError,match='source integrity'):
            propose_usage(db,enc,'source',oracle_id='test-oracle')
    finally:db.close()


def test_no_execution_from_insufficient_request(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    enc=FrozenPrefix(FragmentStore(root),2)
    db=Cooperation()
    try:
        pop=SocialPopulation(db,[TokenCore(0),TokenCore(1)],width=3)
        observe(pop,'one',[7])
        r=propose_usage(db,enc,'source',oracle_id='test-oracle')
        with pytest.raises(ContractError,match='not authorized'):
            fit_requested_adapter(db,enc,r[0]['id'],[],holdout_oracle='fresh',output=tmp_path/'bad')
        assert not (tmp_path/'bad').exists()
    finally:db.close()
