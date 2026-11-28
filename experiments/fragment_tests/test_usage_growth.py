"""Actually admitted usage readout added without invalidating old receipt proofs."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.fragment_adapters import FrozenPrefix,FragmentCompetence
from vibeweights.qwen3_fragments import FragmentStore,_hash
from vibeweights.social_learning import SocialFragmentCore,SocialPopulation
from vibeweights.fragment_social_ports import FragmentPortCore,pack_tokens
from vibeweights.usage_adapters import collect_usage,fit_usage_adapter,validate_usage
from test_qwen3_a0 import snapshot,build
from test_fragment_adapters import fit,TRAIN,TEST


def test_usage_readout_growth_preserves_sources(snapshot,tmp_path):
    root=tmp_path/'fragments';build(snapshot,root)
    enc=FrozenPrefix(FragmentStore(root),2)
    fit(enc,tmp_path/'parent')
    parent=FragmentCompetence(enc,tmp_path/'parent')
    a=FragmentPortCore(SocialFragmentCore('a',parent),port=0,scale=1)
    b=FragmentPortCore(SocialFragmentCore('b',parent),port=1,scale=.25)
    db=Cooperation(tmp_path/'usage.sqlite')
    try:
        p=SocialPopulation(db,[a,b],width=3,credit_policy='isolated-activation-v2')
        def run(pop,ep,x,label,**kw):
            sign=2*label-1;target=(float(sign),.25*sign)
            def oracle(v):return v==target,max(-1.,1.-sum(abs(a-b) for a,b in zip(v,target))/2)
            return pop.run('s',ep,[0.,0.,1.],pack_tokens(x,x),oracle,oracle_id='source',**kw)
        ids=[]
        for i,(x,label) in enumerate(TRAIN):
            ep=f'usage-{i}';ids.append(ep)
            assert run(p,ep,x,label,all_active=True)['verified']
        training,proof=collect_usage(db,'s',ids,port=1,oracle_id='source')
        candidate=FrozenPrefix(enc.store,2)
        out=tmp_path/'candidate'
        spec=fit_usage_adapter(db,candidate,'s',ids,TEST,port=1,oracle_id='source',
            holdout_oracle='fresh',output=out)
        assert spec['report']['admitted']
        new=FragmentCompetence(candidate,out,evidence_store=db)
        c=FragmentPortCore(SocialFragmentCore('c',new),port=1,scale=.25)
        before=db.db.execute('SELECT * FROM social_heads_v1 ORDER BY owner,peer,revision').fetchall()
        grown=p.extend('s',[c],expected_sha=_hash(p.spec))
        assert before==db.db.execute('SELECT * FROM social_heads_v1 ORDER BY owner,peer,revision').fetchall()
        assert validate_usage(db,proof)==training
        assert run(grown,'usage-0',*TRAIN[0],all_active=True)['historical']
        # A redundant shot must not be blindly released as a valid answer.
        dup=run(grown,'duplicate',*TEST[-1],all_active=True,learn=False)
        assert not dup['verified'] and dup['released'] is None
        assert dup['population_sha256']==_hash(grown.spec)
        assert c.contribute(pack_tokens(TEST[-1][0],TEST[-1][0]))==(0.,.25)
        # New receipt references its own revision, old proof remains byte-stable.
        for i in range(10):run(grown,f'practice-{i}',*TRAIN[i%4],learn=True)
        fresh=run(grown,'new-revision',*TEST[-1],learn=False)
        assert fresh['verified'] and fresh['members']==['a','b']
        mixed,mp=collect_usage(db,'s',ids[:3]+['new-revision'],port=1,oracle_id='source')
        assert mp['sources'][-1]['population_sha256']==_hash(grown.spec)
        assert validate_usage(db,proof)==training
        with db.db:
            db.db.execute("UPDATE social_episodes_v1 SET receipt_sha='bad' WHERE session='s' AND episode='usage-0'")
        with pytest.raises(ContractError,match='receipt integrity'):
            c.contribute(pack_tokens(TEST[-1][0],TEST[-1][0]))
    finally:db.close()
