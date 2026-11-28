"""Append-only population growth, stable social history and source revisions."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.social_learning import SocialPopulation,AffineSocialCore
from vibeweights.qwen3_fragments import _hash
from vibeweights.social_revisions import history


def cores():
    return [AffineSocialCore('a',[[1.],[0.]],[0.,0.]),
            AffineSocialCore('b',[[0.],[.25]],[0.,0.])]


def new_core():
    return AffineSocialCore('c',[[-1.],[0.]],[0.,0.])


def episode(pop,id,goal,m=1,**kw):
    z=(1.,0.) if goal==0 else (0.,1.)
    target=(float(m),.25*m) if goal==0 else (-float(m),0.)
    def oracle(v):
        return v==target,max(-1.,1.-sum(abs(a-b) for a,b in zip(v,target))/(2*m))
    return pop.run('s',id,z,[m],oracle,oracle_id='test',**kw)


def test_additive_growth_retains_history_and_learns_new_expert(tmp_path):
    path=tmp_path/'social.sqlite';db=Cooperation(path)
    try:
        p=SocialPopulation(db,cores(),width=2,credit_policy='isolated-activation-v2')
        for i in range(20):
            assert episode(p,f'train-{i}',0,all_active=True)['verified']
        first=episode(p,'old',0,2,learn=False)
        assert first['verified'] and first['members']==['a','b']
        assert any(e['kind']=='INVITE' for e in first['events'])
        head_rows=db.db.execute('SELECT * FROM social_heads_v1 ORDER BY owner,peer,revision').fetchall()
        old_spec=_hash(p.spec)
        with pytest.raises(ContractError,match='admitted usage'):
            p.extend('s',[new_core()],expected_sha=old_spec)
        grown=p.extend('s',[new_core()],expected_sha=old_spec,allow_reference=True)
        assert db.db.execute('SELECT * FROM social_heads_v1 ORDER BY owner,peer,revision').fetchall()==head_rows
        assert history(db.db,'s')==[(p.spec,old_spec),(grown.spec,_hash(grown.spec))]
        assert episode(grown,'old',0,2,learn=False)=={**first,'historical':True}
        with pytest.raises(ContractError,match='population changed'):
            episode(p,'stale',0,2,learn=False)
        with pytest.raises(ContractError,match='stale revision'):
            p.extend('s',[new_core()],expected_sha=old_spec,allow_reference=True)
        initial=episode(grown,'new-neutral',1,learn=False)
        assert initial['scores']['c']==0 and initial['members']==[]
        for i in range(30):
            episode(grown,f'new-train-{i}',1,shots=3,seed=i)
        good=episode(grown,'new',1,5,learn=False)
        assert good['verified'] and good['members']==['c']
        assert good['population_sha256']==_hash(grown.spec)
        retained=episode(grown,'retained',0,7,learn=False)
        assert retained['verified'] and retained['members']==['a','b']
        assert any(e['kind']=='INVITE' for e in retained['events'])
        assert db.db.execute('SELECT count(*) FROM social_population_v1').fetchone()==(1,)
    finally:db.close()
    db=Cooperation(path)
    try:
        p=SocialPopulation(db,[*cores(),new_core()],width=2,credit_policy='isolated-activation-v2')
        assert episode(p,'fresh-old',0,9,learn=False)['verified']
        assert episode(p,'fresh-new',1,9,learn=False)['verified']
        assert episode(p,'old',0,2,learn=False)['historical']
        with db.db:
            db.db.execute("DELETE FROM social_population_revisions_v1 WHERE session='s'")
        with pytest.raises(ContractError,match='head integrity'):
            episode(p,'truncated',0,9,learn=False)
    finally:db.close()


def test_growth_rejects_replacement_changed_policy_and_unbound_session():
    db=Cooperation()
    try:
        p=SocialPopulation(db,cores(),width=2)
        sha=_hash(p.spec)
        with pytest.raises(ContractError,match='stale revision'):
            p.extend('s',[new_core()],expected_sha=sha,allow_reference=True)
        episode(p,'start',0,all_active=True)
        with pytest.raises(ContractError,match='cannot replace'):
            p.extend('s',[cores()[0]],expected_sha=sha,allow_reference=True)
        with pytest.raises(ContractError,match='stale revision'):
            p.extend('s',[new_core()],expected_sha='0'*64,allow_reference=True)
        assert len(history(db.db,'s'))==1
    finally:db.close()
