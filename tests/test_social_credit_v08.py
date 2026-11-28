"""Independent individual credit under clipped coalition utility."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.social_learning import SocialPopulation,AffineSocialCore


def pop(store,policy):
    return SocialPopulation(store,[
        AffineSocialCore('useful',[[0.]],[1.]),
        AffineSocialCore('harmful',[[0.]],[10.])],width=1,credit_policy=policy)


def oracle(v):
    return v==(1.,),max(-1.,1.-abs(v[0]-1.))


def run(p,session,episode,**kw):
    return p.run(session,episode,[1.],[1.],oracle,oracle_id='clipped-oracle',**kw)


def test_clipped_group_no_longer_suppresses_self_credit():
    s=Cooperation()
    try:
        old=run(pop(s,'leave-one-out-v1'),'old','one',all_active=True)
        new=run(pop(s,'isolated-activation-v2'),'new','one',all_active=True)
        assert old['marginal']['useful']==new['marginal']['useful']==0.
        assert new['isolated_activation']['credit']=={'useful':1.,'harmful':-1.}
        assert new['oracle_calls']==6 and old['oracle_calls']==3
        assert new['core_calls']==old['core_calls']==2
        assert not new['verified'] and new['released'] is None
        assert not any(u['reward']>0 for u in new['updates'] if u['peer'])
        assert {u['owner']:u['reward'] for u in new['updates'] if not u['peer']}=={'useful':1.,'harmful':-1.}
    finally:s.close()


def test_isolated_credit_selects_useful_core_after_restart(tmp_path):
    path=tmp_path/'credit.sqlite'
    s=Cooperation(path)
    try:
        for i in range(8):
            run(pop(s,'isolated-activation-v2'),'s',f'train-{i}',all_active=True)
    finally:s.close()
    s=Cooperation(path)
    try:
        p=pop(s,'isolated-activation-v2')
        result=run(p,'s','holdout',learn=False)
        assert result['verified'] and result['members']==['useful']
        assert result['released']==[1.] and result['core_calls']==1
        assert run(p,'s','holdout',learn=False)['historical']
        with pytest.raises(ContractError,match='session population changed'):
            run(pop(s,'leave-one-out-v1'),'s','different')
    finally:s.close()


def test_extra_oracle_failure_rolls_back():
    s=Cooperation()
    try:
        p=pop(s,'isolated-activation-v2');count=0
        def bad(v):
            nonlocal count
            count+=1
            if count==4:return True,2.
            return oracle(v)
        with pytest.raises(ContractError,match='oracle contract'):
            p.run('s','bad',[1.],[1.],bad,oracle_id='bad',all_active=True)
        for table in ('social_heads_v1','social_episodes_v1','social_population_v1'):
            assert s.db.execute('SELECT count(*) FROM '+table).fetchone()==(0,)
    finally:s.close()
