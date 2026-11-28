"""Independent local participation, peer history, shots, recovery and budgets."""
import sqlite3
import pytest
from vibeweights.abi import ContractError
from vibeweights.coalition import Cooperation
from vibeweights.social_experiment import population,episode,train,evaluate


def test_social_emergence_and_restart(tmp_path):
    path=tmp_path/'social.sqlite'
    store=Cooperation(path)
    pop=population(store)
    train(pop,'learned',7)
    store.close()
    store=Cooperation(path)
    try:
        pop=population(store)
        got=evaluate(pop,'learned')
        assert got['passed']==24 and got['core_calls']==48 and got['invitations']==24
        assert all(len(r['self_activated'])==1 and len(r['members'])==2 for r in got['rows'])
        assert evaluate(pop,'learned',prefix='ablation',recall=False)['passed']==0
        # The same episode ID is historical; no retraining or duplicate rows.
        r=episode(pop,'learned','held-0-10',0,10,learn=False,seed=__import__('random').Random(0).randrange(2**63))
        assert r['historical'] and r['verified'] and not r['updates']
        with pytest.raises(ContractError,match='identity/integrity'):
            episode(pop,'learned','held-0-10',0,11,learn=False)
    finally:store.close()


def test_neutral_heads_and_shot_evidence():
    store=Cooperation()
    try:
        pop=population(store)
        a=episode(pop,'neutral','first',0,2,shots=0)
        assert not a['members'] and not a['verified'] and not a['updates']
        b=episode(pop,'neutral','second',0,2,shots=2,seed=1)
        assert not b['self_activated'] and len(b['shot_ids'])==2
        assert b['core_calls']==2 and b['oracle_calls']==3
        assert all(u['owner'] in b['members'] for u in b['updates'])
        assert all(u['reward']==b['marginal'][u['owner']] for u in b['updates'] if not u['peer'])
        assert all(u['owner']!=u['peer'] for u in b['updates'])
        assert len({x['sha256'] for x in b['updates']})==len(b['updates'])
        c=episode(pop,'neutral','second',0,2,shots=2,seed=1)
        assert c['historical'] and c['updates']==b['updates']
    finally:store.close()


def test_budget_oracle_and_integrity_rollback():
    store=Cooperation()
    try:
        pop=population(store)
        with pytest.raises(ContractError,match='member budget'):
            episode(pop,'budget','one',0,1,all_active=True,member_budget=2)
        assert store.db.execute('SELECT count(*) FROM social_episodes_v1').fetchone()==(0,)
        with pytest.raises(ContractError,match='oracle contract'):
            pop.run('bad','one',[1.,0.],[1],lambda value:(True,2.),oracle_id='bad',shots=1)
        assert store.db.execute('SELECT count(*) FROM social_heads_v1').fetchone()==(0,)
        episode(pop,'okay','one',0,1,shots=2,seed=1)
        with store.db:
            store.db.execute("UPDATE social_heads_v1 SET weights='[0,0]' WHERE session='okay'")
        with pytest.raises(ContractError,match='head integrity'):
            episode(pop,'okay','two',0,1)
        assert store.db.execute("SELECT count(*) FROM social_episodes_v1 WHERE session='okay'").fetchone()==(1,)
    finally:store.close()


def test_frozen_and_no_shot_controls():
    store=Cooperation()
    try:
        pop=population(store)
        train(pop,'frozen',19,learn=False,count=20)
        train(pop,'no-shots',19,shots=False,count=20)
        assert evaluate(pop,'frozen')['passed']==0
        assert evaluate(pop,'no-shots')['passed']==0
        assert evaluate(pop,'frozen',prefix='all-active',all_active=True)['passed']==0
        assert store.db.execute('SELECT count(*) FROM social_heads_v1').fetchone()==(0,)
    finally:store.close()
