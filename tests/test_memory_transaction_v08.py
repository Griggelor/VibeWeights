import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest
from vibeweights.abi import ContractError
from vibeweights.context import Action, Op
from vibeweights.environment import Signal, EventLedger
from vibeweights.knowledge import Access, KnowledgeLedger
from vibeweights.knowledge_learning import Domain
from test_knowledge_learning_v08 import fixture, facts, close


def setup_chain(tmp_path):
    c,p,k,a,l=fixture(tmp_path)
    d=Domain('learned',3,2); rid=facts(k,a,d,{0:1})[0]
    l.prepare('job','first',d,[rid],method='DIRECT')
    assert l.promote('job',{0:1}.get,oracle_id='oracle')
    c.stage_affine('second','second',[[0,7]],[0],source_topic='learned:answer',target_scope='global',target_topic='result',parents=('first',))
    assert p.admit('second',[((0,1),(7,))])
    attach(p)
    return c,p,k,a,l,d,rid


def attach(p):
    # Test-only memory-action adapter; the resident affine core still computes.
    e=p._resident['second']
    e.memory_action=lambda snapshot,view: Action(Op.REMEMBER) if not view.remember and any(s.topic=='result' for s in snapshot) else None
    e.forward_with_memory=lambda snapshot,view:e.forward(snapshot)


def run(p,k,d,run_id='run1',validator=None,extra=()):
    return p.run([Signal('global','learned:query',d.encode(0)),*extra],('global','result'),
        {('global','learned:answer'):lambda s,v:v==(0.,1.),('global','result'):validator or (lambda s,v:v==(7.,))},
        knowledge=k,remember_experts=('second',),memory_run_id=run_id)


def test_guarded_commit_provenance_restart_and_source_revision(tmp_path):
    c,p,k,a,l,d,rid=setup_chain(tmp_path)
    result=run(p,k,d)
    assert result['status']=='OK' and result['value']==[7.]
    saved=result['remembered']['id']
    assert k.current_verified(saved,a).parents==(rid,)
    assert {b['expert'] for b in result['memory_core_provenance']}=={'first','second'}
    assert l.memory_runs.receipt('run1')==json.loads(json.dumps(result))
    with pytest.raises(ContractError,match='already recorded'): run(p,k,d)
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path); attach(p)
    assert l.memory_runs.receipt('run1')['remembered']['id']==saved
    k.store(a,scope='global',key='learned:entity:0',topic='learned',values=(0,),source='revision',status='VERIFIED',expected=rid)
    with pytest.raises(ContractError): k.current_verified(saved,a)
    assert l.audit_sources()==('first','second')
    assert l.memory_runs.receipt('run1')['status']=='OK' # historical only
    close(c,p,k)


def test_atomic_rollback_on_receipt_failure(tmp_path,monkeypatch):
    c,p,k,a,l,d,rid=setup_chain(tmp_path)
    before=k.db.execute('SELECT count(*) FROM dep_source_events_v08').fetchone()
    save=l.memory_runs.save
    def fail(*args):
        save(*args)
        raise RuntimeError('interrupted after receipt insert')
    with monkeypatch.context() as mp:
        mp.setattr(l.memory_runs,'save',fail)
        with pytest.raises(RuntimeError):run(p,k,d)
    assert not k.db.in_transaction and k.search(a,topic='result',status=None)==()
    assert k.db.execute('SELECT count(*) FROM dep_source_events_v08').fetchone()==before
    with pytest.raises(ContractError):l.memory_runs.receipt('run1')
    assert run(p,k,d)['status']=='OK'
    close(c,p,k)


def test_scope_and_invalid_goal_never_saved(tmp_path):
    c,p,k,a,l,d,rid=setup_chain(tmp_path)
    with pytest.raises(ContractError,match='scope cannot be widened'):
        run(p,k,d,extra=(Signal('private:second','private-input',(9,)),))
    assert k.search(a,topic='result',status=None)==()
    with pytest.raises(ContractError): l.memory_runs.receipt('run1')
    r=run(p,k,d,validator=lambda s,v:False)
    assert r['status']!='OK' and 'remembered' not in r
    assert l.memory_runs.receipt('run1')['status']!='OK'
    close(c,p,k)


def test_nested_writes_and_competing_connection(tmp_path):
    c,p,k,a,l,d,rid=setup_chain(tmp_path)
    other=KnowledgeLedger(tmp_path/'knowledge.sqlite');other.db.execute('PRAGMA busy_timeout=10')
    seen=[]
    def validator(s,v):
        with pytest.raises(ContractError,match='nested public'):
            k.store(a,scope='global',key='bad',topic='bad',values=(0,),source='test')
        with pytest.raises(ContractError,match='nested public'):k.link(a,rid,'self',rid)
        with pytest.raises(sqlite3.OperationalError,match='locked'):
            other.store(a,scope='global',key='bad',topic='bad',values=(0,),source='test')
        assert k.db.in_transaction
        seen.append(True)
        return v==(7.,)
    assert run(p,k,d,validator=validator)['status']=='OK' and seen
    assert k.search(a,topic='bad',status=None)==()
    other.close();close(c,p,k)


def test_external_ledger_and_nested_guard_rejected(tmp_path):
    c,p,k,a,l,d,rid=setup_chain(tmp_path)
    ev=EventLedger(tmp_path/'events.sqlite')
    with pytest.raises(ContractError,match='atomic source receipt'):
        p.run([],('global','result'),{},knowledge=k,remember_experts=('second',),ledger=ev)
    with l._guard():
        with pytest.raises(ContractError,match='outer source transaction'):run(p,k,d)
        assert k.db.in_transaction
    ev.close();close(c,p,k)


@pytest.mark.parametrize('boundary',['before-commit','after-commit'])
def test_process_exit_at_commit_boundary(tmp_path,boundary):
    c,p,k,a,l,d,rid=setup_chain(tmp_path);close(c,p,k)
    code='''import os,sys
from pathlib import Path
from test_memory_transaction_v08 import fixture,attach,run,Domain
c,p,k,a,l=fixture(Path(sys.argv[1]));attach(p)
if sys.argv[2]=='before-commit':
    original=l.memory_runs.save
    def stop(*args):
        original(*args);os._exit(73)
    l.memory_runs.save=stop
run(p,k,Domain('learned',3,2))
os._exit(73)
'''
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'tests')]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cp=subprocess.run([sys.executable,'-c',code,str(tmp_path),boundary],env=env,timeout=35)
    assert cp.returncode==73
    c,p,k,a,l=fixture(tmp_path);attach(p)
    if boundary=='before-commit':
        assert k.search(a,topic='result')==()
        with pytest.raises(ContractError):l.memory_runs.receipt('run1')
        assert run(p,k,d)['status']=='OK'
    else:
        receipt=l.memory_runs.receipt('run1'); assert receipt['status']=='OK'
        assert k.current_verified(receipt['remembered']['id'],a).values==(7.,)
        with pytest.raises(ContractError):run(p,k,d)
    close(c,p,k)
