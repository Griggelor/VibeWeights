"""Same-task continuation across recruitment without host child selection."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.continuation import RecruitmentContinuation
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.request_heads import RequestHeads
from vibeweights.task_learning import TaskLearning
from test_knowledge_learning_v08 import fixture, close
from test_targeted_recruitment_v08 import setup, cases
from test_neural_request_heads_v08 import train


def setup_session(tmp_path,*,foreign=False):
    c,p,k,a,l,h,_,event=setup(tmp_path)
    heads=RequestHeads(l);train(heads,'consumer',True);train(heads,'middle',False)
    if foreign:
        c.stage_affine('foreign','foreign',[[100]],[0],source_topic='input',target_scope='global',target_topic='other')
        assert p.admit('foreign',[((0,),(0,)),((2,),(200,))])
        policy=lambda k,r,p,o:r
        h.stage('foreign-head','foreign',['unrelated'],cases(0,policy),steps=512,feature_version=2)
        assert h.admit('foreign-head',cases(1,policy),oracle_id='foreign-proof')['accepted']
    w=GroupWork(h,request_heads=heads)
    return c,p,k,a,l,h,w,event


def execute(w,event,session,*,x=4,max_runs=3,max_deliveries=64,oracle='independent-formula'):
    return RecruitmentContinuation(w).run(session,event,'explore','research',{'input':(x,)},'answer',
        {'pre':lambda s,v:v==(4*x-1,), 'mid':lambda s,v:v==(8*x-1,),
         'answer':lambda s,v:v==(24*x-1,)},oracle_id=oracle,max_runs=max_runs,max_deliveries=max_deliveries)


def test_same_task_automatic_continuation_and_group_boundary(tmp_path):
    c,p,k,a,l,h,w,event=setup_session(tmp_path,foreign=True)
    out=execute(w,event,'session')
    assert out['status']=='OK' and out['value']==[95] and out['runs_reserved']==3
    assert 0<out['deliveries_reserved']<=64
    runs=[w.receipt(r) for (r,) in w.db.execute('SELECT run FROM continuation_runs_v08 WHERE session=?',('session',))]
    assert len(runs)==3
    assert all(r['inputs']=={'input':[4]} and r['group']=='research' for r in runs)
    assert all('foreign' not in {m['expert'] for m in r['members']} for r in runs)
    assert any(g['group']=='unrelated' and 'foreign' in g['experts'] for g in h.bus.groups(out['event']))
    assert {r['result']['status'] for r in runs}=={'STAGNATION','OK'}
    historical=execute(w,event,'session')
    assert historical['status']=='HISTORICAL' and historical['run']==out['run'] and 'value' not in historical
    assert historical['runs_reserved']==out['runs_reserved'] and historical['deliveries_reserved']==out['deliveries_reserved']
    close(c,p,k)


def test_persistent_session_limits_and_task_identity(tmp_path):
    c,p,k,a,l,h,w,event=setup_session(tmp_path)
    out=execute(w,event,'limited',max_runs=2)
    assert out['status']=='RUN_BUDGET' and out['runs_reserved']==2 and 'value' not in out
    with pytest.raises(ContractError,match='task identity changed'):
        execute(w,event,'limited',max_runs=3)
    with pytest.raises(ContractError,match='task identity changed'):
        execute(w,event,'limited',max_runs=2,x=5)
    with pytest.raises(ContractError,match='task identity changed'):
        execute(w,event,'limited',max_runs=2,oracle='different')
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path);h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    w=GroupWork(h,request_heads=RequestHeads(l))
    replay=execute(w,event,'limited',max_runs=2)
    assert replay['status']=='RUN_BUDGET' and replay['runs_reserved']==2
    assert replay['deliveries_reserved']==out['deliveries_reserved']
    close(c,p,k)


def test_restart_after_group_receipt_and_source_commit(tmp_path,monkeypatch):
    c,p,k,a,l,h,w,event=setup_session(tmp_path)
    real=w.execute
    def interrupt(*args,**kwargs):
        real(*args,**kwargs)
        raise RuntimeError('interrupted after group and dispatch')
    with monkeypatch.context() as m:
        m.setattr(w,'execute',interrupt)
        with pytest.raises(RuntimeError,match='interrupted after group and dispatch'):
            execute(w,event,'recovery')
    assert w.db.execute('SELECT COUNT(*) FROM group_runs_v08').fetchone()==(1,)
    assert h.bus.db.execute("SELECT COUNT(*) FROM learning_events_v08 WHERE kind='RECRUIT'").fetchone()==(1,)
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path);h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    w=GroupWork(h,request_heads=RequestHeads(l))
    out=execute(w,event,'recovery')
    assert out['status']=='OK' and out['value']==[95] and out['runs_reserved']==3
    assert w.db.execute('SELECT COUNT(*) FROM group_runs_v08').fetchone()==(3,)
    assert h.bus.db.execute("SELECT COUNT(*) FROM learning_events_v08 WHERE kind='RECRUIT'").fetchone()==(2,)
    close(c,p,k)


def test_delivery_budget_no_partial_group_execution(tmp_path):
    c,p,k,a,l,h,w,event=setup_session(tmp_path)
    out=execute(w,event,'delivery-limited',max_deliveries=1)
    assert out['status']=='DELIVERY_BUDGET' and out['runs_reserved']==1
    assert out['deliveries_reserved']==1 and 'value' not in out
    assert execute(w,event,'delivery-limited',max_deliveries=1)==out
    close(c,p,k)
