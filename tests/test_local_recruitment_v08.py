"""Opt-in local initiation; no host-specified requester or route."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_knowledge_learning_v08 import fixture, close
from test_targeted_recruitment_v08 import setup, members


def execute(w,event,run_id,*,budget=2,x=4):
    return w.execute(run_id,event,'explore','research',{'input':(x,)},'answer',
        {'pre':lambda s,v:v==(4*x-1,), 'mid':lambda s,v:v==(8*x-1,),
         'answer':lambda s,v:v==(24*x-1,)},oracle_id='independent-formula',
        recruitment_budget=budget)


def events(h,run):
    return tuple(e for e in h.bus.poll('provider',limit=64)
                 if e['kind']=='RECRUIT' and e['payload']['run']==run)


def count(h):
    return h.bus.db.execute("SELECT COUNT(*) FROM learning_events_v08 WHERE kind='RECRUIT'").fetchone()[0]


def test_local_initiation_two_generations_and_restart(tmp_path):
    c,p,k,a,l,h,w,event=setup(tmp_path)
    before={m['id']:m['sha256'] for m in c.list()}
    first=execute(w,event,'local-first')
    assert first['result']['status']=='STAGNATION'
    assert first['recruitment']=={'contract':'local-recruitment-v1','budget':2,
                                  'requesters':['consumer'],'state':'READY'}
    e1,=events(h,'local-first') # already published; no host requester
    assert e1['payload']['requested_topic']=='mid'
    assert w.dispatch_recruitment('local-first')==(e1['id'],)
    h.tick(delivery_budget=64)
    assert members(h,e1['id'])==('consumer','middle')
    second=execute(w,e1['id'],'local-second')
    assert second['result']['status']=='STAGNATION'
    assert second['recruitment']['requesters']==['consumer','middle']
    requests=events(h,'local-second')
    assert {e['payload']['requested_topic'] for e in requests}=={'mid','pre'}
    e2=next(e for e in requests if e['payload']['requested_topic']=='pre')
    assert e2['payload']['depth']==2 and 'inputs' not in e2['payload']
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    h=PerceptionHeads(LearningEvents(TaskLearning(l)));w=GroupWork(h)
    assert set(w.dispatch_recruitment('local-second'))=={e['id'] for e in requests}
    h.tick(delivery_budget=64)
    assert members(h,e2['id'])==('consumer','middle','provider')
    third=execute(w,e2['id'],'local-third')
    assert third['result']['status']=='OK' and third['result']['value']==[95]
    assert third['recruitment']['state']=='NOT_APPLICABLE'
    assert w.dispatch_recruitment('local-third')==()
    assert before=={m['id']:m['sha256'] for m in c.list()}
    close(c,p,k)


def test_all_or_none_budget_and_opt_in(tmp_path):
    c,p,k,a,l,h,w,event=setup(tmp_path)
    old=execute(w,event,'disabled',budget=0)
    assert 'recruitment' not in old and count(h)==0
    assert w.dispatch_recruitment('disabled')==()
    first=execute(w,event,'first',budget=1)
    assert first['recruitment']['state']=='READY'
    e1,=events(h,'first')
    h.tick(delivery_budget=64)
    before=count(h)
    second=execute(w,e1['id'],'limit',budget=1)
    assert second['recruitment']['state']=='BUDGET'
    assert second['recruitment']['requesters']==['consumer','middle']
    assert count(h)==before and w.dispatch_recruitment('limit')==()
    with pytest.raises(ContractError,match='local recruitment budget'):
        execute(w,event,'invalid-budget',budget=True)
    no_request=w.execute('available',event,'explore','research',{'mid':(1,)},'answer',
                         {'answer':lambda s,v:False},oracle_id='reject',
                         recruitment_budget=1)
    assert no_request['result']['status']=='STAGNATION'
    assert no_request['recruitment']['state']=='NO_REQUEST'
    assert count(h)==before
    close(c,p,k)


def test_source_batch_rollback_and_recovery(tmp_path,monkeypatch):
    c,p,k,a,l,h,w,event=setup(tmp_path)
    execute(w,event,'first')
    e1,=events(h,'first')
    h.tick(delivery_budget=64)
    before=count(h)
    insert=h.bus._insert
    calls=[]
    def interrupt(kind,payload):
        calls.append(payload['requester'])
        if len(calls)==2:
            raise RuntimeError('interrupted between inserts')
        return insert(kind,payload)
    with monkeypatch.context() as m:
        m.setattr(h.bus,'_insert',interrupt)
        with pytest.raises(RuntimeError,match='interrupted between inserts'):
            execute(w,e1['id'],'interrupted')
    assert calls==['consumer','middle']
    assert count(h)==before
    assert w.receipt('interrupted')['recruitment']['state']=='READY'
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    h=PerceptionHeads(LearningEvents(TaskLearning(l)));w=GroupWork(h)
    recovered=w.dispatch_recruitment('interrupted')
    assert len(set(recovered))==2 and count(h)==before+2
    assert w.dispatch_recruitment('interrupted')==recovered
    assert count(h)==before+2
    with pytest.raises(ContractError):execute(w,e1['id'],'interrupted')
    close(c,p,k)
