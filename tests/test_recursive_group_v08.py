"""A newly learned core becomes a participant and generates another learned core."""
import pytest
from vibeweights.abi import ContractError
from vibeweights.group_learning import GroupLearning
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_group_learning_v08 import prepare, holdout
from test_group_work_v08 import training
from test_knowledge_learning_v08 import fixture, close


def first_generation(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    g.stage('first-gen','student',['first','second'],family='combined',method='TRAINED')
    return c,p,k,a,l,h,w,g


def add_head(h,expert,group):
    result=h.stage(expert+'-head',expert,[group],training(0),steps=512)
    assert result['loss_after']<result['loss_before']
    assert h.admit(expert+'-head',training(1),oracle_id='head-holdout')['accepted']
    return result


def ready(tmp_path):
    c,p,k,a,l,h,w,g=first_generation(tmp_path)
    assert g.promote('first-gen',holdout(),oracle_id='first-proof')
    head=add_head(h,'student','next')
    c.stage_affine('tail','tail',[[4]],[-1],source_topic='answer',target_scope='global',target_topic='final')
    assert p.admit('tail',[((0,),(-1,)),((2,),(7,))])
    add_head(h,'tail','next')
    event=h.bus.publish_group_knowledge('first-gen')
    assert h.bus.publish_group_knowledge('first-gen')==event
    return c,p,k,a,l,h,w,g,event,head


def next_run(w,event,run,x):
    return w.execute(run,event,'explore','next',{'input':(x,)},'final',
        {'answer':lambda s,v:v==(6*x+5,), 'final':lambda s,v:v==(24*x+19,)},oracle_id='second-teacher')


def test_two_generation_training(tmp_path):
    c,p,k,a,l,h,w,g,event,head=ready(tmp_path)
    # Old and new cores may see the same event; no central selection of group.
    delivered=h.tick(delivery_budget=32)
    assert all(r['status']=='RECORDED' for r in delivered)
    assert {'target':'explore','group':'next','experts':('student','tail')} in h.bus.groups(event)
    before={m['id']:m['sha256'] for m in c.list()}
    for name,x in [('next-first',2),('next-second',5)]:
        out=next_run(w,event,name,x)['result']
        assert out['status']=='OK' and out['value']==[24*x+19]
        assert {t[1] for t in out['trace']}=={'student','tail'}
    m=g.stage('second-gen','grandstudent',['next-first','next-second'],family='combined-next',method='TRAINED')
    assert m['parents']==['student','tail']
    assert g.promote('second-gen',[((-3,),(-53,)),((0,),(19,)),((11,),(283,))],oracle_id='second-proof')
    add_head(h,'grandstudent','third')
    new_event=h.bus.publish_group_knowledge('second-gen')
    assert new_event!=event
    assert all(c.manifest(i)['sha256']==sha for i,sha in before.items())
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path);h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    assert h._load('student-head')[2]==head['sha256']
    h.tick(delivery_budget=64)
    assert {'target':'explore','group':'third','experts':('grandstudent',)} in h.bus.groups(new_event)
    assert all(p.compute('grandstudent',(x,))==(24*x+19,) for x in range(-20,21))
    close(c,p,k)


def test_staged_or_rejected_no_announcement(tmp_path):
    c,p,k,a,l,h,w,g=first_generation(tmp_path)
    with pytest.raises(ContractError):h.bus.publish_group_knowledge('first-gen')
    with pytest.raises(ContractError):add_head(h,'student','next')
    assert not g.promote('first-gen',[((-3,),(0,)),*holdout()[1:]],oracle_id='bad')
    with pytest.raises(ContractError):h.bus.publish_group_knowledge('first-gen')
    close(c,p,k)


def test_revoked_ancestor_withholds_announcement(tmp_path):
    c,p,k,a,l,h,w,g,event,_=ready(tmp_path);h.tick(delivery_budget=32)
    assert h.bus.groups(event)
    p.quarantine(['z-first'],reason='test-source-hold')
    assert not h.bus.groups(event)
    assert c.manifest('student')['state']=='QUARANTINED'
    with pytest.raises(ContractError):h.bus.publish_group_knowledge('first-gen')
    with pytest.raises(ContractError):next_run(w,event,'stale',2)
    close(c,p,k)


def test_report_mismatch_withholds_event(tmp_path):
    c,p,k,a,l,h,w,g,event,_=ready(tmp_path)
    with l.db:l.db.execute("UPDATE group_learning_reports_v08 SET artifact_sha='wrong' WHERE job_id='first-gen'")
    assert not h.bus.groups(event)
    assert all(e['id']!=event for e in h.bus.poll('tail'))
    with pytest.raises(ContractError):h.bus.publish_group_knowledge('first-gen')
    close(c,p,k)
