"""Targeted, voluntary recruitment from a genuine group stagnation."""
from itertools import product
import pytest
from vibeweights.abi import ContractError
from vibeweights.group_work import GroupWork
from vibeweights.knowledge_learning import Domain
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_knowledge_learning_v08 import fixture, close, facts
from test_group_work_v08 import training


def cases(affected, policy):
    return [((k,r,s,affected,lo,rep,rec,p,own),(int(policy(k,rec,p,own)),))
            for k,r,s,lo,rep,rec,p,own in product((0,1),repeat=8)]


def setup(tmp_path):
    c,p,k,a,l=fixture(tmp_path)
    d=Domain('notice',4,3)
    l.prepare('notice-job','notice',d,facts(k,a,d,{0:1}),method='DIRECT')
    assert l.promote('notice-job',{0:1}.get,oracle_id='notice-proof')
    h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    for eid,w,b,src,dst in [
        ('consumer',3,2,'mid','answer'),('middle',2,1,'pre','mid'),
        ('provider',4,-1,'input','pre'),('irrelevant',6,5,'input','other'),
        ('legacy',1,0,'input','other')]:
        c.stage_affine(eid,eid,[[w]],[b],source_topic=src,target_scope='global',target_topic=dst)
        assert p.admit(eid,[((0,),(b,)),((2,),(2*w+b,))])
        if eid=='legacy':
            h.stage(eid+'-head',eid,['legacy-group'],training(0),steps=512)
            assert h.admit(eid+'-head',training(1),oracle_id='heldout')['accepted']
            continue
        policy=(lambda k,r,p,o:k or r) if eid=='consumer' else (lambda k,r,p,o:r and (p or o))
        h.stage(eid+'-head',eid,['research'],cases(0,policy),steps=512,feature_version=2)
        assert h.admit(eid+'-head',cases(1,policy),oracle_id='heldout')['accepted']
    event=h.bus.publish_knowledge('notice-job')
    h.tick(delivery_budget=64)
    return c,p,k,a,l,h,GroupWork(h),event


def run(w,event,run_id,x=4):
    return w.execute(run_id,event,'explore','research',{'input':(x,)},'answer',
        {'pre':lambda s,v:v==(4*x-1,), 'mid':lambda s,v:v==(8*x-1,),
         'answer':lambda s,v:v==(24*x-1,)},oracle_id='independent-formula')


def members(h,event):
    return next(g['experts'] for g in h.bus.groups(event) if g['group']=='research')


def test_recursive_recruitment_restart_and_abstention(tmp_path):
    c,p,k,a,l,h,w,event=setup(tmp_path)
    hashes={m['id']:m['sha256'] for m in c.list()}
    assert members(h,event)==('consumer',)
    first=run(w,event,'first')
    assert first['result']['status']=='STAGNATION'
    with pytest.raises(ContractError):h.bus.publish_recruitment('first','provider')
    e1=h.bus.publish_recruitment('first','consumer')
    assert h.bus.publish_recruitment('first','consumer')==e1
    e=h.bus.event(e1)
    assert e['kind']=='RECRUIT' and e['payload']['requested_topic']=='mid'
    assert e['payload']['depth']==1 and e['payload']['parent_event']==event
    assert 'inputs' not in e['payload'] and 'facts' not in e['payload']
    assert h.propose('legacy',e)==[]
    h.tick(delivery_budget=64)
    assert members(h,e1)==('consumer','middle')
    second=run(w,e1,'second')
    assert second['result']['status']=='STAGNATION'
    e2=h.bus.publish_recruitment('second','middle')
    assert h.bus.event(e2)['payload']['depth']==2
    assert h.bus.event(e2)['payload']['requested_topic']=='pre'
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path);h=PerceptionHeads(LearningEvents(TaskLearning(l)));w=GroupWork(h)
    h.tick(delivery_budget=64)
    assert members(h,e2)==('consumer','middle','provider')
    out=run(w,e2,'third')['result']
    assert out['status']=='OK' and out['value']==[95]
    assert {t[1] for t in out['trace']}=={'consumer','middle','provider'}
    with pytest.raises(ContractError):h.bus.publish_recruitment('third','consumer')
    assert hashes=={m['id']:m['sha256'] for m in c.list()}
    p.quarantine(['consumer'],reason='test-source-hold')
    assert h.bus.groups(e1)==() and h.bus.groups(e2)==()
    with pytest.raises(ContractError):run(w,e2,'stale')
    close(c,p,k)


def test_depth_budget_and_receipt_integrity(tmp_path):
    c,p,k,a,l,h,w,event=setup(tmp_path)
    # No provider for the consumer's input: deliberately omit the preceding
    # input and publish repeated missing-mid requests across child events.
    for depth in range(1,5):
        h.tick(delivery_budget=64)
        run_id=f'depth-{depth}'
        out=w.execute(run_id,event,'explore','research',{'unrelated':(1,)},'answer',
                      {'answer':lambda s,v:False},oracle_id='no-answer')
        assert out['result']['status']=='STAGNATION'
        event=h.bus.publish_recruitment(run_id,'consumer')
        assert h.bus.event(event)['payload']['depth']==depth
    h.tick(delivery_budget=64)
    out=w.execute('too-deep',event,'explore','research',{'unrelated':(1,)},'answer',
                  {'answer':lambda s,v:False},oracle_id='no-answer')
    assert out['result']['status']=='STAGNATION'
    with pytest.raises(ContractError,match='depth budget'):h.bus.publish_recruitment('too-deep','consumer')
    with w.db:w.db.execute("UPDATE group_runs_v08 SET receipt='{}' WHERE id='depth-1'")
    assert h.bus.groups(event)==()
    close(c,p,k)
