"""Learned local request decisions: abstention, admission, recovery and liveness."""
from itertools import product
import pytest
from vibeweights.abi import ContractError
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.request_heads import RequestHeads, decide
from vibeweights.task_learning import TaskLearning
from test_knowledge_learning_v08 import fixture, close
from test_targeted_recruitment_v08 import setup, members
from test_local_recruitment_v08 import events, count


def samples(split,initial):
    return [((a,b,c,split),(int(a and b and (not c if initial else c)),))
            for a,b,c in product((0,1),repeat=3)]


def train(heads,expert,initial):
    staged=heads.stage(expert+'-request',expert,samples(0,initial))
    assert staged['loss_after']<staged['loss_before']
    report=heads.admit(expert+'-request',samples(1,initial),oracle_id='request-independent')
    assert report['accepted'] and report['passed']==8
    return staged


def execute(w,event,run_id,budget=2):
    return w.execute(run_id,event,'explore','research',{'input':(4,)},'answer',
        {'pre':lambda s,v:v==(15,), 'mid':lambda s,v:v==(31,),
         'answer':lambda s,v:v==(95,)},oracle_id='independent-formula',
        recruitment_budget=budget,recruitment_policy='learned')


def test_different_local_decisions_recursive_restart(tmp_path):
    c,p,k,a,l,h,_,event=setup(tmp_path)
    heads=RequestHeads(l);train(heads,'consumer',True);train(heads,'middle',False)
    w=GroupWork(h,request_heads=heads)
    hashes={m['id']:m['sha256'] for m in c.list()}
    first=execute(w,event,'neural-first')
    assert first['result']['status']=='STAGNATION'
    assert first['recruitment']['requesters']==['consumer']
    e1,=events(h,'neural-first')
    h.tick(delivery_budget=64)
    assert members(h,e1['id'])==('consumer','middle')
    second=execute(w,e1['id'],'neural-second')
    assert second['result']['status']=='STAGNATION'
    decisions={d['expert']:d for d in second['recruitment']['decisions']}
    assert decisions['consumer']['features'][:3]==decisions['middle']['features'][:3]==[1,1,1]
    assert decisions['consumer']['request'] is False
    assert decisions['middle']['request'] is True
    assert second['recruitment']['requesters']==['middle']
    with pytest.raises(ContractError,match='learned recruitment not authorized'):
        h.bus.publish_recruitment('neural-second','consumer')
    e2,=events(h,'neural-second')
    assert e2['payload']['requested_topic']=='pre'
    assert 'decisions' not in e2['payload'] and 'facts' not in e2['payload']
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    h=PerceptionHeads(LearningEvents(TaskLearning(l)));heads=RequestHeads(l);w=GroupWork(h,request_heads=heads)
    assert w.dispatch_recruitment('neural-second')==(e2['id'],)
    h.tick(delivery_budget=64)
    assert members(h,e2['id'])==('consumer','middle','provider')
    out=execute(w,e2['id'],'neural-third')
    assert out['result']['status']=='OK' and out['result']['value']==[95]
    assert out['recruitment']['state']=='NOT_APPLICABLE'
    assert hashes=={m['id']:m['sha256'] for m in c.list()}
    # A request head is part of the causal evidence, not just a dispatch hint.
    with l.db:l.db.execute("UPDATE request_heads_v08 SET weights=x'00' WHERE id='middle-request'")
    assert h.bus.groups(e2['id'])==()
    with pytest.raises(ContractError):w.dispatch_recruitment('neural-second')
    close(c,p,k)


def test_no_head_no_symbolic_fallback(tmp_path):
    c,p,k,a,l,h,_,event=setup(tmp_path)
    heads=RequestHeads(l);w=GroupWork(h,request_heads=heads)
    out=execute(w,event,'unbound')
    assert out['result']['status']=='STAGNATION'
    assert out['recruitment']['state']=='NO_REQUEST'
    assert out['recruitment']['decisions']==[] and count(h)==0
    with pytest.raises(ContractError,match='learned recruitment not authorized'):
        h.bus.publish_recruitment('unbound','consumer')
    close(c,p,k)


def test_request_admission_independent_and_immutable(tmp_path):
    c,p,k,a,l,h,_,event=setup(tmp_path)
    heads=RequestHeads(l)
    heads.stage('wrong','consumer',samples(0,True))
    with pytest.raises(ContractError,match='disjoint'):
        heads.admit('wrong',samples(0,True),oracle_id='overlap')
    bad=heads.admit('wrong',samples(1,False),oracle_id='wrong')
    assert not bad['accepted'] and decide(l,'consumer',[1,1,0,1]) is None
    with pytest.raises(ContractError,match='report changed'):
        heads.admit('wrong',samples(1,True),oracle_id='changed')
    staged=train(heads,'consumer',True)
    assert decide(l,'consumer',[1,1,0,1])['sha256']==staged['sha256']
    assert decide(l,'consumer',[1,1,1,1])['request'] is False
    with l.db:l.db.execute("UPDATE request_reports_v08 SET receipt='{}' WHERE id='consumer-request'")
    with pytest.raises(ContractError,match='admission integrity'):decide(l,'consumer',[1,1,0,1])
    close(c,p,k)


def test_learned_dispatch_recovery(tmp_path,monkeypatch):
    c,p,k,a,l,h,_,event=setup(tmp_path)
    heads=RequestHeads(l);train(heads,'consumer',True)
    w=GroupWork(h,request_heads=heads)
    def interrupted(*args):raise RuntimeError('interrupted before dispatch')
    with monkeypatch.context() as m:
        m.setattr(h.bus,'publish_recruitments',interrupted)
        with pytest.raises(RuntimeError,match='interrupted before dispatch'):
            execute(w,event,'interrupted-neural')
    assert w.receipt('interrupted-neural')['recruitment']['state']=='READY' and count(h)==0
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    h=PerceptionHeads(LearningEvents(TaskLearning(l)));heads=RequestHeads(l);w=GroupWork(h,request_heads=heads)
    assert len(w.dispatch_recruitment('interrupted-neural'))==1
    assert w.dispatch_recruitment('interrupted-neural')==w.dispatch_recruitment('interrupted-neural')
    assert count(h)==1
    close(c,p,k)
