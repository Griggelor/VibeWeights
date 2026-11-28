from itertools import product
import pytest
from vibeweights.abi import ContractError
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from vibeweights.knowledge_learning import Domain
from vibeweights.population import ARCHIVED, SelectionEvidence
from test_knowledge_learning_v08 import fixture, close, facts
from test_learning_events_v08 import residents


def cases(affected, reverse=False):
    out=[]
    for k,r,s,l,p in product((0,1),repeat=5):
        x=(k,r,s,affected,l,p)
        out.append((x,(int(k and (not s if reverse else s)),int(r and s))))
    return out


def install(h, name, expert, reverse=False):
    r=h.stage(name,expert,['research','verify'],cases(0,reverse),steps=512)
    assert r['loss_after'] < r['loss_before']
    assert h.admit(name,cases(1,reverse),oracle_id='heldout')['accepted']
    return r


def test_local_weights_different_decisions_restart(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path); bus=LearningEvents(t); h=PerceptionHeads(bus)
    before={m['id']:m['sha256'] for m in c.list()}
    r=install(h,'head-base','base'); r2=install(h,'head-new','new',True)
    assert r['sha256']!=r2['sha256']
    event=bus.publish_knowledge('base-job')
    assert h.propose('base',bus.event(event))==[('explore','research')]
    assert h.propose('new',bus.event(event))==[]
    close(c,p,k); c,p,k,a,l=fixture(tmp_path); bus=LearningEvents(TaskLearning(l));h=PerceptionHeads(bus)
    assert len(h.tick())==2 and bus.groups(event)==({'target':'explore','group':'research','experts':('base',)},)
    assert not h.tick() and before=={m['id']:m['sha256'] for m in c.list()}
    close(c,p,k)


def test_cross_topic_different_weights_same_event(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path); h=PerceptionHeads(LearningEvents(t));bus=h.bus
    install(h,'head-base','base');install(h,'head-new','new',True)
    other=Domain('other',4,3);l.prepare('other-job','other',other,facts(k,a,other,{1:2}),method='DIRECT')
    assert l.promote('other-job',{1:2}.get,oracle_id='other')
    event=bus.publish_knowledge('other-job')
    assert h.propose('base',bus.event(event))==[] and h.propose('new',bus.event(event))==[('explore','research')]
    assert len(h.tick())==2 and bus.groups(event)[0]['experts']==('new',)
    close(c,p,k)


def test_staged_rejected_and_overlap(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path);h=PerceptionHeads(LearningEvents(t));e=h.bus.publish_knowledge('new-job')
    h.stage('bad','base',['research','verify'],cases(0),steps=512)
    with pytest.raises(ContractError):h.propose('base',h.bus.event(e))
    with pytest.raises(ContractError):h.admit('bad',list(reversed(cases(0))),oracle_id='heldout')
    assert not h.admit('bad',cases(1,True),oracle_id='heldout')['accepted']
    with pytest.raises(ContractError):h.propose('base',h.bus.event(e))
    with pytest.raises(ContractError):h.admit('bad',cases(1),oracle_id='heldout')
    install(h,'good','base');assert h.propose('base',h.bus.event(e))
    close(c,p,k)


def test_corruption_and_archive(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path);h=PerceptionHeads(LearningEvents(t));install(h,'base-head','base')
    e=h.bus.publish_knowledge('new-job');
    with h.db:h.db.execute("UPDATE perception_heads_v08 SET weights=? WHERE id='base-head'",(b'corrupt',))
    with pytest.raises(ContractError):h.propose('base',h.bus.event(e))
    assert h.tick()[0]['status']=='DEFERRED' and not h.bus.groups(e)
    p.select('base',ARCHIVED,expected_revision=p.revision('base'),evidence=SelectionEvidence('report',1,True,False,True,True,True,True))
    assert not h.tick();assert c.manifest('base')['state']==ARCHIVED
    close(c,p,k)


def test_relearn_target_and_stale_event(tmp_path,monkeypatch):
    from test_global_relearning_v08 import scenario,repair
    c,p,k,a,l,d,t,g,child,grand,rid=scenario(tmp_path,monkeypatch)
    h=PerceptionHeads(LearningEvents(t));expert=p.resident_ids()[0]
    # Learn a different policy: verify any RELEARN repair target, independently of topic.
    def data(a):return [(x,(int(x[1] and x[5]),)) for x in (tuple((k,r,s,a,lo,rep)) for k,r,s,lo,rep in product((0,1),repeat=5))]
    h.stage('repair-head',expert,['verify'],data(0),steps=512)
    assert h.admit('repair-head',data(1),oracle_id='holdout')['accepted']
    e,=h.bus.sync_conflicts();event=h.bus.event(e)
    assert set(h.propose(expert,event))=={('repair/child','verify'),('repair/grand','verify')}
    repair(g); assert g.promote('child-repair',{1:0}.get,oracle_id='independent')
    with pytest.raises(ContractError):h.propose(expert,event)
    close(c,p,k)
