import pytest
from vibeweights.abi import ContractError
from vibeweights.learning_events import LearningEvents
from vibeweights.task_learning import TaskLearning
from vibeweights.knowledge_learning import Domain
from vibeweights.population import ARCHIVED, SelectionEvidence
from test_knowledge_learning_v08 import fixture, close, facts
from test_extension_learning_v08 import setup, stage
from test_learning_initiative_v08 import Model
from test_global_relearning_v08 import scenario, repair


def residents(tmp_path):
    c,p,k,a,l,d,t,e=setup(tmp_path)
    stage(e,Model('{"entity":2,"label":2}')); assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    return c,p,k,a,l,d,t


def test_global_fanout_and_overlapping_groups_restart(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path); bus=LearningEvents(t)
    event=bus.publish_knowledge('new-job'); assert bus.publish_knowledge('new-job')==event
    assert bus.poll('base')==bus.poll('new')==(bus.event(event),)
    bus.respond(event,'base',[('explore','research'),('explore','verify')])
    bus.respond(event,'new',[('explore','research')])
    assert bus.groups(event)==({'target':'explore','group':'research','experts':('base','new')},
                                {'target':'explore','group':'verify','experts':('base',)})
    close(c,p,k); c,p,k,a,l=fixture(tmp_path); bus=LearningEvents(TaskLearning(l))
    assert not bus.poll('base') and not bus.poll('new') and len(bus.groups(event))==2
    assert bus.respond(event,'base',[('explore','verify'),('explore','research')])['expert']=='base'
    with pytest.raises(ContractError): bus.respond(event,'base',[])
    close(c,p,k)


def test_budget_and_invalid_targets_no_partial_receipt(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path);bus=LearningEvents(t,membership_budget=1,memberships_per_expert=1)
    event=bus.publish_knowledge('new-job')
    with pytest.raises(ContractError): bus.respond(event,'base',[('unknown','research')])
    assert len(bus.poll('base'))==1
    bus.respond(event,'base',[('explore','research')])
    with pytest.raises(ContractError): bus.respond(event,'new',[('explore','research')])
    assert len(bus.poll('new'))==1 and len(bus.groups(event))==1
    bus.respond(event,'new',[]);assert not bus.poll('new')
    assert len(c.list())==2 and not l.job('new-job')['state']=='STAGED'
    close(c,p,k)


def test_no_staged_notification_and_revoked_event(tmp_path):
    c,p,k,a,l,d,t,e=setup(tmp_path);bus=LearningEvents(t)
    stage(e,Model('{"entity":2,"label":2}'))
    with pytest.raises(ContractError): bus.publish_knowledge('new-job')
    assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    event=bus.publish_knowledge('new-job');bus.respond(event,'base',[('explore','research')])
    rid=l.job('new-job')['spec']['roots'][0];r=k.read(rid,a)
    k.store(a,scope='global',key=r.key,topic=r.topic,values=r.values,source='doubt',status='UNCERTAIN',expected=rid)
    assert not bus.groups(event) and not bus.poll('base')
    with pytest.raises(ContractError): bus.respond(event,'new',[])
    assert c.manifest('new')['state']=='QUARANTINED'
    close(c,p,k)


def test_archived_expert_not_loaded_or_dispatched(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path);bus=LearningEvents(t);event=bus.publish_knowledge('base-job')
    bus.respond(event,'base',[('explore','research')])
    ev=SelectionEvidence('report',1,True,False,True,True,True,True)
    p.select('base',ARCHIVED,expected_revision=p.revision('base'),evidence=ev)
    assert p.resident_ids()==('new',) and not bus.groups(event)
    with pytest.raises(ContractError): bus.poll('base')
    seen=[]; result=bus.tick({'base':lambda e:seen.append('base'),'new':lambda e:[]},delivery_budget=1)
    assert not seen and result[0]['expert']=='new' and c.manifest('base')['state']==ARCHIVED
    close(c,p,k)


def test_bounded_tick_fair_after_failure_and_restart(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path);bus=LearningEvents(t);event=bus.publish_knowledge('base-job')
    def fail(e): raise RuntimeError('transient')
    assert bus.tick({'base':fail,'new':lambda e:[]},delivery_budget=1)[0]['status']=='DEFERRED'
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);bus=LearningEvents(TaskLearning(l))
    r=bus.tick({'base':fail,'new':lambda e:[('explore','research')]},delivery_budget=1)
    assert r[0]['expert']=='new' and r[0]['status']=='RECORDED'
    assert bus.tick({'base':lambda e:[('explore','research')]},delivery_budget=1)[0]['status']=='RECORDED'
    assert len(bus.groups(event)[0]['experts'])==2 and not bus.tick({'base':fail,'new':fail})
    close(c,p,k)


def test_conflict_snapshot_superseded_then_reopened(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,g,child,grand,rid=scenario(tmp_path,monkeypatch);bus=LearningEvents(t)
    event,=bus.sync_conflicts();assert not bus.sync_conflicts()
    assert bus.event(event)['payload']['targets']==['repair/child','repair/grand']
    # The resident local repair expert sees the global work without a router.
    resident=p.resident_ids()[0];bus.respond(event,resident,[('repair/child','rebuild')])
    repair(g);assert g.promote('child-repair',{1:0}.get,oracle_id='independent')
    event2,=bus.sync_conflicts();assert event2!=event and not bus.groups(event)
    assert bus.event(event2)['payload']['targets']==['repair/grand']
    # Invalidate the admitted repair; same original targets must be notified anew.
    rr=l.job('child-repair')['spec']['roots'][0];r=k.read(rr,a)
    k.store(a,scope='global',key=r.key,topic=r.topic,values=r.values,source='fresh-doubt',status='UNCERTAIN',expected=rr)
    event3,=bus.sync_conflicts();assert event3 not in (event,event2)
    assert bus.event(event3)['payload']['targets']==['repair/child','repair/grand']
    assert not bus.groups(event2) and bus.poll(resident)==(bus.event(event3),)
    close(c,p,k)


def test_revalidate_after_callback(tmp_path):
    c,p,k,a,l,d,t=residents(tmp_path);bus=LearningEvents(t);event=bus.publish_knowledge('new-job')
    rid=l.job('new-job')['spec']['roots'][0];r=k.read(rid,a)
    def callback(e):
        assert not k.db.in_transaction
        k.store(a,scope='global',key=r.key,topic=r.topic,values=r.values,source='doubt',status='UNCERTAIN',expected=rid)
        return [('explore','research')]
    result=bus.tick({'base':callback},delivery_budget=1)
    assert result[0]['status']=='DEFERRED' and not bus.groups(event) and not bus.poll('base')
    assert not k.db.execute('SELECT 1 FROM learning_event_decisions_v08 WHERE event=?',(event,)).fetchone()
    close(c,p,k)
