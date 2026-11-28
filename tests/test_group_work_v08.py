from itertools import product
import pytest
from vibeweights.abi import ContractError
from vibeweights.environment import Limits
from vibeweights.knowledge_learning import Domain
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.group_work import GroupWork
from vibeweights.task_learning import TaskLearning
from vibeweights.population import ARCHIVED, SelectionEvidence
from test_knowledge_learning_v08 import fixture, close, facts


def training(affected, join=True):
    return [((k,r,s,affected,lo,rep),(int(k if join else r),))
            for k,r,s,lo,rep in product((0,1),repeat=5)]


def setup(tmp_path, *, include_shortcut=True):
    c,p,k,a,l=fixture(tmp_path)
    d=Domain('notice',4,3); ids=facts(k,a,d,{0:1})
    l.prepare('notice-job','notice',d,ids,method='DIRECT')
    assert l.promote('notice-job',{0:1}.get,oracle_id='notice-proof')
    # Deliberately reversed alphabetical order: runtime needs a second round.
    for eid,w,b,src,dst in [('z-first',2,1,'input','mid'),('a-second',3,2,'mid','answer'),('shortcut',6,5,'input','answer')]:
        if eid=='shortcut' and not include_shortcut: continue
        c.stage_affine(eid,eid,[[w]],[b],source_topic=src,target_scope='global',target_topic=dst)
        assert p.admit(eid,[((0,),(b,)),((2,),(2*w+b,))])
    bus=LearningEvents(TaskLearning(l));h=PerceptionHeads(bus)
    for eid in ('z-first','a-second','shortcut'):
        if eid=='shortcut' and not include_shortcut: continue
        join=eid!='shortcut'
        h.stage(eid+'-head',eid,['research'],training(0,join),steps=512)
        assert h.admit(eid+'-head',training(1,join),oracle_id='heldout')['accepted']
    e=bus.publish_knowledge('notice-job')
    assert len(h.tick())==(3 if include_shortcut else 2)
    assert bus.groups(e)==({'target':'explore','group':'research','experts':('a-second','z-first')},)
    return c,p,k,a,l,h,e,ids[0]


def validators(x=4):
    # Separate formula; experts never receive expected values or this callable.
    return {'mid':lambda snapshot,v:v==(2*x+1,), 'answer':lambda snapshot,v:v==(6*x+5,)}


def run(w,e,name='run',x=4,**kw):
    return w.execute(name,e,'explore','research',{'input':(x,)},'answer',kw.pop('validators',validators(x)),oracle_id='math-oracle',**kw)


def test_neural_group_multistep_and_restart(tmp_path):
    c,p,k,a,l,h,e,_=setup(tmp_path); w=GroupWork(h)
    before={m['id']:m['sha256'] for m in c.list()}
    r=run(w,e); out=r['result']; assert out['status']=='OK' and out['value']==[29.0]
    assert {row[1] for row in out['trace']}=={'z-first','a-second'}
    assert {x['source'] for x in out['hypotheses'] if x['topic']=='answer'}=={'a-second'}
    assert all(f['scope']==r['scope'] for f in out['facts'])
    assert before=={m['id']:m['sha256'] for m in c.list()}
    close(c,p,k); c,p,k,a,l=fixture(tmp_path); w=GroupWork(PerceptionHeads(LearningEvents(TaskLearning(l))))
    assert w.receipt('run')==r
    with pytest.raises(ContractError): run(w,e)
    assert run(w,e,'new-run',x=7)['result']['value']==[47.0]
    close(c,p,k)


@pytest.mark.parametrize('eid',['z-first','a-second'])
def test_member_ablation_no_external_shortcut(tmp_path,eid):
    c,p,k,a,l,h,e,_=setup(tmp_path);w=GroupWork(h)
    assert run(w,e)['result']['value']==[29.0]
    ev=SelectionEvidence('ablation',1,True,False,True,True,True,True)
    p.select(eid,ARCHIVED,expected_revision=p.revision(eid),evidence=ev)
    out=run(w,e,'ablate')['result'];assert out['status']=='STAGNATION' and out['value'] is None
    assert 'shortcut' in p.resident_ids() and c.manifest(eid)['state']==ARCHIVED
    close(c,p,k)


def test_verifier_budget_and_faults(tmp_path,monkeypatch):
    c,p,k,a,l,h,e,_=setup(tmp_path);w=GroupWork(h)
    assert run(w,e,limits=Limits(1,16,256))['result']['status']=='BUDGET'
    wrong=validators();wrong['mid']=lambda s,v:False
    out=run(w,e,'reject',validators=wrong)['result'];assert out['value'] is None and out['status']=='STAGNATION'
    def bad(*args):raise RuntimeError('failed')
    monkeypatch.setattr(p._resident['a-second'],'compute',bad)
    out=run(w,e,'fault')['result'];assert out['status']=='INVALID' and out['value'] is None
    close(c,p,k)


def test_stale_event_and_receipt_integrity(tmp_path):
    c,p,k,a,l,h,e,rid=setup(tmp_path);w=GroupWork(h);r=run(w,e)
    rec=k.read(rid,a);k.store(a,scope='global',key=rec.key,topic=rec.topic,values=rec.values,source='doubt',status='UNCERTAIN',expected=rid)
    with pytest.raises(ContractError):run(w,e,'stale')
    assert w.receipt('run')==r
    with w.db:w.db.execute("UPDATE group_runs_v08 SET receipt='{}' WHERE id='run'")
    with pytest.raises(ContractError):w.receipt('run')
    close(c,p,k)


def test_untrained_callback_group_rejected(tmp_path):
    c,p,k,a,l=fixture(tmp_path);d=Domain('notice',3,2)
    l.prepare('notice-job','notice',d,facts(k,a,d,{0:1}),method='DIRECT')
    assert l.promote('notice-job',{0:1}.get,oracle_id='oracle')
    bus=LearningEvents(TaskLearning(l));e=bus.publish_knowledge('notice-job');bus.respond(e,'notice',[('explore','research')])
    with pytest.raises(ContractError):run(GroupWork(PerceptionHeads(bus)),e)
    close(c,p,k)
