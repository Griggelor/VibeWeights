import pytest
from vibeweights.abi import ContractError
from vibeweights.global_relearning import GlobalRelearning
from vibeweights.conflict_learning import ConflictLearning
from vibeweights.knowledge_learning import Domain
from vibeweights.task_learning import TaskLearning
from test_knowledge_learning_v08 import fixture, close
from test_conflict_learning_v08 import args
from test_extension_learning_v08 import setup, stage
from test_learning_initiative_v08 import Model


def scenario(tmp_path, monkeypatch):
    c,p,k,a,l,d,t,e=setup(tmp_path)
    stage(e,Model('{"entity":2,"label":2}')); assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    root=l.job('new-job')['spec']['roots'][0]
    child=Domain('child-domain',4,3); rec=k.store(a,scope='global',key='child-domain:entity:1',topic=child.topic,
        values=(2,),source='derived-fixture',status='VERIFIED',parents=(root,))
    l.prepare('child-job','child',child,[rec.id],method='DIRECT',parents=('new',)); assert l.promote('child-job',{1:2}.get,oracle_id='oracle')
    grand=Domain('grand-domain',4,3); gr=k.store(a,scope='global',key='grand-domain:entity:0',topic=grand.topic,
        values=(1,),source='derived-fixture',status='VERIFIED',parents=(rec.id,))
    l.prepare('grand-job','grand',grand,[gr.id],method='DIRECT',parents=('child',)); assert l.promote('grand-job',{0:1}.get,oracle_id='oracle')
    original=p.compute
    with monkeypatch.context() as mp:
        mp.setattr(p,'compute',lambda id,x:d.target(1) if id=='base' and x==d.encode(2) else original(id,x))
        assert t.observe('conflict',d,2)['outcome']=='CONFLICT'
    g=GlobalRelearning(t); assert g.impulses()[0]['local_open']; assert set(g.impulses()[0]['affected'])=={'base','new','child','grand'}
    f=ConflictLearning(t); truth={0:1,2:0}.get; f.stage('conflict',**args(truth)); assert f.promote('conflict',truth,oracle_id='proof')
    return c,p,k,a,l,d,t,g,child,grand,rec.id


def repair(g, predecessor='child', job='child-repair', oracle={1:0}.get, **kwargs):
    return g.stage('conflict',predecessor,job_id=job,expert_id=job+'-expert',oracle=oracle,
                   oracle_id='independent',evidence_id='fresh-proof',**kwargs)


def test_global_cross_domain_repair(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,g,child,grand,rid=scenario(tmp_path,monkeypatch)
    assert g.impulses()[0]['pending_reconstruction']==['child','grand']
    hashes={m['id']:m['sha256'] for m in c.list()}
    assert repair(g)['state']=='STAGED'; assert len(g.impulses()[0]['pending_reconstruction'])==2
    assert g.promote('child-repair',{1:0}.get,oracle_id='independent')
    assert t.observe('child-after',child,1)['answer']==0
    assert g.impulses()[0]['pending_reconstruction']==['grand']
    assert repair(g,'grand','grand-repair',{0:2}.get,method='TRAINED')['state']=='STAGED'
    assert g.promote('grand-repair',{0:2}.get,oracle_id='independent')
    assert t.observe('grand-after',grand,0)['answer']==2 and not g.impulses()
    assert k.read(rid,a).values==(2,)
    assert all(c.manifest(i)['sha256']==h for i,h in hashes.items())
    assert c.manifest('child')['state']=='QUARANTINED' and c.manifest('grand')['state']=='QUARANTINED'
    close(c,p,k)


def test_no_evidence_no_repair(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,g,child,grand,rid=scenario(tmp_path,monkeypatch)
    with pytest.raises(ContractError): repair(g,oracle={}.get)
    assert not g.attempts('conflict','child') and g.impulses()[0]['pending_reconstruction']==['child','grand']
    with pytest.raises(ContractError): repair(g,'base')
    close(c,p,k)


def test_recovery_and_retry(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,g,child,grand,rid=scenario(tmp_path,monkeypatch)
    with monkeypatch.context() as mp:
        def fail(*a): raise RuntimeError('interrupted')
        mp.setattr(l,'resume_intake',fail)
        with pytest.raises(RuntimeError): repair(g,method='TRAINED',steps=1)
    close(c,p,k); c,p,k,a,l=fixture(tmp_path); t=TaskLearning(l); g=GlobalRelearning(t)
    assert repair(g,method='TRAINED',steps=1)['state']=='STAGED'
    assert not g.promote('child-repair',{1:0}.get,oracle_id='independent')
    assert repair(g,job='child-retry')['state']=='STAGED'
    with pytest.raises(ContractError): g.promote('child-retry',{1:2}.get,oracle_id='independent')
    assert g.promote('child-retry',{1:0}.get,oracle_id='independent')
    assert g.promote('child-retry',{1:0}.get,oracle_id='independent')
    assert len(g.attempts('conflict','child'))==2 and g.impulses()[0]['pending_reconstruction']==['grand']
    close(c,p,k)
