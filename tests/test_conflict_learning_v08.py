import pytest
from vibeweights.abi import ContractError
from vibeweights.conflict_learning import ConflictLearning
from vibeweights.knowledge_learning import Domain
from vibeweights.task_learning import TaskLearning
from test_knowledge_learning_v08 import fixture, close, facts
from test_extension_learning_v08 import setup, stage
from test_learning_initiative_v08 import Model


def conflict(tmp_path, monkeypatch, *, detect=True):
    c,p,k,a,l,d,t,e=setup(tmp_path)
    stage(e,Model('{"entity":2,"label":2}')); assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    old=p.compute
    def faulty(id,x):
        if id=='base' and x==d.encode(2): return d.target(1)
        return old(id,x)
    with monkeypatch.context() as mp:
        mp.setattr(p,'compute',faulty)
        if detect: assert t.observe('conflict',d,2)['outcome']=='CONFLICT'
    return c,p,k,a,l,d,t,e


def args(oracle, **kwargs):
    return dict(job_id='repair-job',expert_id='repair',oracle=oracle,oracle_id='proof',evidence_id='independent-fixture',**kwargs)


@pytest.mark.parametrize('method',['DIRECT','TRAINED'])
def test_hold_and_relearn_preserves_old_competence(tmp_path,monkeypatch,method):
    c,p,k,a,l,d,t,e=conflict(tmp_path,monkeypatch); f=ConflictLearning(t)
    r=f.case('conflict')['receipt']; old=r['prior_id']; uncertain=r['uncertain_id']
    assert k.read(old,a).status=='VERIFIED' and k.read(uncertain,a).status=='CONFLICTED'
    assert old not in k.search(a,topic=d.topic)
    with pytest.raises(ContractError): k.current_verified(old,a)
    assert not p.resident_ids() and len(f.pending())==1
    assert t.observe('during-hold',d,2)['answer'] is None
    hashes={m['id']:m['sha256'] for m in c.list()}
    truth={0:1,2:0}.get  # independent new evidence corrects both competing labels
    assert f.stage('conflict',**args(truth,method=method))['state']=='STAGED'
    assert not p.resident_ids() and t.observe('during-stage',d,2)['outcome']=='CONFLICT'
    assert f.promote('conflict',truth,oracle_id='proof')
    assert t.observe('after',d,2)['answer']==0 and t.observe('retained',d,0)['answer']==1
    assert t.observe('unknown',d,1)['outcome']=='ABSTAINED'
    assert p.resident_ids()==('repair',) and not f.pending()
    assert all(c.manifest(i)['sha256']==h and c.manifest(i)['state']=='QUARANTINED' for i,h in hashes.items())
    assert f.case('conflict')['resolution']['pending_reconstruction']==[]
    assert len(l.intakes())==2 and f.promote('conflict',truth,oracle_id='proof')
    close(c,p,k)


def test_unknown_truth_stays_uncertain_after_restart(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,e=conflict(tmp_path,monkeypatch);f=ConflictLearning(t)
    with pytest.raises(ContractError): f.stage('conflict',**args({}.get))
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);f=ConflictLearning(t)
    assert len(f.pending())==1 and f.case('conflict')['resolution'] is None
    assert t.observe('restart-conflict',d,2)['outcome']=='CONFLICT'
    assert len(c.list())==2 and not p.resident_ids()
    close(c,p,k)


def test_wrong_promotion_and_insufficient_training_remain_open(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,e=conflict(tmp_path,monkeypatch);f=ConflictLearning(t);truth={0:1,2:2}.get
    f.stage('conflict',**args(truth,steps=1))
    with pytest.raises(ContractError): f.promote('conflict',{0:1,2:1}.get,oracle_id='proof')
    assert not f.promote('conflict',truth,oracle_id='proof')
    assert t.observe('still-conflict',d,2)['outcome']=='CONFLICT' and not p.resident_ids()
    assert not f.case('conflict')['closed'];close(c,p,k)


def test_recovery_before_build_and_after_admission(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,e=conflict(tmp_path,monkeypatch);f=ConflictLearning(t);truth={0:1,2:2}.get
    def crash(*a,**kw): raise RuntimeError('interrupted')
    with monkeypatch.context() as mp:
        mp.setattr(l,'resume_intake',crash)
        with pytest.raises(RuntimeError): f.stage('conflict',**args(truth))
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);f=ConflictLearning(t)
    assert f.stage('conflict',**args(truth))['state']=='STAGED'
    original=l.promote
    def admitted_then_crash(*a,**kw):
        assert original(*a,**kw); raise RuntimeError('after admission')
    with monkeypatch.context() as mp:
        mp.setattr(l,'promote',admitted_then_crash)
        with pytest.raises(RuntimeError): f.promote('conflict',truth,oracle_id='proof')
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);f=ConflictLearning(t)
    assert t.observe('after-crash',d,2)['outcome']=='CONFLICT'
    assert f.promote('conflict',truth,oracle_id='proof')
    assert t.observe('recovered',d,2)['answer']==2 and len(c.list())==3
    close(c,p,k)


def test_transitive_quarantine_and_pending_reconstruction(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,e=setup(tmp_path);stage(e,Model('{"entity":2,"label":2}'));assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    other=Domain('other',4,3)
    l.prepare('child-job','child',other,facts(k,a,other,{1:1}),method='DIRECT',parents=('new',))
    assert l.promote('child-job',{1:1}.get,oracle_id='oracle')
    unrelated=Domain('unrelated',4,3)
    l.prepare('unrelated-job','unrelated',unrelated,facts(k,a,unrelated,{1:2}),method='DIRECT')
    assert l.promote('unrelated-job',{1:2}.get,oracle_id='oracle')
    original=p.compute
    with monkeypatch.context() as mp:
        mp.setattr(p,'compute',lambda id,x: d.target(1) if id=='base' and x==d.encode(2) else original(id,x))
        t.observe('conflict',d,2)
    f=ConflictLearning(t);truth={0:1,2:2}.get
    assert p.resident_ids()==('unrelated',) and c.manifest('child')['state']=='QUARANTINED'
    assert f.stage('conflict',**args(truth))['state']=='STAGED'
    assert f.promote('conflict',truth,oracle_id='proof')
    assert f.reconstruction_pending()==(('conflict',('child',)),)
    assert t.observe('other-answer',unrelated,1)['answer']==2
    close(c,p,k)


def test_case_survives_failure_before_task_journal(tmp_path,monkeypatch):
    import vibeweights.task_learning as task_module
    c,p,k,a,l,d,t,e=setup(tmp_path);stage(e,Model('{"entity":2,"label":2}'));assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    original=p.compute; save=task_module.record_conflict
    def crash(*a,**kw):
        save(*a,**kw);raise RuntimeError('after source commit')
    with monkeypatch.context() as mp:
        mp.setattr(p,'compute',lambda id,x:d.target(1) if id=='base' and x==d.encode(2) else original(id,x))
        mp.setattr(task_module,'record_conflict',crash)
        with pytest.raises(RuntimeError):t.observe('conflict',d,2)
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);f=ConflictLearning(t)
    assert t.observation('conflict')['outcome']=='CONFLICT' and len(f.pending())==1
    assert t.observe('after-crash',d,2)['answer'] is None and not p.resident_ids()
    close(c,p,k)
