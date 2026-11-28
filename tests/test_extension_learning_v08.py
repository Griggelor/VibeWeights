import pytest
from vibeweights.abi import ContractError
from vibeweights.knowledge_learning import Domain
from vibeweights.task_learning import TaskLearning
from vibeweights.extension_learning import ExtensionLearning
from test_knowledge_learning_v08 import fixture, close, facts
from test_learning_initiative_v08 import Model


def setup(tmp_path):
    c,p,k,a,l=fixture(tmp_path);d=Domain('catalog',4,3)
    l.prepare('base-job','base',d,facts(k,a,d,{0:1}),method='DIRECT')
    assert l.promote('base-job',{0:1}.get,oracle_id='oracle')
    t=TaskLearning(l);assert t.observe('gap',d,2)['outcome']=='ABSTAINED'
    return c,p,k,a,l,d,t,ExtensionLearning(t)


def stage(e,m,**kw):
    args=dict(request='learn entity 2',job_id='new-job',expert_id='new',oracle={0:1,2:2}.get,oracle_id='oracle')
    args.update(kw);return e.stage('gap',m,**args)

@pytest.mark.parametrize('method',['DIRECT','TRAINED'])
def test_extension_additive_restart(tmp_path,method):
    c,p,k,a,l,d,t,e=setup(tmp_path);sha=c.manifest('base')['sha256'];m=Model('{"entity":2,"label":2}')
    assert stage(e,m,method=method)['state']=='STAGED' and p.resident_ids()==('base',)
    assert e.promote('new-job',{0:1,2:2}.get,oracle_id='oracle')
    assert t.observe('after',d,2)['answer']==2 and t.observe('old',d,0)['answer']==1
    assert d.decode(p.compute('new',d.encode(0))) is None and c.manifest('base')['sha256']==sha
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);e=ExtensionLearning(t)
    assert stage(e,m,method=method)['state']=='ACTIVE' and len(m.prompts)==1
    assert t.observe('restart',d,2)['answer']==2 and len(l.intakes())==1 and len(c.list())==2
    close(c,p,k)

@pytest.mark.parametrize('raw',['{"entity":2,"label":1}','{"entity":1,"label":2}','{"entity":2,"label":2,"label":1}','{"entity":true,"label":2}'])
def test_bad_teacher_no_writes(tmp_path,raw):
    c,p,k,a,l,d,t,e=setup(tmp_path)
    with pytest.raises(ContractError): stage(e,Model(raw))
    assert not l.intakes() and len(c.list())==1 and len(k.search(a,topic=d.topic))==1
    close(c,p,k)


def test_existing_fact_blocks_before_teacher(tmp_path):
    c,p,k,a,l,d,t,e=setup(tmp_path);facts(k,a,d,{2:2});m=Model('{}')
    with pytest.raises(ContractError): stage(e,m)
    assert not m.prompts and not l.intakes();close(c,p,k)


def test_recheck_after_model_io(tmp_path):
    c,p,k,a,l,d,t,e=setup(tmp_path)
    class Racing(Model):
        def propose(self,text):
            facts(k,a,d,{2:2});return super().propose(text)
    with pytest.raises(ContractError): stage(e,Racing('{"entity":2,"label":2}'))
    assert not l.intakes() and len(c.list())==1;close(c,p,k)


def test_pending_resume_after_failure(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,e=setup(tmp_path);m=Model('{"entity":2,"label":2}')
    with monkeypatch.context() as mp:
        def crash(*a): raise RuntimeError('interrupted')
        mp.setattr(l,'resume_intake',crash)
        with pytest.raises(RuntimeError): stage(e,m)
    assert len(l.intakes())==1 and l.intakes()[0]['state']=='PENDING'
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);e=ExtensionLearning(TaskLearning(l))
    assert stage(e,m)['state']=='STAGED' and len(m.prompts)==1
    with pytest.raises(ContractError): stage(e,m,request='changed')
    assert not e.promote('new-job',{2:1}.get,oracle_id='oracle')
    assert p.resident_ids()==('base',);close(c,p,k)


def test_conflict_review_no_mutation(tmp_path,monkeypatch):
    c,p,k,a,l,d,t,e=setup(tmp_path);stage(e,Model('{"entity":2,"label":2}'));assert e.promote('new-job',{2:2}.get,oracle_id='oracle')
    original=p.compute
    def faulty(id,x):
        if id=='base' and x==d.encode(2): return d.target(1)
        return original(id,x)
    monkeypatch.setattr(p,'compute',faulty)
    assert t.observe('conflict',d,2)['outcome']=='CONFLICT'
    r=e.review_conflict('review','conflict',{2:2}.get,oracle_id='oracle')
    assert r['disagreeing']==['base'] and r['outcome']=='EVIDENCE_RECORDED'
    assert e.review_conflict('review','conflict',{2:2}.get,oracle_id='oracle')==r
    assert p.resident_ids()==('base','new')
    assert e.review_conflict('unknown','conflict',{}.get,oracle_id='oracle')['outcome']=='UNRESOLVED'
    with pytest.raises(ContractError): e.review_conflict('review','conflict',{2:1}.get,oracle_id='oracle')
    close(c,p,k)
