"""Process-exit and fault-injection tests for bounded learning recovery."""
import os
import subprocess
import sys
import pytest
from vibeweights.abi import ContractError
from vibeweights.bootstrap import BootstrapExpert, BootstrapSpec
from vibeweights.knowledge_learning import Domain, KnowledgeLearning
from vibeweights.relation_learning import stage_teacher_knowledge
from test_knowledge_learning_v08 import fixture, facts, close
from test_relation_learning_v08 import setup_rel, derive


def reopen(path):
    return fixture(path)


@pytest.mark.parametrize('point', ['intake', 'job', 'build', 'artifact', 'receipt', 'admission'])
def test_process_exit_and_recovery(tmp_path, point):
    # Real unclean exit at committed boundaries (not machine power loss).
    code = '''
import os,sys
from pathlib import Path
from vibeweights.factory import Catalog
from vibeweights.population import Population
from vibeweights.knowledge import KnowledgeLedger, Access
from vibeweights.knowledge_learning import KnowledgeLearning, Domain
from vibeweights.bootstrap import BootstrapExpert, BootstrapSpec
from vibeweights.relation_learning import stage_teacher_knowledge
p=Path(sys.argv[1]); point=sys.argv[2]
c=Catalog(p/'experts'); pop=Population(c,tensor_budget=65536)
k=KnowledgeLedger(p/'knowledge.sqlite'); l=KnowledgeLearning(pop,k,Access('host'))
class Teacher(BootstrapExpert):
    def propose(self,text): return {'proposal':'{"facts":[{"entity":0,"label":2}]}','verified':False}
if point=='intake': l.resume_intake=lambda *a,**kw: os._exit(73)
if point=='job': l.resume_build=lambda *a,**kw: os._exit(73)
if point=='build': c.stage_affine=lambda *a,**kw: os._exit(73)
if point=='artifact':
    stage=c.stage_affine
    def stop(*a,**kw): stage(*a,**kw); os._exit(73)
    c.stage_affine=stop
if point=='receipt': pop.admit=lambda *a,**kw: os._exit(73)
if point=='admission':
    admit=c.admit
    def stop(*a,**kw): assert admit(*a,**kw); os._exit(73)
    c.admit=stop
stage_teacher_knowledge(l,Teacher(BootstrapSpec('teacher','fixture')),domain=Domain('recovery',4,3),request='fixture',job_id='job',expert_id='expert',oracle={0:2}.get,oracle_id='oracle',method='TRAINED')
l.promote('job',{0:2}.get,oracle_id='oracle')
raise RuntimeError('exit injection was not reached')
'''
    env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'),os.environ.get('PYTHONPATH','')]))
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path),point],env=env,timeout=45)
    assert result.returncode==73
    c,p,k,a,l=reopen(tmp_path)
    assert len(l.intakes())==1 and len(k.search(a,topic='recovery'))==1
    assert l.intakes()[0]['state']=={'intake':'PENDING','job':'INCOMPLETE','build':'INCOMPLETE','artifact':'STAGED','receipt':'STAGED','admission':'ACTIVE'}[point]
    m=l.resume_intake('job'); assert l.resume_intake('job')==m
    assert len(c.list())==1
    if point!='admission':
        assert m['state']=='STAGED' and p.resident_ids()==()
        assert l.promote('job',{0:2}.get,oracle_id='oracle')
    assert p.resident_ids()==('expert',)
    d=Domain('recovery',4,3)
    assert [d.decode(p.compute('expert',d.encode(i))) for i in range(-1,5)]==[None,2,None,None,None,None]
    close(c,p,k)


def test_relation_resume_and_stale_sources(tmp_path, monkeypatch):
    c,p,k,a,l,d1,d2,ar,br=setup_rel(tmp_path)
    with monkeypatch.context() as mp:
        def stop(*args,**kwargs): raise RuntimeError('after facts committed')
        mp.setattr(l,'resume_intake',stop)
        with pytest.raises(RuntimeError): derive(l)
    assert l.intakes()[0]['state']=='PENDING'
    assert len(k.search(a,topic='object-class'))==3
    k.store(a,scope='global',key=d1.topic+':entity:0',topic=d1.topic,values=(3,),source='correction',status='VERIFIED',expected=ar[0])
    with pytest.raises(ContractError): l.resume_intake('j3')
    assert not any(m['id']=='e3' for m in c.list())
    close(c,p,k)


def test_relation_intent_and_facts_rollback_together(tmp_path, monkeypatch):
    c,p,k,a,l,*_=setup_rel(tmp_path)
    orig=l.record_intake
    def stop(**kwargs):
        orig(**kwargs); raise RuntimeError('before commit')
    monkeypatch.setattr(l,'record_intake',stop)
    with pytest.raises(RuntimeError): derive(l)
    assert l.intakes()==[] and k.search(a,topic='object-class')==()
    assert len(c.list())==2
    close(c,p,k)


def test_plain_build_recovery(tmp_path, monkeypatch):
    c,p,k,a,l=fixture(tmp_path); d=Domain('plain',5,3); ids=facts(k,a,d,{0:2})
    with monkeypatch.context() as mp:
        def stop(*args,**kwargs): raise RuntimeError('stage interrupted')
        mp.setattr(c,'stage_affine',stop)
        with pytest.raises(RuntimeError): l.prepare('job','expert',d,ids,method='DIRECT')
    assert l.job('job')['state']=='INCOMPLETE'
    close(c,p,k); c,p,k,a,l=reopen(tmp_path)
    assert l.resume_build('job')['state']=='STAGED'
    assert l.promote('job',{0:2}.get,oracle_id='oracle')
    with pytest.raises(ContractError): l.resume_build('job')
    close(c,p,k)


def test_intake_cannot_resume_with_changed_job_identity(tmp_path, monkeypatch):
    c,p,k,a,l,*_=setup_rel(tmp_path)
    with monkeypatch.context() as mp:
        def stop(*args,**kwargs): raise RuntimeError('after facts committed')
        mp.setattr(l,'resume_intake',stop)
        with pytest.raises(RuntimeError): derive(l)
    d=Domain('other',4,3)
    # A pending intake cannot be hijacked by an unrelated direct job.
    with pytest.raises(ContractError): l.prepare('j3','different',d,facts(k,a,d,{0:1}),method='DIRECT')
    assert l.intakes()[0]['state']=='PENDING'
    assert l.resume_intake('j3')['id']=='e3'
    close(c,p,k)


def test_economics_and_sequential_retention(tmp_path):
    import runpy
    from pathlib import Path
    run=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'learning_economics_demo.py'))['run']
    result=run(tmp_path,packages=10,repeats=16)
    assert result['restart_correct']==result['restart_cases']==100
    assert result['originals_unchanged']
    assert all(x['correct']==x['cases'] for x in result['retention'])
    assert result['population_parameters_after_consolidation']==result['parent_parameters']+result['aggregate_parameters']
    assert all(x['correct']==16 for x in result['timings'].values())
