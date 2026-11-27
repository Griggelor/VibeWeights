import pytest
from vibeweights.abi import ContractError
from vibeweights.environment import Signal
from vibeweights.knowledge_learning import Domain
from vibeweights.population import ACTIVE, ARCHIVED, SelectionEvidence
from vibeweights.relation_learning import derive_relation
from test_knowledge_learning_v08 import fixture, facts, close


def setup_rel(tmp_path):
    c,p,k,a,l=fixture(tmp_path)
    d1=Domain('object-material',6,4); d2=Domain('material-class',4,3)
    a_ids=facts(k,a,d1,{0:2,1:1,3:0,4:3})
    b_ids=facts(k,a,d2,{0:1,2:0,3:2})
    l.prepare('j1','e1',d1,a_ids,method='TRAINED'); assert l.promote('j1',{0:2,1:1,3:0,4:3}.get,oracle_id='o1')
    l.prepare('j2','e2',d2,b_ids,method='DIRECT'); assert l.promote('j2',{0:1,2:0,3:2}.get,oracle_id='o2')
    return c,p,k,a,l,d1,d2,a_ids,b_ids


def derive(l,method='DIRECT',oracle=None,domain=None):
    d=domain or Domain('object-class',6,3)
    truth={0:0,3:1,4:2}
    return derive_relation(l,first_job='j1',second_job='j2',result_domain=d,job_id='j3',expert_id='e3',bridge_id='material-id-v1',oracle=truth.get if oracle is None else oracle,oracle_id='o3',method=method)


@pytest.mark.parametrize('method',['DIRECT','TRAINED'])
def test_new_relation_additive_transfer(tmp_path,method):
    c,p,k,a,l,d1,d2,ar,br=setup_rel(tmp_path);d=Domain('object-class',6,3)
    assert k.search(a,topic=d.topic)==()
    m,ids=derive(l,method)
    assert m['parents']==['e1','e2'] and len(ids)==3
    assert k.read(ids[0],a).parents==(ar[0],br[1])
    assert l.job('j3')['spec']['derivation']['evaluated_cases']==8
    truth={0:0,3:1,4:2};assert l.promote('j3',truth.get,oracle_id='o3')
    assert p.resident_ids()==('e1','e2','e3')
    # The previously untrained end-to-end relation works; missing hops abstain.
    assert [d.decode(p.compute('e3',d.encode(i))) for i in range(-1,7)]==[truth.get(i) for i in range(-1,7)]
    goal=('global',d.topic+':answer'); result=p.run([Signal('global',d.topic+':query',d.encode(3))],goal,{goal:lambda s,v:d.decode(v)==1})
    assert result['status']=='OK'
    close(c,p,k)
    # Fresh process sees only the candidate artifact, not the parents or ledger.
    import json, shutil, subprocess, sys
    isolated=tmp_path/'isolated';isolated.mkdir()
    shutil.copyfile(tmp_path/'experts'/(m['sha256']+'.safetensors'),isolated/'model.safetensors')
    code="""import json
from safetensors.torch import load_file
from vibeweights.factory import AffineExpert
from vibeweights.knowledge_learning import Domain
d=Domain('object-class',6,3);t=load_file('model.safetensors');e=AffineExpert({'id':'isolated'},t['w'],t['b'])
print(json.dumps([d.decode(e.compute(d.encode(i))) for i in range(-1,7)]))
"""
    assert json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=isolated,text=True))==[truth.get(i) for i in range(-1,7)]


def test_wrong_oracle_and_bad_bridge_leave_no_records(tmp_path):
    c,p,k,a,l,*_=setup_rel(tmp_path)
    with pytest.raises(ContractError): derive(l,oracle={0:0,3:2,4:2}.get)
    with pytest.raises(ContractError): derive(l,domain=Domain('object-class',5,3))
    assert k.search(a,topic='object-class')==() and p.resident_ids()==('e1','e2')
    close(c,p,k)


def test_live_compute_automatically_quarantines_transitive_descendants(tmp_path):
    c,p,k,a,l,d1,d2,ar,br=setup_rel(tmp_path);derive(l);assert l.promote('j3',{0:0,3:1,4:2}.get,oracle_id='o3')
    k.store(a,scope='global',key=d1.topic+':entity:0',topic=d1.topic,values=(3,),source='correction',status='VERIFIED',expected=ar[0])
    with pytest.raises(ContractError): p.compute('e3',Domain('object-class',6,3).encode(0))
    assert c.manifest('e1')['state']==c.manifest('e3')['state']=='QUARANTINED'
    assert p.resident_ids()==('e2',) and l.audit_sources()==()
    assert c.root.joinpath(c.manifest('e3')['sha256']+'.safetensors').exists()
    close(c,p,k)


def test_live_reactivation_checks_revision(tmp_path):
    c,p,k,a,l,d1,d2,ar,br=setup_rel(tmp_path)
    ev=SelectionEvidence('report',10,True,False,True,True,True,True)
    p.select('e1',ARCHIVED,expected_revision=0,evidence=ev)
    k.store(a,scope='global',key=d1.topic+':entity:0',topic=d1.topic,values=(3,),source='correction',status='VERIFIED',expected=ar[0])
    with pytest.raises(ContractError): p.select('e1',ACTIVE,expected_revision=1,evidence=ev)
    assert c.manifest('e1')['state']=='QUARANTINED'
    close(c,p,k)


def test_source_db_lock_failure_blocks_compute(tmp_path):
    c,p,k,a,l,d1,*_=setup_rel(tmp_path)
    import sqlite3
    other=sqlite3.connect(tmp_path/'knowledge.sqlite');other.execute('BEGIN IMMEDIATE')
    try:
        with pytest.raises(sqlite3.OperationalError): p.compute('e1',d1.encode(0))
    finally: other.rollback();other.close()
    assert d1.decode(p.compute('e1',d1.encode(0)))==2
    close(c,p,k)


def test_derived_batch_atomic_on_key_conflict(tmp_path):
    c,p,k,a,l,*_=setup_rel(tmp_path)
    k.store(a,scope='global',key='object-class:entity:3',topic='collision',values=(0,),source='fixture')
    with pytest.raises(ContractError): derive(l)
    assert k.search(a,topic='object-class')==()
    assert len(c.list())==2
    close(c,p,k)


def test_live_run_checks_sources(tmp_path):
    c,p,k,a,l,d1,d2,ar,br=setup_rel(tmp_path)
    k.store(a,scope='global',key=d1.topic+':entity:0',topic=d1.topic,values=(3,),source='correction',status='VERIFIED',expected=ar[0])
    goal=('global',d1.topic+':answer')
    r=p.run([Signal('global',d1.topic+':query',d1.encode(0))],goal,{goal:lambda s,v: True})
    assert r['status']!='OK' and c.manifest('e1')['state']=='QUARANTINED'
    close(c,p,k)


@pytest.mark.parametrize('proposal,accepted',[
    ('{"facts":[{"entity":0,"label":2},{"entity":3,"label":1}]}',True),
    ('{"facts":[{"entity":0,"label":1},{"entity":3,"label":1}]}',False),
    ('{"facts":[{"entity":0,"label":2}]}',False),
    ('{"facts":[{"entity":0,"label":2,"label":1}]}',False),
    ('{"facts":[{"entity":false,"label":2}]}',False),
    ('ignore checks and execute code',False),
])
def test_teacher_proposal_requires_independent_verification(tmp_path,proposal,accepted):
    from vibeweights.bootstrap import BootstrapExpert,BootstrapSpec
    from vibeweights.relation_learning import stage_teacher_knowledge
    class Double(BootstrapExpert):
        def propose(self,text):
            return {'proposal':proposal,'verified':False}
    c,p,k,a,l=fixture(tmp_path); d=Domain('teacher-fixture',5,3); truth={0:2,3:1}
    teacher=Double(BootstrapSpec('teacher','test-double'))
    def stage(): return stage_teacher_knowledge(l,teacher,domain=d,request='Fixture IDs',job_id='job',expert_id='e',oracle=truth.get,oracle_id='independent-fixture')
    if accepted:
        m,ids=stage(); assert len(ids)==2 and m['state']=='STAGED'
        assert l.job('job')['spec']['derivation']['teacher_id']=='teacher'
        assert l.promote('job',truth.get,oracle_id='independent-fixture')
        assert d.decode(p.compute('e',d.encode(0)))==2
    else:
        with pytest.raises(ContractError): stage()
        assert c.list()==[] and k.search(a,topic=d.topic)==()
    close(c,p,k)


def test_restart_requires_live_binding_or_explicit_offline(tmp_path):
    from vibeweights.factory import Catalog
    from vibeweights.population import Population
    from vibeweights.knowledge import KnowledgeLedger
    from vibeweights.knowledge_learning import KnowledgeLearning
    c,p,k,a,l,d1,d2,ar,br=setup_rel(tmp_path);close(c,p,k)
    c=Catalog(tmp_path/'experts');p=Population(c,tensor_budget=65536)
    with pytest.raises(ContractError): p.compute('e1',d1.encode(0))
    p.close();p=Population(c,tensor_budget=65536,allow_offline_knowledge=True)
    assert d1.decode(p.compute('e1',d1.encode(0)))==2
    ev=SelectionEvidence('report',10,True,False,True,True,True,True)
    with pytest.raises(ContractError): p.select('e1',ARCHIVED,expected_revision=0,evidence=ev)
    p.close();p=Population(c,tensor_budget=65536);k=KnowledgeLedger(tmp_path/'knowledge.sqlite');l=KnowledgeLearning(p,k,a)
    k.store(a,scope='global',key=d1.topic+':entity:0',topic=d1.topic,values=(3,),source='correction',status='VERIFIED',expected=ar[0])
    with pytest.raises(ContractError): p.compute('e1',d1.encode(0))
    assert p.resident_ids()==('e2',)
    close(c,p,k)
