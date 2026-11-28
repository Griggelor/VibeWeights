"""Evidence-bound nonlinear transfer, independent holdout and immutable parents."""
import json
import subprocess
import sys
import pytest
import torch
from vibeweights.abi import ContractError
from vibeweights.factory import quadratic_features
from vibeweights.group_learning import GroupLearning, train_quadratic
from vibeweights.group_work import GroupWork
from vibeweights.knowledge_learning import Domain
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_group_work_v08 import training
from test_knowledge_learning_v08 import fixture, close, facts


def truth(x):return 6*x*x+3*x+5


def setup(tmp_path):
    c,p,k,a,l=fixture(tmp_path)
    d=Domain('notice',4,3)
    l.prepare('notice-job','notice',d,facts(k,a,d,{0:1}),method='DIRECT')
    assert l.promote('notice-job',{0:1}.get,oracle_id='notice-proof')
    c.stage_quadratic('curve','curve',[[1,2]],[1],input_width=1,
                      source_topic='input',target_scope='global',target_topic='mid')
    assert p.admit('curve',[((0,),(1,)),((2,),(11,)),((-2,),(7,))])
    c.stage_affine('tail','tail',[[3]],[2],source_topic='mid',target_scope='global',target_topic='answer')
    assert p.admit('tail',[((0,),(2,)),((2,),(8,))])
    h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    for eid in ('curve','tail'):
        h.stage(eid+'-head',eid,['research'],training(0),steps=512)
        assert h.admit(eid+'-head',training(1),oracle_id='perception-proof')['accepted']
    event=h.bus.publish_knowledge('notice-job')
    h.tick(delivery_budget=64)
    w=GroupWork(h);g=GroupLearning(w)
    runs=[]
    for i,x in enumerate(range(-3,4)):
        run='sample-'+str(i);runs.append(run)
        out=w.execute(run,event,'explore','research',{'input':(x,)},'answer',
            {'mid':lambda s,v,x=x:v==(2*x*x+x+1,), 'answer':lambda s,v,x=x:v==(truth(x),)},
            oracle_id='training-formula')['result']
        assert out['status']=='OK' and out['value']==[truth(x)]
    return c,p,k,a,l,h,w,g,event,runs


def holdout():return [((x,),(truth(x),)) for x in (-8,-5,4,6,9)]


def stage(g,runs,**kw):
    return g.stage('quadratic-job','student',runs,family='combined',
                   method='TRAINED_QUADRATIC',**kw)


def test_quadratic_transfer_weights_only_restart_and_rejoin(tmp_path,monkeypatch):
    c,p,k,a,l,h,w,g,event,runs=setup(tmp_path)
    originals={m['id']:m['sha256'] for m in c.list()}
    before=p.tensor_bytes
    def forbidden(*args,**kw):raise AssertionError('analytical composition not allowed')
    monkeypatch.setattr(c,'compose_affine',forbidden)
    m=stage(g,runs,steps=512)
    assert m['op']=='quadratic' and m['shape']==[1,1] and m['parameters']==3
    assert m['origin']=='TRAINED' and m['state']=='STAGED'
    build=g.job('quadratic-job')['build']
    assert build['feature_contract']=='quadratic-upper-v1' and build['feature_width']==2
    assert build['initial_loss']>0 and build['final_loss']<1e-12
    assert g.promote('quadratic-job',holdout(),oracle_id='independent-holdout')
    assert p.tensor_bytes==before+12
    assert all(c.manifest(i)['sha256']==sha for i,sha in originals.items())
    notice=h.bus.publish_group_knowledge('quadratic-job')
    h.stage('student-head','student',['next'],training(0),steps=512)
    assert h.admit('student-head',training(1),oracle_id='new-head-proof')['accepted']
    isolated=tmp_path/'student.safetensors'
    isolated.write_bytes((c.root/(m['sha256']+'.safetensors')).read_bytes())
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    h=PerceptionHeads(LearningEvents(TaskLearning(l)));w=GroupWork(h)
    h.tick(delivery_budget=64)
    for x in range(-20,21):assert p.compute('student',(x,))==(truth(x),)
    out=w.execute('new-group',notice,'explore','next',{'input':(11,)},'answer',
        {'answer':lambda s,v:v==(truth(11),)},oracle_id='new-group-proof')['result']
    assert out['status']=='OK' and out['value']==[truth(11)]
    code="""import json,sys,torch
from safetensors.torch import load_file
t=load_file(sys.argv[1])
x=torch.arange(-20,21,dtype=torch.float32)
z=torch.stack((x,x*x),dim=1)
print(json.dumps(torch.nn.functional.linear(z,t['w'],t['b']).flatten().tolist()))
"""
    values=json.loads(subprocess.check_output([sys.executable,'-c',code,str(isolated)],cwd=tmp_path,text=True))
    assert values==[truth(x) for x in range(-20,21)]
    p.quarantine(['curve'],reason='test-source-hold')
    with pytest.raises(ContractError):p.compute('student',(4,))
    assert c.manifest('student')['state']=='QUARANTINED' and h.bus.groups(notice)==()
    close(c,p,k)


def test_independent_admission_and_undertraining(tmp_path):
    c,p,k,a,l,h,w,g,event,runs=setup(tmp_path)
    stage(g,runs,steps=1)
    with pytest.raises(ContractError,match='independent'):
        g.promote('quadratic-job',holdout(),oracle_id='training-formula')
    with pytest.raises(ContractError,match='unseen'):
        g.promote('quadratic-job',[((0,),(truth(0),)),*holdout()],oracle_id='holdout')
    with pytest.raises(ContractError):p.admit('student',holdout())
    assert not g.promote('quadratic-job',holdout(),oracle_id='holdout')
    assert c.manifest('student')['state']=='STAGED' and 'student' not in p.resident_ids()
    close(c,p,k)


def test_quadratic_multivariate_full_rank_and_operation_contract(tmp_path):
    obs=[([x,y],[2*x*x+3*x*y-y*y+4*x-2*y+7]) for x in (-1,0,1) for y in (-1,0,1)]
    w,b,_=train_quadratic(obs,1024)
    assert w==[[4.,-2.,2.,3.,-1.]] and b==[7.]
    z=quadratic_features(torch.tensor([2.,-3.]))
    assert torch.nn.functional.linear(z,torch.tensor(w),torch.tensor(b)).tolist()==[2*4-18-9+8+6+7]
    with pytest.raises(ContractError):train_quadratic([([x,x],[x*x]) for x in range(7)],512)
    c,p,k,a,l,h,work,g,event,runs=setup(tmp_path)
    with pytest.raises(ContractError,match='nonlinear chain'):
        g.stage('affine-job','affine-student',runs,family='combined',method='TRAINED')
    with pytest.raises(ContractError,match='affine composition'):
        c.compose_affine('wrong','wrong','curve','tail')
    with pytest.raises(ContractError,match='quadratic parameter'):
        c.stage_quadratic('bad','bad',[[1]],[0],input_width=1,source_topic='input',target_scope='global',target_topic='out')
    close(c,p,k)


def test_quadratic_training_recovery(tmp_path,monkeypatch):
    c,p,k,a,l,h,w,g,event,runs=setup(tmp_path)
    def interrupted(*args,**kw):raise RuntimeError('interrupted before nonlinear artifact')
    with monkeypatch.context() as m:
        m.setattr(c,'stage_quadratic',interrupted)
        with pytest.raises(RuntimeError,match='interrupted'):stage(g,runs,steps=512)
    build=g.job('quadratic-job')['build']
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    g=GroupLearning(GroupWork(PerceptionHeads(LearningEvents(TaskLearning(l)))))
    m=stage(g,runs,steps=512)
    assert m['state']=='STAGED' and g.job('quadratic-job')['build']==build
    assert g.promote('quadratic-job',holdout(),oracle_id='independent-holdout')
    close(c,p,k)


def test_quadratic_artifact_binds_operation(tmp_path):
    c,p,k,a,l=fixture(tmp_path)
    common={'source_topic':'input','target_scope':'global','target_topic':'output'}
    affine=c.stage_affine('linear','linear',[[1,2]],[3],**common)
    quadratic=c.stage_quadratic('curve','curve',[[1,2]],[3],input_width=1,**common)
    assert affine['sha256']!=quadratic['sha256']
    assert p.admit('linear',[((2,4),(13,))])
    assert p.admit('curve',[((2,),(13,))])
    # Identical coefficient tensors do not imply identical semantics.
    raw=dict(c.manifest('curve'));raw.pop('state')
    raw['op']='affine';raw['shape']=[1,2]
    with c.db:c.db.execute('UPDATE experts SET manifest=? WHERE id=?',(json.dumps(raw,sort_keys=True),'curve'))
    with pytest.raises(ContractError,match='operation binding'):c.load('curve')
    close(c,p,k)
