"""Learned hidden layers, independently validated from verified coalition observations."""
import json
import math
import subprocess
import sys
import pytest
import torch
from vibeweights.abi import ContractError
from vibeweights.group_learning import GroupLearning, train_mlp
from vibeweights.group_work import GroupWork
from vibeweights.knowledge_learning import Domain
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_group_work_v08 import training
from test_knowledge_learning_v08 import fixture, close, facts


def truth(x):return 3*max(0,2*x-1)+2


def same(a,b):return math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6)


def holdout():
    return [((x,),(truth(x),)) for x in (-8,-5,.25,.75,5,8)]


def setup(tmp_path):
    c,p,k,a,l=fixture(tmp_path)
    d=Domain('notice',4,3)
    l.prepare('notice-job','notice',d,facts(k,a,d,{0:1}),method='DIRECT')
    assert l.promote('notice-job',{0:1}.get,oracle_id='notice-proof')
    c.stage_mlp('hinge','hinge',[[1]],[0],hidden_matrix=[[2]],hidden_bias=[-1],
                source_topic='input',target_scope='global',target_topic='mid')
    assert p.admit('hinge',[((0,),(0,)),((2,),(3,)),((-2,),(0,))])
    c.stage_affine('tail','tail',[[3]],[2],source_topic='mid',target_scope='global',target_topic='answer')
    assert p.admit('tail',[((0,),(2,)),((2,),(8,))])
    h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    for eid in ('hinge','tail'):
        h.stage(eid+'-head',eid,['research'],training(0),steps=512)
        assert h.admit(eid+'-head',training(1),oracle_id='perception-proof')['accepted']
    event=h.bus.publish_knowledge('notice-job')
    h.tick(delivery_budget=64)
    w=GroupWork(h);g=GroupLearning(w)
    runs=[]
    for i in range(-8,9):
        x=i/2;run=f'sample-{i+8:02}';runs.append(run)
        out=w.execute(run,event,'explore','research',{'input':(x,)},'answer',
            {'mid':lambda s,v,x=x:v==(max(0,2*x-1),), 'answer':lambda s,v,x=x:v==(truth(x),)},
            oracle_id='training-formula')['result']
        assert out['status']=='OK' and out['value']==[truth(x)]
    return c,p,k,a,l,h,w,g,event,runs


def stage(g,runs,**kw):
    return g.stage('mlp-job','student',runs,family='combined',
                   method='TRAINED_MLP',hidden_width=2,**kw)


def test_mlp_transfer_independent_weights_restart_and_rejoin(tmp_path,monkeypatch):
    c,p,k,a,l,h,w,g,event,runs=setup(tmp_path)
    originals={m['id']:m['sha256'] for m in c.list()}
    before=p.tensor_bytes
    def forbidden(*args,**kw):raise AssertionError('analytical composition not allowed')
    monkeypatch.setattr(c,'compose_affine',forbidden)
    m=stage(g,runs,steps=256)
    assert m['op']=='mlp' and m['shape']==[1,1] and m['hidden_width']==2 and m['parameters']==7
    build=g.job('mlp-job')['build']
    assert build['optimizer']=='LBFGS' and build['initial_loss']>1 and build['final_loss']<1e-12
    assert g.promote('mlp-job',holdout(),oracle_id='independent-holdout')
    assert p.tensor_bytes==before+28
    assert all(c.manifest(i)['sha256']==sha for i,sha in originals.items())
    notice=h.bus.publish_group_knowledge('mlp-job')
    h.stage('student-head','student',['next'],training(0),steps=512)
    assert h.admit('student-head',training(1),oracle_id='new-head-proof')['accepted']
    isolated=tmp_path/'student.safetensors'
    isolated.write_bytes((c.root/(m['sha256']+'.safetensors')).read_bytes())
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    h=PerceptionHeads(LearningEvents(TaskLearning(l)));w=GroupWork(h)
    h.tick(delivery_budget=64)
    for i in range(-32,33):
        x=i/4;assert same(p.compute('student',(x,))[0],truth(x))
    out=w.execute('new-group',notice,'explore','next',{'input':(7.5,)},'answer',
        {'answer':lambda s,v:same(v[0],truth(7.5))},oracle_id='new-group-proof')['result']
    assert out['status']=='OK' and same(out['value'][0],truth(7.5))
    code="""import json,sys,torch
from safetensors.torch import load_file
t=load_file(sys.argv[1])
x=torch.arange(-32,33,dtype=torch.float32).reshape(-1,1)/4
print(json.dumps(torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x,t['hw'],t['hb'])),t['w'],t['b']).flatten().tolist()))
"""
    values=json.loads(subprocess.check_output([sys.executable,'-c',code,str(isolated)],cwd=tmp_path,text=True))
    assert len(values)==65 and all(same(v,truth(i/4)) for i,v in zip(range(-32,33),values))
    p.quarantine(['hinge'],reason='test-source-hold')
    with pytest.raises(ContractError):p.compute('student',(4,))
    assert c.manifest('student')['state']=='QUARANTINED' and h.bus.groups(notice)==()
    close(c,p,k)


def test_mlp_rejection_and_independent_holdout(tmp_path):
    c,p,k,a,l,h,w,g,event,runs=setup(tmp_path)
    stage(g,runs,steps=1)
    with pytest.raises(ContractError,match='independent'):
        g.promote('mlp-job',holdout(),oracle_id='training-formula')
    with pytest.raises(ContractError,match='unseen'):
        g.promote('mlp-job',[((0,),(truth(0),)),*holdout()],oracle_id='holdout')
    with pytest.raises(ContractError):p.admit('student',holdout())
    assert not g.promote('mlp-job',holdout(),oracle_id='holdout')
    assert c.manifest('student')['state']=='STAGED' and 'student' not in p.resident_ids()
    close(c,p,k)


def test_mlp_recovery(tmp_path,monkeypatch):
    c,p,k,a,l,h,w,g,event,runs=setup(tmp_path)
    def interrupted(*args,**kw):raise RuntimeError('interrupted before MLP artifact')
    with monkeypatch.context() as m:
        m.setattr(c,'stage_mlp',interrupted)
        with pytest.raises(RuntimeError,match='interrupted'):stage(g,runs,steps=256)
    build=g.job('mlp-job')['build']
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path)
    g=GroupLearning(GroupWork(PerceptionHeads(LearningEvents(TaskLearning(l)))))
    m=stage(g,runs,steps=256)
    assert m['state']=='STAGED' and g.job('mlp-job')['build']==build
    assert g.promote('mlp-job',holdout(),oracle_id='independent-holdout')
    close(c,p,k)


def test_mlp_contract_and_budget(tmp_path):
    c,p,k,a,l=fixture(tmp_path,budget=28)
    common={'source_topic':'input','target_scope':'global','target_topic':'output'}
    m=c.stage_mlp('net','net',[[2,3]],[1],hidden_matrix=[[1],[-1]],hidden_bias=[0,0],**common)
    assert m['parameters']==7 and p.admit('net',[((2,),(5,)),((-2,),(7,))])
    assert p.tensor_bytes==28
    c.stage_affine('extra','extra',[[1]],[0],**common)
    with pytest.raises(ContractError,match='resident tensor budget'):p.admit('extra',[((1,),(1,))])
    with pytest.raises(ContractError,match='MLP hidden tensor contract'):
        c.stage_mlp('bad','bad',[[1]],[0],hidden_matrix=[[1],[-1]],hidden_bias=[0,0],**common)
    with pytest.raises(ContractError,match='affine composition'):
        c.compose_affine('wrong','wrong','net','net')
    raw=dict(c.manifest('net'));raw.pop('state');raw['hidden_width']=1
    with c.db:c.db.execute('UPDATE experts SET manifest=? WHERE id=?',(json.dumps(raw,sort_keys=True),'net'))
    with pytest.raises(ContractError,match='operation binding'):c.load('net')
    with pytest.raises(ContractError,match='MLP training dimensions/budget'):
        train_mlp([([0],[0]),([1],[1])],256,2)
    close(c,p,k)
