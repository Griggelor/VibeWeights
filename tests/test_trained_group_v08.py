import json
import subprocess
import sys
import pytest
from vibeweights.abi import ContractError
from vibeweights.group_learning import GroupLearning, train_affine
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_group_learning_v08 import prepare, holdout
from test_group_work_v08 import run
from test_knowledge_learning_v08 import fixture, close


def test_trained_group_independent_weights(tmp_path,monkeypatch):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    original={m['id']:m['sha256'] for m in c.list()}
    def forbidden(*args,**kwargs): raise AssertionError('analytical composition not allowed')
    monkeypatch.setattr(c,'compose_affine',forbidden)
    m=g.stage('trained','student',['first','second'],family='combined',method='TRAINED')
    assert m['origin']=='TRAINED' and m['state']=='STAGED'
    build=g.job('trained')['build'];assert build['initial_loss']>0 and build['final_loss']<1e-12
    assert g.promote('trained',holdout(),oracle_id='new-oracle')
    assert all(c.manifest(i)['sha256']==sha for i,sha in original.items())
    path=c.root/(m['sha256']+'.safetensors');isolated=tmp_path/'student.safetensors';isolated.write_bytes(path.read_bytes())
    code='''import sys,json,torch
from safetensors.torch import load_file
t=load_file(sys.argv[1]); print(json.dumps([float(torch.nn.functional.linear(torch.tensor([float(i)]),t['w'],t['b'])[0]) for i in range(-20,21)]))'''
    close(c,p,k)
    ys=json.loads(subprocess.check_output([sys.executable,'-c',code,str(isolated)],cwd=tmp_path,text=True))
    assert ys==[6*i+5 for i in range(-20,21)]


def test_holdout_rejection_and_no_admission(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    g.stage('trained','student',['first','second'],family='combined',method='TRAINED')
    with pytest.raises(ContractError):g.promote('trained',holdout(),oracle_id='math-oracle')
    with pytest.raises(ContractError):g.promote('trained',[((4,),(29,)),*holdout()],oracle_id='new-oracle')
    with pytest.raises(ContractError):p.admit('student',holdout())
    assert not g.promote('trained',[((-3,),(-12,)),*holdout()[1:]],oracle_id='new-oracle')
    assert c.manifest('student')['state']=='STAGED' and 'student' not in p.resident_ids()
    close(c,p,k)


def test_undertrained_candidate_rejected(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    g.stage('trained','student',['first','second'],family='combined',method='TRAINED',steps=1)
    assert not g.promote('trained',holdout(),oracle_id='new-oracle')
    assert g.job('trained')['build']['final_loss']>1
    close(c,p,k)


def test_recovery_after_training_before_artifact(tmp_path,monkeypatch):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    original=c.stage_affine
    def fail(*args,**kwargs):raise RuntimeError('interrupted')
    monkeypatch.setattr(c,'stage_affine',fail)
    with pytest.raises(RuntimeError):g.stage('trained','student',['first','second'],family='combined',method='TRAINED')
    stats=g.job('trained')['build']; assert stats is not None
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);g=GroupLearning(GroupWork(PerceptionHeads(LearningEvents(TaskLearning(l)))))
    with pytest.raises(ContractError):g.stage('trained','student',['first','second'],family='combined',method='TRAINED',steps=1)
    assert g.stage('trained','student',['first','second'],family='combined',method='TRAINED')['state']=='STAGED'
    assert g.job('trained')['build']==stats
    assert g.promote('trained',holdout(),oracle_id='new-oracle')
    close(c,p,k)


def test_multivariate_training_and_rank_gate():
    import torch
    # A full-rank set with no labels from the evaluation grid.
    samples=[([x,y],[2*x-3*y+4,x+5*y-2]) for x,y in [(-1,-1),(-1,1),(1,-1),(1,1)]]
    w,b,stats=train_affine(samples,256)
    assert w==[[2.,-3.],[1.,5.]] and b==[4.,-2.]
    for x,y in [(2,3),(-7,1),(0,0)]:
        assert torch.nn.functional.linear(torch.tensor([float(x),float(y)]),torch.tensor(w),torch.tensor(b)).tolist()==[2*x-3*y+4,x+5*y-2]
    with pytest.raises(ContractError):train_affine([([1,1],[0]),([2,2],[0]),([3,3],[0])],256)


def test_parent_quarantine_propagates(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    g.stage('trained','student',['first','second'],family='combined',method='TRAINED');assert g.promote('trained',holdout(),oracle_id='new-oracle')
    p.quarantine(['z-first'],reason='test-safety')
    with pytest.raises(ContractError):p.compute('student',(4,))
    assert c.manifest('student')['state']=='QUARANTINED'
    close(c,p,k)
