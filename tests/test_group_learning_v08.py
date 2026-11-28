import json
import shutil
import subprocess
import sys
import pytest
from vibeweights.abi import ContractError
from vibeweights.group_learning import GroupLearning
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning
from test_group_work_v08 import setup, run
from test_knowledge_learning_v08 import fixture, close


def prepare(tmp_path):
    c,p,k,a,l,h,e,rid=setup(tmp_path, include_shortcut=False); w=GroupWork(h); g=GroupLearning(w)
    run(w,e,'first',x=4); run(w,e,'second',x=7)
    return c,p,k,a,l,h,e,rid,w,g


def holdout(): return [((-3,),(-13,)),((0,),(5,)),((11,),(71,))]


def test_group_consolidation_restart_and_weight_only(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    old={m['id']:m['sha256'] for m in c.list()}; before=p.tensor_bytes
    m=g.stage('fuse','sum',['second','first'],family='combined')
    assert m['parents']==['z-first','a-second'] and m['parameters']==2 and m['state']=='STAGED'
    assert p.tensor_bytes==before
    with pytest.raises(ContractError): p.admit('sum',holdout())
    close(c,p,k); c,p,k,a,l=fixture(tmp_path);g=GroupLearning(GroupWork(PerceptionHeads(LearningEvents(TaskLearning(l)))))
    assert g.stage('fuse','sum',['first','second'],family='combined')==m
    assert g.promote('fuse',holdout(),oracle_id='heldout-math')
    assert g.promote('fuse',holdout(),oracle_id='heldout-math')
    assert p.tensor_bytes==before+8 and p.compute('sum',(13,))==(83.,)
    assert old=={x:c.manifest(x)['sha256'] for x in old}
    iso=tmp_path/'isolated';iso.mkdir();shutil.copyfile(c.root/(m['sha256']+'.safetensors'),iso/'model.safetensors')
    code='from safetensors.torch import load_file; from vibeweights.factory import AffineExpert; import json; t=load_file("model.safetensors"); e=AffineExpert({"id":"isolated"},t["w"],t["b"]); print(json.dumps([e.compute((x,)) for x in range(-10,11)]))'
    assert json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=iso,text=True))==[[6*x+5] for x in range(-10,11)]
    close(c,p,k)


def test_failed_holdout_and_immutable_report(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path);g.stage('fuse','sum',['first','second'],family='combined')
    with pytest.raises(ContractError):g.promote('fuse',[((4,),(29,)),*holdout()[:2]],oracle_id='bad')
    bad=holdout();bad[0]=((-3,),(0,))
    assert not g.promote('fuse',bad,oracle_id='bad')
    assert c.manifest('sum')['state']=='STAGED' and 'sum' not in p.resident_ids()
    with pytest.raises(ContractError):g.promote('fuse',holdout(),oracle_id='good')
    close(c,p,k)


def test_stale_event_before_promotion(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path);g.stage('fuse','sum',['first','second'],family='combined')
    r=k.read(rid,a);k.store(a,scope='global',key=r.key,topic=r.topic,values=r.values,source='doubt',status='UNCERTAIN',expected=rid)
    with pytest.raises(ContractError):g.promote('fuse',holdout(),oracle_id='heldout')
    close(c,p,k)


def test_parent_quarantine_propagates(tmp_path):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path);g.stage('fuse','sum',['first','second'],family='combined');assert g.promote('fuse',holdout(),oracle_id='heldout')
    p.quarantine(['z-first'],reason='test-safety')
    with pytest.raises(ContractError):p.compute('sum',(4,))
    assert c.manifest('sum')['state']=='QUARANTINED'
    close(c,p,k)


def test_insufficient_evidence_and_interrupted_build(tmp_path,monkeypatch):
    c,p,k,a,l,h,e,rid,w,g=prepare(tmp_path)
    run(w,e,'duplicate',x=4)
    with pytest.raises(ContractError):g.stage('fuse','sum',['first','duplicate'],family='combined')
    original=c.compose_affine
    def fail(*args,**kwargs):raise RuntimeError('interrupted')
    monkeypatch.setattr(c,'compose_affine',fail)
    with pytest.raises(RuntimeError):g.stage('fuse','sum',['first','second'],family='combined')
    assert g.job('fuse')['report'] is None
    monkeypatch.setattr(c,'compose_affine',original)
    assert g.stage('fuse','sum',['first','second'],family='combined')['state']=='STAGED'
    close(c,p,k)
