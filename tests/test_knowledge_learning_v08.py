import json
import random
import shutil
import subprocess
import sys
import pytest
from vibeweights.abi import ContractError
from vibeweights.environment import Signal
from vibeweights.factory import Catalog
from vibeweights.knowledge import Access, KnowledgeLedger
from vibeweights.knowledge_learning import Domain, KnowledgeLearning
from vibeweights.population import Population, ACTIVE, ARCHIVED, SelectionEvidence


def fixture(tmp_path, budget=65536):
    c=Catalog(tmp_path/'experts'); p=Population(c,tensor_budget=budget)
    k=KnowledgeLedger(tmp_path/'knowledge.sqlite'); a=Access('host')
    return c,p,k,a,KnowledgeLearning(p,k,a)


def facts(k,a,d,truth,**kwargs):
    return [k.store(a,scope=kwargs.get('scope','global'),key=f'{d.topic}:entity:{i}',topic=d.topic,
                    values=(v,),source='fixture-source',status=kwargs.get('status','VERIFIED')).id
            for i,v in truth.items()]


def close(c,p,k):
    p.close(); c.close(); k.close()


@pytest.mark.parametrize('method',['DIRECT','TRAINED'])
def test_learning_and_isolated_restart(tmp_path,method):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',12,5)
    rng=random.Random(67); truth={i:rng.randrange(d.labels) for i in (0,1,3,5,7,9)}
    # Oracle is the pre-existing fixture, never created from candidate/ledger.
    ids=facts(k,a,d,truth)
    assert p.resident_ids()==()
    m=l.prepare('job','new-expert',d,ids,method=method)
    assert m['origin']==method and m['state']=='STAGED'
    with pytest.raises(ContractError): p.admit('new-expert',[(d.encode(0),d.target(truth[0]))])
    assert l.promote('job',truth.get,oracle_id='fixture-oracle')
    j=l.job('job'); assert j['state']==ACTIVE and j['report']['passed']==14
    assert j['build']['optimizer_steps']==(80 if method=='TRAINED' else 0)
    assert j['build']['initial_loss']>0 and j['build']['final_loss']<1e-10
    for i in range(-1,13): assert d.decode(p.compute('new-expert',d.encode(i)))==truth.get(i)
    close(c,p,k)
    # Restart with fact DB removed; no teacher, no ledger in inference.
    (tmp_path/'knowledge.sqlite').unlink()
    c=Catalog(tmp_path/'experts'); p=Population(c,tensor_budget=65536,allow_offline_knowledge=True)
    assert all(d.decode(p.compute('new-expert',d.encode(i)))==truth.get(i) for i in range(-1,13))
    # Fresh Python process receives ONLY a copied weight file and identity dimensions.
    isolated=tmp_path/'isolated'; isolated.mkdir()
    shutil.copyfile(c.root/(m['sha256']+'.safetensors'),isolated/'model.safetensors')
    code='''import json,sys,torch
from safetensors.torch import load_file
from vibeweights.factory import AffineExpert
from vibeweights.knowledge_learning import Domain
d=Domain('catalog',12,5); t=load_file('model.safetensors'); e=AffineExpert({'id':'isolated'},t['w'],t['b'])
print(json.dumps([[d.decode(e.compute(d.encode(i))),d.decode(torch.zeros(d.labels).tolist())] for i in range(-1,13)]))
'''
    result=json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=isolated,text=True))
    assert result==[[truth.get(i),None] for i in range(-1,13)]
    p.close(); c.close()


def test_wrong_verified_fact_rejected_by_independent_oracle(tmp_path):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',4,3)
    l.prepare('job','wrong',d,facts(k,a,d,{0:2}),method='DIRECT')
    assert not l.promote('job',{0:1}.get,oracle_id='oracle')
    assert l.job('job')['report']['accepted'] is False and p.resident_ids()==()
    with pytest.raises(ContractError): l.promote('job',{0:2}.get,oracle_id='changed-oracle')
    close(c,p,k)


@pytest.mark.parametrize('status,scope',[('OBSERVED','global'),('HYPOTHESIS','global'),('VERIFIED','private:host')])
def test_learning_rejects_unverified_and_private(tmp_path,status,scope):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',4,3)
    with pytest.raises(ContractError): l.prepare('job','wrong',d,facts(k,a,d,{0:2},status=status,scope=scope),method='DIRECT')
    assert c.list()==[]; close(c,p,k)


def test_undertrained_candidate_not_admitted(tmp_path):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',4,3)
    l.prepare('job','weak',d,facts(k,a,d,{0:2}),method='TRAINED',steps=1)
    assert not l.promote('job',{0:2}.get,oracle_id='oracle')
    assert l.job('job')['build']['optimizer_steps']==1 and p.resident_ids()==()
    close(c,p,k)


def test_source_revision_before_promotion(tmp_path):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',4,3); ids=facts(k,a,d,{0:2})
    l.prepare('job','old',d,ids,method='DIRECT')
    k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(1,),source='correction',status='VERIFIED',expected=ids[0])
    with pytest.raises(ContractError): l.promote('job',{0:2}.get,oracle_id='oracle')
    assert l.audit_sources()==('old',) and c.manifest('old')['state']=='QUARANTINED'
    close(c,p,k)


def test_revision_quarantines_descendants_and_keeps_unrelated(tmp_path):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',4,3); ids=facts(k,a,d,{0:2})
    m=l.prepare('job','old',d,ids,method='DIRECT'); assert l.promote('job',{0:2}.get,oracle_id='oracle')
    c.stage_affine('descendant','child',[[1]],[0],source_topic='x',target_scope='global',target_topic='y',parents=('old',)); assert p.admit('descendant',[((1,),(1,))])
    c.stage_affine('unrelated','u',[[1]],[0],source_topic='x',target_scope='global',target_topic='y'); assert p.admit('unrelated',[((1,),(1,))])
    rec=k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(1,),source='correction',status='VERIFIED',expected=ids[0])
    assert l.audit_sources()==('descendant','old') and p.resident_ids()==('unrelated',)
    assert l.audit_sources()==() and (c.root/(m['sha256']+'.safetensors')).exists()
    with pytest.raises(ContractError): p.select('old',ACTIVE,expected_revision=1,evidence=SelectionEvidence('report',10,True,True,True,True,True,True))
    l.prepare('job2','corrected',d,[rec.id],method='TRAINED'); assert l.promote('job2',{0:1}.get,oracle_id='oracle2')
    assert d.decode(p.compute('corrected',d.encode(0)))==1
    assert p.resident_ids()==('corrected','unrelated')
    close(c,p,k)


def test_source_parent_revision_invalidates_job(tmp_path):
    c,p,k,a,l=fixture(tmp_path); d=Domain('catalog',4,3)
    parent=k.store(a,scope='global',key='evidence',topic='evidence',values=(1,),source='origin',status='VERIFIED')
    root=k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(2,),source='derived',status='VERIFIED',parents=(parent.id,))
    l.prepare('job','derived',d,[root.id],method='DIRECT'); assert l.promote('job',{0:2}.get,oracle_id='oracle')
    k.store(a,scope='global',key='evidence',topic='evidence',values=(0,),source='revocation',status='HYPOTHESIS',expected=parent.id)
    assert l.audit_sources()==('derived',); close(c,p,k)


def test_budget_refusal_receipt_and_restart(tmp_path):
    c,p,k,a,l=fixture(tmp_path,budget=8); d=Domain('catalog',4,3)
    l.prepare('job','candidate',d,facts(k,a,d,{0:2}),method='DIRECT')
    with pytest.raises(ContractError): l.promote('job',{0:2}.get,oracle_id='oracle')
    assert l.job('job')['report']['accepted'] and l.job('job')['state']=='STAGED'
    close(c,p,k)
    c=Catalog(tmp_path/'experts');p=Population(c,tensor_budget=64);k=KnowledgeLedger(tmp_path/'knowledge.sqlite');l=KnowledgeLearning(p,k,a)
    with pytest.raises(ContractError): p.admit('candidate',[(d.encode(i),d.target({0:2}.get(i))) for i in range(-1,5)])
    assert l.promote('job',{0:2}.get,oracle_id='oracle') and p.resident_ids()==('candidate',)
    close(c,p,k)


def test_three_sequential_packages_and_runtime(tmp_path):
    c,p,k,a,l=fixture(tmp_path); previous=[]
    for n in range(3):
        # Same entity IDs can occur in independently namespaced domains.
        d=Domain(f'package{n}',12,4); truth={i:(i+n)%4 for i in range(4)}
        ident=f'expert{n}'
        l.prepare(f'job{n}',ident,d,facts(k,a,d,truth),method=('TRAINED' if n%2 else 'DIRECT'))
        assert l.promote(f'job{n}',truth.get,oracle_id=f'oracle{n}')
        previous.append((ident,d,truth))
        for id,domain,expected in previous:
            assert all(domain.decode(p.compute(id,domain.encode(i)))==expected.get(i) for i in range(12))
    id,d,truth=previous[0]
    goal=('global',d.topic+':answer'); result=p.run([Signal('global',d.topic+':query',d.encode(0))],goal,{goal:lambda s,v:d.decode(v)==truth[0]})
    assert result['status']=='OK' and d.decode(result['value'])==truth[0]
    assert len(p.resident_ids())==3 and l.audit_sources()==()
    close(c,p,k)
