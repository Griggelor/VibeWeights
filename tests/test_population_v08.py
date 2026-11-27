from dataclasses import replace
import pytest
from vibeweights.abi import ContractError
from vibeweights.factory import Catalog
from vibeweights.population import Population, SelectionEvidence, ACTIVE, LOW_ACTIVE_IMPORTANT, ARCHIVED


def stage(c, id, *, parents=(), src='input', dst='result', factor=2):
    return c.stage_affine(id, 'test', [[factor]], [0], source_topic=src,
                          target_scope='global', target_topic=dst, parents=parents)


def cases(factor=2):
    return [((3,), (3*factor,)), ((-2,), (-2*factor,))]


def evidence(**kw):
    return replace(SelectionEvidence('report', 10, True, False, True, True, True, True), **kw)


def test_additive_learning_and_consolidation(tmp_path):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=24)
    a=stage(c,'a',dst='mid'); assert p.admit('a',cases())
    b=stage(c,'b',src='mid',factor=3); assert p.admit('b',cases(3))
    fused=c.compose_affine('sum','sum','a','b')
    assert fused['parents']==['a','b'] and p.admit('sum',cases(6))
    assert p.resident_ids()==('a','b','sum') and p.tensor_bytes==24
    assert p.compute('sum',(7,))==p.compute('b',p.compute('a',(7,)))==(42,)
    assert all(c.manifest(i)['state']==ACTIVE for i in ('a','b','sum'))
    assert c.manifest('a')['sha256']==a['sha256'] and c.manifest('b')['sha256']==b['sha256']
    p.close(); c.close()


def test_rare_important_protected_and_archive_reactivation(tmp_path):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8)
    m=stage(c,'a'); assert p.admit('a',cases())
    assert p.select('a',LOW_ACTIVE_IMPORTANT,expected_revision=0,evidence=evidence(important=True))==1
    assert p.resident_ids()==('a',) and p.tensor_bytes==8 and p.compute('a',(5,))==(10,)
    with pytest.raises(ContractError): p.select('a',ARCHIVED,expected_revision=1,evidence=evidence(important=True))
    with pytest.raises(ContractError): p.select('a',ARCHIVED,expected_revision=1,evidence=evidence(coverage=False))
    assert p.select('a',ARCHIVED,expected_revision=1,evidence=evidence())==2
    assert p.resident_ids()==() and p.tensor_bytes==0
    assert (tmp_path/(m['sha256']+'.safetensors')).exists()
    with pytest.raises(ContractError): p.compute('a',(5,))
    with pytest.raises(ContractError): c.load('a')
    p.close(); c.close()
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8)
    assert p.resident_ids()==() and p.revision('a')==2
    assert p.select('a',LOW_ACTIVE_IMPORTANT,expected_revision=2,evidence=evidence(important=True))==3
    assert p.compute('a',(5,))==(10,) and p.tensor_bytes==8
    assert [x['new_state'] for x in p.history('a')]==[LOW_ACTIVE_IMPORTANT,ARCHIVED,LOW_ACTIVE_IMPORTANT]
    p.close(); c.close()


def test_no_eviction_and_budget_restart(tmp_path):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8)
    stage(c,'a'); assert p.admit('a',cases())
    p.select('a',LOW_ACTIVE_IMPORTANT,expected_revision=0,evidence=evidence(important=True))
    stage(c,'b',factor=3)
    with pytest.raises(ContractError): p.admit('b',cases(3))
    assert c.manifest('b')['state']=='STAGED' and p.resident_ids()==('a',)
    p.close()
    with pytest.raises(ContractError): Population(c,tensor_budget=7)
    assert c.manifest('a')['state']==LOW_ACTIVE_IMPORTANT
    c.close()


@pytest.mark.parametrize('bad', [dict(coverage=False),dict(ablation_ok=False),dict(dependencies_clear=False),dict(observations=1)])
def test_archive_requires_evidence(tmp_path,bad):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8,min_observations=2)
    stage(c,'a'); assert p.admit('a',cases())
    with pytest.raises(ContractError): p.select('a',ARCHIVED,expected_revision=0,evidence=evidence(**bad))
    assert p.resident_ids()==('a',) and p.history('a')==[]
    p.close();c.close()


def test_archive_corruption_and_stale_revision(tmp_path):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8)
    m=stage(c,'a'); assert p.admit('a',cases())
    p.select('a',ARCHIVED,expected_revision=0,evidence=evidence())
    with pytest.raises(ContractError): p.select('a',ACTIVE,expected_revision=0,evidence=evidence())
    with pytest.raises(ContractError): p.select('a',ACTIVE,expected_revision=1,evidence=evidence(beneficial=False))
    path=tmp_path/(m['sha256']+'.safetensors'); path.write_bytes(b'broken')
    with pytest.raises(ContractError): p.select('a',ACTIVE,expected_revision=1,evidence=evidence())
    assert c.manifest('a')['state']==ARCHIVED and p.tensor_bytes==0 and p.revision('a')==1
    p.close(); c.close()


def test_archive_reactivation_budget_and_failed_admission(tmp_path):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8)
    stage(c,'a'); assert not p.admit('a',cases(4)); assert p.tensor_bytes==0
    assert p.admit('a',cases()); p.select('a',ARCHIVED,expected_revision=0,evidence=evidence())
    stage(c,'b',factor=3); assert p.admit('b',cases(3))
    with pytest.raises(ContractError): p.select('a',ACTIVE,expected_revision=1,evidence=evidence())
    assert c.manifest('a')['state']==ARCHIVED and p.resident_ids()==('b',)
    p.close();c.close()


def test_direct_catalog_changes_fail_closed(tmp_path):
    c=Catalog(tmp_path); p=Population(c,tensor_budget=16)
    stage(c,'a'); assert c.admit('a',cases())
    with pytest.raises(ContractError): p.resident_ids()
    p.close();p=Population(c,tensor_budget=16)
    assert p.resident_ids()==('a',)
    p.close();c.close()


def test_environment_only_resident_experts(tmp_path):
    from vibeweights.environment import Signal
    c=Catalog(tmp_path); p=Population(c,tensor_budget=8)
    stage(c,'a'); assert p.admit('a',cases())
    p.select('a',LOW_ACTIVE_IMPORTANT,expected_revision=0,evidence=evidence(important=True))
    args=([Signal('global','input',(5,))],('global','result'),{('global','result'):lambda s,v:v==(10,)})
    assert p.run(*args)['value']==[10.0]
    p.select('a',ARCHIVED,expected_revision=1,evidence=evidence())
    assert p.run(*args)['status']=='STAGNATION'
    p.select('a',ACTIVE,expected_revision=2,evidence=evidence())
    assert p.run(*args)['value']==[10.0]
    p.close();c.close()
