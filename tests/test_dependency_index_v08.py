import sqlite3
import pytest
from vibeweights.abi import ContractError
from vibeweights.knowledge_learning import Domain
from test_knowledge_learning_v08 import fixture, facts, close


def add(l,k,a,n):
    d=Domain(f'topic{n}',3,2); ids=facts(k,a,d,{0:1})
    l.prepare(f'job{n}',f'expert{n}',d,ids,method='DIRECT')
    assert l.promote(f'job{n}',{0:1}.get,oracle_id='oracle')
    return d,ids[0]


def test_incremental_only_checks_affected_jobs(tmp_path,monkeypatch):
    c,p,k,a,l=fixture(tmp_path)
    ds=[add(l,k,a,n) for n in range(12)]
    c.stage_affine('child','child',[[1]],[0],source_topic='x',target_scope='global',target_topic='y',parents=('expert0',))
    assert p.admit('child',[((1,),(1,))]); l.audit_sources(); l.audit_sources()
    checked=[]; orig=l._check_sources
    def check(job): checked.append(job['id']); return orig(job)
    monkeypatch.setattr(l,'_check_sources',check)
    for _ in range(5): assert p.compute('expert11',ds[11][0].encode(0))==(0.,1.)
    assert checked==[] and l.dependencies.last_audit['candidates']==0
    d,rid=ds[0]
    assert l.dependencies.consumers(rid)=={'expert0'}
    assert l.dependencies.descendants({'expert0'})=={'expert0','child'}
    k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(0,),source='revision',status='VERIFIED',expected=rid)
    assert l.audit_sources()==('child','expert0') and checked==['job0']
    assert len(p.resident_ids())==11
    assert p.compute('expert11',ds[11][0].encode(0))==(0.,1.)
    close(c,p,k)


def test_restart_external_revision_and_backfill(tmp_path):
    c,p,k,a,l=fixture(tmp_path);d,rid=add(l,k,a,0);add(l,k,a,1)
    l.audit_sources();l.audit_sources(); close(c,p,k)
    # Simulate a pre-index catalog and a correction made before reopening.
    raw=sqlite3.connect(tmp_path/'experts'/'catalog.sqlite')
    with raw: raw.execute('DELETE FROM dep_sources_v08')
    raw.close()
    from vibeweights.knowledge import KnowledgeLedger
    with KnowledgeLedger(tmp_path/'knowledge.sqlite') as k:
        k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(0,),source='external',status='VERIFIED',expected=rid)
    c,p,k,a,l=fixture(tmp_path)
    assert l.dependencies.consumers(rid)=={'expert0'}
    assert l.audit_sources()==('expert0',) and l.dependencies.last_audit['full']
    assert p.resident_ids()==('expert1',)
    close(c,p,k)


def test_rollback_events_not_consumed(tmp_path):
    c,p,k,a,l=fixture(tmp_path);d,rid=add(l,k,a,0);l.audit_sources();l.audit_sources()
    old=l.dependencies.checkpoint
    k.db.execute('BEGIN IMMEDIATE')
    k._store_locked(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(0,),source='rollback',status='VERIFIED',expected=rid)
    k.db.rollback()
    assert l.audit_sources()==() and l.dependencies.checkpoint==old
    k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(0,),source='commit',status='VERIFIED',expected=rid)
    assert l.audit_sources()==('expert0',)
    close(c,p,k)


def test_new_descendant_of_quarantined_parent(tmp_path):
    c,p,k,a,l=fixture(tmp_path);d,rid=add(l,k,a,0);l.audit_sources()
    p.quarantine(['expert0'],reason='test'); l.audit_sources(); l.audit_sources()
    c.stage_affine('late','child',[[1]],[0],source_topic='x',target_scope='global',target_topic='y',parents=('expert0',))
    assert l.audit_sources()==('late',)
    with pytest.raises(ContractError): p.admit('late',[((1,),(1,))])
    close(c,p,k)


def test_external_connection_and_failed_audit_retry(tmp_path,monkeypatch):
    c,p,k,a,l=fixture(tmp_path);d,rid=add(l,k,a,0);l.audit_sources();l.audit_sources()
    from vibeweights.knowledge import KnowledgeLedger
    with KnowledgeLedger(tmp_path/'knowledge.sqlite') as other:
        other.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(0,),source='other',status='VERIFIED',expected=rid)
    old=l.dependencies.checkpoint
    def fail(*args,**kw): raise RuntimeError('interrupted')
    with monkeypatch.context() as mp:
        mp.setattr(p,'quarantine',fail)
        with pytest.raises(RuntimeError): l.audit_sources()
    assert l.dependencies.checkpoint==old
    assert l.audit_sources()==('expert0',)
    close(c,p,k)


def test_reserved_job_cannot_be_bypassed_by_plain_manifest(tmp_path,monkeypatch):
    c,p,k,a,l=fixture(tmp_path);d=Domain('topic',3,2);ids=facts(k,a,d,{0:1})
    def interrupted(*a,**kw): raise RuntimeError('interrupted')
    with monkeypatch.context() as mp:
        mp.setattr(l,'resume_build',interrupted)
        with pytest.raises(RuntimeError): l.prepare('job','expert',d,ids,method='DIRECT')
    c.stage_affine('expert','topic',[[1]],[0],source_topic='x',target_scope='global',target_topic='y')
    assert l.audit_sources()==('expert',)
    with pytest.raises(ContractError): p.admit('expert',[((1,),(1,))])
    close(c,p,k)
