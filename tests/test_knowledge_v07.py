import json,sys,subprocess
from pathlib import Path
import pytest
from vibeweights.abi import ContractError
from vibeweights.knowledge import KnowledgeLedger,Access
from vibeweights.context import WorkingContext,Action,Op,ContextLimits
from vibeweights.environment import Environment,Signal,Limits
from vibeweights.memory_demo import demo,RecallExpert

ROOT=Path(__file__).resolve().parents[1]
A=Access('expert.a',('group:one',))
B=Access('expert.b',('group:one',))
C=Access('expert.c')

def put(l,key,values=(1,),**kw):
    return l.store(A,**dict(dict(scope='global',key=key,topic='math',values=values,source='fixture',status='VERIFIED'),**kw))

def test_versions_and_restart(tmp_path):
    path=tmp_path/'knowledge.sqlite'
    with KnowledgeLedger(path) as l:
        a=put(l,'one',(2,)); b=put(l,'one',(3,),expected=a.id,parents=(a.id,))
        with pytest.raises(ContractError): put(l,'one',(99,),expected=a.id)
        with pytest.raises(ContractError): put(l,'one',(99,))
    with KnowledgeLedger(path) as l:
        assert l.search(A,topic='math')==(b.id,)
        assert l.read(a.id,A).values==(2,)
        r=l.read(b.id,A); assert (r.version,r.previous,r.parents,r.values)==(2,a.id,(a.id,),(3,))

def test_concurrent_stale_writer(tmp_path):
    path=tmp_path/'k.sqlite'
    with KnowledgeLedger(path) as l, KnowledgeLedger(path) as m:
        a=put(l,'x'); b=put(m,'x',(4,),expected=a.id)
        with pytest.raises(ContractError): put(l,'x',(9,),expected=a.id)
        assert l.search(A)==(b.id,)

def test_scope_and_relation_boundaries(tmp_path):
    with KnowledgeLedger(tmp_path/'k') as l:
        g=put(l,'g'); p=put(l,'p',scope='private:expert.a'); t=put(l,'t',scope='group:one')
        assert set(l.search(A))=={g.id,p.id,t.id}
        assert set(l.search(B))=={g.id,t.id}
        assert l.search(C)==(g.id,)
        for actor,rid in [(B,p.id),(C,t.id)]:
            with pytest.raises(ContractError): l.read(rid,actor)
        with pytest.raises(ContractError): put(l,'leak',parents=(p.id,))
        with pytest.raises(ContractError): l.link(A,g.id,'derived',p.id)
        l.link(A,p.id,'supports',g.id); l.link(A,t.id,'supports',g.id)
        assert l.related(A,p.id,'supports')==(g.id,)
        with pytest.raises(ContractError): l.related(B,p.id,'supports')
        assert l.related(B,t.id,'supports')==(g.id,)

def test_kind_source_and_unverified(tmp_path):
    with KnowledgeLedger(tmp_path/'k') as l:
        a=put(l,'a',kind='episodic',source='episode.1'); b=put(l,'b',status='HYPOTHESIS'); c=put(l,'c',source='episode.2')
        assert l.search(A,kind='episodic')==(a.id,)
        assert l.search(A,source='episode.2')==(c.id,)
        assert b.id not in l.search(A)
        assert l.search(A,status='HYPOTHESIS')==(b.id,)
        w=WorkingContext(l,{A.expert:A})
        with pytest.raises(ContractError): w.apply(A.expert,Action(Op.READ,record=b.id))
        assert not w.view(A.expert).records

def test_vector_retrieval_and_versioned_spaces(tmp_path):
    with KnowledgeLedger(tmp_path/'k') as l:
        a=put(l,'a',space='embedding.v1',embedding=(1,0,0)); b=put(l,'b',space='embedding.v1',embedding=(0,1,0)); c=put(l,'c',space='embedding.v1',embedding=(2,0,0))
        put(l,'d',space='embedding.v2',embedding=(1,0)); put(l,'p',scope='private:expert.a',space='embedding.v1',embedding=(1,0,0))
        assert l.search(B,space='embedding.v1',query=(100,0,0),limit=3)==(a.id,c.id,b.id)
        with pytest.raises(ContractError): l.search(B,space='embedding.v1',query=(1,0))
        with pytest.raises(ContractError): put(l,'bad',space='embedding.v1',embedding=(1,0))
        with pytest.raises(ContractError): l.search(B,space='embedding.v1',query=(1,0,0),scan_limit=2)
        with pytest.raises(ContractError): l.search(B,space='embedding.v1',query=(0,0,0))

def test_working_context_evict_and_total_budget(tmp_path):
    with KnowledgeLedger(tmp_path/'k') as l:
        a=put(l,'a',(1,2)); b=put(l,'b',(3,4));
        w=WorkingContext(l,{A.expert:A,B.expert:B},ContextLimits(records=2,components=4,total_components=2,total_records=1,actions=6))
        assert w.apply(A.expert,Action(Op.READ,record=a.id))
        with pytest.raises(ContractError): w.apply(B.expert,Action(Op.READ,record=b.id))
        assert not w.view(B.expert).records
        assert w.apply(A.expert,Action(Op.EVICT,record=a.id))
        assert l.read(a.id,A).values==(1,2)
        assert w.apply(B.expert,Action(Op.READ,record=b.id))
        assert w.apply(B.expert,Action(Op.EVICT,record=b.id))
        assert w.apply(A.expert,Action(Op.READ,record=a.id))
        with pytest.raises(ContractError): w.apply(A.expert,Action(Op.EVICT,record=a.id))
        assert w.actions==6

def test_native_memory_demo_recall_restart_and_eviction(tmp_path):
    assert demo(tmp_path,7)['status']!='OK'
    r=demo(tmp_path,7,initialize=True)
    assert r['status']=='OK' and r['value']==[13]
    assert [a[1] for a in r['memory_trace']]==[Op.SEARCH,Op.READ,Op.EVICT]
    assert r['memory_trace'][-1][-1]==()
    r2=demo(tmp_path,-7)
    assert r2['value']==[-1] and r2['status']=='OK'
    with KnowledgeLedger(tmp_path/'Data'/'knowledge_ledger.sqlite') as l:
        assert len(l.search(Access('demo.recall@1')))==2

def test_weight_mutation_breaks_recall(tmp_path):
    class Add:
        def compute(self,values): return (sum(values),)
    # The arithmetic double is only for isolating the action-head dependency.
    e=RecallExpert(Add()); e.head.zero_(); e.head[2,:]=1
    with KnowledgeLedger(tmp_path/'k') as l:
        put(l,'offset',(6,),topic='demo.offset')
        r=Environment([e],{('global','result'):lambda s,v:v==(13,)}).run([Signal('global','input',(7,))],('global','result'),knowledge=l)
        assert r['status']!='OK' and r['memory_actions']==0

def test_no_context_leak_between_experts(tmp_path):
    class Reader:
        def __init__(self,id,rid): self.id=id; self.rid=rid
        def relevant(self,s): return True
        def memory_action(self,s,view): return None if view.records else Action(Op.READ,record=self.rid)
        def forward_with_memory(self,s,view): return [Signal('global','result',view.records[0].values)]
    with KnowledgeLedger(tmp_path/'k') as l:
        private=put(l,'private',(5,),scope='private:expert.a')
        e=Reader(B.expert,private.id)
        r=Environment([e],{('global','result'):lambda s,v:True}).run([],('global','result'),knowledge=l,access={B.expert:B})
        assert r['status']!='OK' and r['faults'][0][1]=='memory'

def test_consolidation_keeps_originals(tmp_path):
    with KnowledgeLedger(tmp_path/'k') as l:
        a=put(l,'a',(1,)); b=put(l,'b',(2,))
        c=put(l,'summary',(1,2),kind='procedural',parents=(a.id,b.id),status='HYPOTHESIS')
        assert c.id not in l.search(A)
        assert l.read(c.id,A).parents==(a.id,b.id)
        assert l.read(a.id,A).values==(1,) and l.read(b.id,A).values==(2,)

def test_temporal_index(tmp_path,monkeypatch):
    ticks=iter((100,200))
    monkeypatch.setattr("vibeweights.knowledge.time.time_ns",lambda:next(ticks))
    with KnowledgeLedger(tmp_path/'k') as l:
        a=put(l,'a'); b=put(l,'b')
        assert l.search(A,since_ns=b.created_ns)==(b.id,)
        assert l.search(A,until_ns=a.created_ns)==(a.id,)
        with pytest.raises(ContractError): l.search(A,since_ns=5,until_ns=3)

def test_verified_memory_commit_with_provenance(tmp_path):
    r=demo(tmp_path,7,initialize=True,remember=True)
    assert r['status']=='OK' and r['value']==[13]
    assert [a[1] for a in r['memory_trace']]==[Op.SEARCH,Op.READ,Op.REMEMBER,Op.EVICT]
    with KnowledgeLedger(tmp_path/'Data'/'knowledge_ledger.sqlite') as l:
        acc=Access('demo.recall@1')
        saved=l.read(r['remembered']['id'],acc)
        assert saved.values==(13,) and saved.source=='run:'+r['run_id'] and saved.status=='VERIFIED'
        assert len(saved.parents)==1 and l.read(saved.parents[0],acc).values==(6,)
        assert l.search(acc,topic='result')==(saved.id,)
    from vibeweights.environment import EventLedger
    events=EventLedger(tmp_path/'Data'/'environment.sqlite')
    assert events.read(r['run_id'])['remembered']['id']==saved.id
    events.close()

def test_remember_denied_without_grant_and_on_unverified_goal(tmp_path):
    class WantsSave:
        id='expert.a'
        def relevant(self,s): return True
        def memory_action(self,s,view): return Action(Op.REMEMBER) if not view.remember else None
        def forward_with_memory(self,s,view): return [Signal('global','result',(1,))]
    with KnowledgeLedger(tmp_path/'k') as l:
        r=Environment([WantsSave()],{('global','result'):lambda s,v:False}).run([],('global','result'),knowledge=l,remember_experts=(A.expert,))
        assert r['status']!='OK' and not l.search(A,status=None)
        ctx=WorkingContext(l,{A.expert:A})
        with pytest.raises(ContractError): ctx.apply(A.expert,Action(Op.REMEMBER))
        assert not ctx.view(A.expert).remember

def test_scope_filtered_workspace(tmp_path):
    seen={}
    class Observer:
        def __init__(self,id): self.id=id
        def relevant(self,s): seen[self.id]=tuple(v.scope for v in s); return False
    with KnowledgeLedger(tmp_path/'k') as l:
        Environment([Observer(A.expert),Observer(B.expert),Observer(C.expert)],{}).run([Signal('global','g',(1,)),Signal('private:expert.a','p',(2,)),Signal('group:one','t',(3,))],('global','result'),knowledge=l,access={a.expert:a for a in (A,B,C)})
        assert set(seen[A.expert])=={'global','private:expert.a','group:one'}
        assert set(seen[B.expert])=={'global','group:one'}
        assert seen[C.expert]==('global',)

def test_private_loaded_source_cannot_directly_publish_globally(tmp_path):
    class Reader:
        id=A.expert
        def relevant(self,s): return True
        def memory_action(self,s,view): return None if view.records else Action(Op.READ,record=self.rid)
        def forward_with_memory(self,s,view): return [Signal('global','result',view.records[0].values)]
    with KnowledgeLedger(tmp_path/'k') as l:
        private=put(l,'p',scope='private:expert.a'); e=Reader();e.rid=private.id
        r=Environment([e],{('global','result'):lambda s,v:True}).run([],('global','result'),knowledge=l)
        assert r['status']!='OK' and r['faults'][-1][1]=='forward'

def test_bootstrap_explicit_context_and_budget(tmp_path,monkeypatch):
    from vibeweights.bootstrap import BootstrapExpert,BootstrapSpec
    e=BootstrapExpert(BootstrapSpec(A.expert,'local'))
    sent=[]
    def propose(text): sent.append(json.loads(text));return {'verified':False}
    monkeypatch.setattr(e,'propose',propose)
    with KnowledgeLedger(tmp_path/'k') as l:
        a=put(l,'a',(3,)); b=l.store(B,scope='private:expert.b',key='b',topic='math',values=(4,),source='fixture',status='VERIFIED')
        h=put(l,'h',status='HYPOTHESIS')
        r=e.propose_with_context('test',l,A,(a.id,))
        assert r['context_ids']==[a.id] and not r['verified'] and sent[0]['reference_data'][0]['values']==[3]
        for ids,limit in [((b.id,),1000),((h.id,),1000),((a.id,),1)]:
            with pytest.raises(ContractError): e.propose_with_context('test',l,A,ids,max_chars=limit)
        assert len(sent)==1

def test_remembered_fact_reused_after_restart_with_distractors(tmp_path):
    first=demo(tmp_path,7,initialize=True,remember=True)
    from vibeweights.factory import Catalog
    with KnowledgeLedger(tmp_path/'Data'/'knowledge_ledger.sqlite') as l:
        for i in range(1000): put(l,'noise'+str(i),(i,),topic='noise')
        class FollowUp:
            id='followup@1'
            def relevant(self,s): return True
            def memory_action(self,s,view):
                if view.records: return None
                if view.hits: return Action(Op.READ,record=view.hits[0])
                return Action(Op.SEARCH,topic='result',limit=1)
            def forward_with_memory(self,s,view):
                yield Signal('global','doubled',self.affine.compute(view.records[0].values))
        cat=Catalog(tmp_path/'factory')
        try:
            cat.stage_affine('double@1','double',[[2]],[0],source_topic='result',target_scope='global',target_topic='doubled')
            assert cat.admit('double@1',[((4,),(8,)),((-3,),(-6,))])
            e=FollowUp(); e.affine=cat.load('double@1')
            r=Environment([e],{('global','doubled'):lambda s,v:v==(26,)}).run([],('global','doubled'),knowledge=l)
            assert r['status']=='OK' and r['value']==[26]
            assert r['memory_trace'][1][2]==first['remembered']['id']
        finally: cat.close()
