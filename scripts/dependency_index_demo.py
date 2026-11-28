"""Reproducible CPU guard benchmark; no model/GPU or end-to-end inference claim."""
import json, statistics, tempfile, time
from pathlib import Path
from vibeweights.factory import Catalog
from vibeweights.population import Population
from vibeweights.knowledge import KnowledgeLedger, Access
from vibeweights.knowledge_learning import KnowledgeLearning, Domain


def run(n=100, repeats=100):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); c=Catalog(root/'experts'); p=Population(c,tensor_budget=1024*1024)
        k=KnowledgeLedger(root/'knowledge.sqlite'); a=Access('host'); l=KnowledgeLearning(p,k,a)
        try:
            ids=[]
            for i in range(n):
                d=Domain(f'topic{i}',4,3)
                r=k.store(a,scope='global',key=d.topic+':entity:0',topic=d.topic,values=(1,),source='fixture',status='VERIFIED')
                ids.append(r.id); l.prepare(f'job{i}',f'expert{i}',d,[r.id],method='DIRECT')
                assert l.promote(f'job{i}',{0:1}.get,oracle_id='fixture')
            for i in range(3):
                e=f'child{i}'; parent='expert0' if i==0 else f'child{i-1}'
                c.stage_affine(e,e,[[1]],[0],source_topic='x',target_topic='y',target_scope='global',parents=(parent,))
                assert p.admit(e,[((1,),(1,))])
            l.audit_sources(); l.audit_sources()
            results={}
            # Alternate modes to reduce temporal measurement bias.
            samples={False:[],True:[]}; checks={False:[],True:[]}
            for i in range(repeats):
                for full in ((False,True) if i%2==0 else (True,False)):
                    start=time.perf_counter_ns(); assert l.audit_sources(full=full)==(); elapsed=time.perf_counter_ns()-start
                    samples[full].append(elapsed/1000); checks[full].append(l.dependencies.last_audit['source_checks'])
            for full in (False,True):
                results['full' if full else 'incremental']={'median_us':statistics.median(samples[full]),'samples_us':samples[full], 'source_checks':sorted(set(checks[full]))}
            before={m['id']:m['sha256'] for m in c.list()}
            r=k.store(a,scope='global',key='topic0:entity:0',topic='topic0',values=(2,),source='corrected',status='VERIFIED',expected=ids[0])
            start=time.perf_counter_ns(); affected=l.audit_sources(); elapsed=time.perf_counter_ns()-start
            assert affected==('child0','child1','child2','expert0')
            correction={'elapsed_us':elapsed/1000,'affected':affected,'audit':dict(l.dependencies.last_audit),'remaining_resident':len(p.resident_ids())}
            assert all(c.manifest(e)['sha256']==sha for e,sha in before.items())
            assert l.audit_sources(full=True)==()
            return {'learning_jobs':n,'additional_descendants':3,'repeats_per_mode':repeats,'guard_only':results,'single_correction':correction,'parent_weights_unchanged':True,'full_audit_after_incremental_matches':True}
        finally:
            p.close(); c.close(); k.close()


if __name__=='__main__': print(json.dumps(run(),indent=2))
