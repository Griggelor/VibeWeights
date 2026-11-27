"""CPU microbenchmark on verified synthetic facts, plus ten sequential learning packages.
Not a GPU estimate or a claim of superiority over indexed retrieval.
"""
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from vibeweights.factory import Catalog
from vibeweights.population import Population
from vibeweights.knowledge import Access, KnowledgeLedger
from vibeweights.knowledge_learning import Domain, KnowledgeLearning
from vibeweights.relation_learning import derive_relation


def run(root, *, packages=10, repeats=100):
    root=Path(root); c=Catalog(root/'experts'); p=Population(c,tensor_budget=65536)
    k=KnowledgeLedger(root/'knowledge.sqlite'); a=Access('host'); l=KnowledgeLearning(p,k,a)
    d1,d2,d3=Domain('object-material',6,4),Domain('material-class',4,3),Domain('object-class',6,3)
    t1,t2={0:2,1:1,3:0,4:3},{0:1,2:0,3:2}
    truth={0:0,3:1,4:2}  # independent fixture oracle
    def learn(job,d,values,method):
        start=perf_counter_ns()
        ids=[k.store(a,scope='global',key=f'{d.topic}:entity:{i}',topic=d.topic,values=(v,),source='fixture',status='VERIFIED').id for i,v in values.items()]
        l.prepare(job,job,d,ids,method=method)
        assert l.promote(job,values.get,oracle_id='fixture-oracle')
        return perf_counter_ns()-start
    parent_build=[learn('first',d1,t1,'TRAINED'),learn('second',d2,t2,'DIRECT')]
    start=perf_counter_ns()
    m,ids=derive_relation(l,first_job='first',second_job='second',result_domain=d3,job_id='joined',expert_id='joined',bridge_id='material-id-v1',oracle=truth.get,oracle_id='join-fixture',method='TRAINED')
    assert l.promote('joined',truth.get,oracle_id='join-fixture')
    aggregate_ns=perf_counter_ns()-start
    def ledger_only(i):
        row=k.db.execute('SELECT id FROM knowledge_heads_v1 WHERE scope=? AND key=?',('global',f'{d3.topic}:entity:{i}')).fetchone()
        if row is None: return None
        r=k.read(row[0],a)
        if r.status!='VERIFIED': return None
        # Same fixed fixture; check current parents as well, including transitive source revisions.
        todo=list(r.parents); seen=set()
        while todo:
            rid=todo.pop()
            if rid in seen: continue
            seen.add(rid); parent=k.read(rid,a)
            head=k.db.execute('SELECT id FROM knowledge_heads_v1 WHERE scope=? AND key=?',(parent.scope,parent.key)).fetchone()
            if head!=(rid,) or parent.status!='VERIFIED': return None
            todo.extend(parent.parents)
        return r.values[0]
    def cooperation(i):
        # Single live guard for both hops. No answer lookup during forward pass.
        with l._guard():
            mid=d1.decode(p.compute('first',d1.encode(i)))
            return None if mid is None else d2.decode(p.compute('second',d2.encode(mid)))
    def aggregate(i): return d3.decode(p.compute('joined',d3.encode(i)))
    queries=list(range(-1,d3.entities+1)); paths={'ledger_only':ledger_only,'two_hop_live':cooperation,'aggregate_live':aggregate}
    timings={}
    for name,fn in paths.items():
        assert [fn(i) for i in queries]==[truth.get(i) for i in queries]
        for i in queries: fn(i)
        samples=[]
        for n in range(repeats):
            i=queries[n%len(queries)];start=perf_counter_ns(); v=fn(i);samples.append(perf_counter_ns()-start)
            assert v==truth.get(i)
        timings[name]={'queries':repeats,'median_us':median(samples)/1000,'min_us':min(samples)/1000,'correct':repeats}
    manifests=c.list(); parent_params=sum(c.manifest(i)['parameters'] for i in ('first','second'))
    original_hashes={x['id']:x['sha256'] for x in manifests}
    # Sequential disjoint packages, both methods, full replay at every step.
    retained=[]; domains=[]
    for n in range(packages):
        d=Domain(f'package-{n}',8,4); expected={i:(i+n)%4 for i in range(5)}; job=f'package-{n}'
        build_ns=learn(job,d,expected,'DIRECT' if n%2==0 else 'TRAINED');domains.append((job,d,expected))
        correct=total=0
        for name,domain,values in domains:
            for i in range(-1,domain.entities+1):
                total+=1;correct+=domain.decode(p.compute(name,domain.encode(i)))==values.get(i)
        assert correct==total
        retained.append({'packages':n+1,'correct':correct,'cases':total,'build_and_admission_ms':build_ns/1e6,'resident_tensor_bytes':p.tensor_bytes})
    assert all(c.manifest(i)['sha256']==h for i,h in original_hashes.items())
    p.close(); c.close(); k.close()
    c=Catalog(root/'experts'); p=Population(c,tensor_budget=65536); k=KnowledgeLedger(root/'knowledge.sqlite');l=KnowledgeLearning(p,k,a)
    restart=sum(domain.decode(p.compute(name,domain.encode(i)))==values.get(i) for name,domain,values in domains for i in range(-1,domain.entities+1))
    assert restart==retained[-1]['cases']
    result={'scope':'synthetic CPU single-process microbenchmark; no GPU or generalization',
        'timings':timings,'parent_build_and_admission_ms':[x/1e6 for x in parent_build],
        'aggregate_derivation_build_and_admission_ms':aggregate_ns/1e6,
        'parent_parameters':parent_params,'aggregate_parameters':m['parameters'],
        'population_parameters_after_consolidation':parent_params+m['parameters'],
        'aggregate_artifact_bytes':m['bytes'],'retention':retained,'restart_correct':restart,
        'restart_cases':retained[-1]['cases'],'originals_unchanged':True,
        'knowledge_db_bytes':(root/'knowledge.sqlite').stat().st_size,'catalog_db_bytes':(root/'experts'/'catalog.sqlite').stat().st_size,
        'all_artifact_bytes':sum(x.stat().st_size for x in (root/'experts').glob('*.safetensors'))}
    p.close(); c.close(); k.close();return result


if __name__=='__main__':
    with TemporaryDirectory() as tmp: print(json.dumps(run(tmp),indent=2,sort_keys=True))
