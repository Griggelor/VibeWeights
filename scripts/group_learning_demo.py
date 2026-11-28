"""Reproducible CPU experiment: trained participation -> cooperation -> DIRECT fusion.

Run from the repository with PYTHONPATH=src. All data are synthetic.
"""
from itertools import product
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from vibeweights.factory import Catalog
from vibeweights.population import Population
from vibeweights.knowledge import KnowledgeLedger, Access
from vibeweights.knowledge_learning import KnowledgeLearning, Domain
from vibeweights.task_learning import TaskLearning
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.group_work import GroupWork
from vibeweights.group_learning import GroupLearning


def main():
    with TemporaryDirectory() as root:
        c=Catalog(Path(root)/'experts');p=Population(c,tensor_budget=65536)
        k=KnowledgeLedger(Path(root)/'knowledge.sqlite');a=Access('host');l=KnowledgeLearning(p,k,a)
        try:
            d=Domain('notice',4,3)
            rid=k.store(a,scope='global',key='notice:entity:0',topic='notice',values=(1,),source='fixture',status='VERIFIED').id
            l.prepare('notice-job','notice',d,[rid],method='DIRECT');assert l.promote('notice-job',{0:1}.get,oracle_id='notice-proof')
            for eid,w,b,src,dst in [('z-first',2,1,'input','mid'),('a-second',3,2,'mid','answer')]:
                c.stage_affine(eid,eid,[[w]],[b],source_topic=src,target_scope='global',target_topic=dst)
                assert p.admit(eid,[((0,),(b,)),((2,),(2*w+b,))])
            bus=LearningEvents(TaskLearning(l));h=PerceptionHeads(bus)
            def samples(affected):
                return [((kind,repair,same,affected,local,rep),(kind,)) for kind,repair,same,local,rep in product((0,1),repeat=5)]
            for eid in ('z-first','a-second'):
                h.stage(eid+'-head',eid,['research'],samples(0),steps=512)
                assert h.admit(eid+'-head',samples(1),oracle_id='head-holdout')['accepted']
            event=bus.publish_knowledge('notice-job');h.tick();work=GroupWork(h);learn=GroupLearning(work)
            runs=[]
            for x in (4,7):
                runs.append(work.execute(f'run-{x}',event,'explore','research',{'input':(x,)},'answer',
                    {'mid':lambda snap,v:v==(2*x+1,), 'answer':lambda snap,v:v==(6*x+5,)},oracle_id='math'))
            hashes={m['id']:m['sha256'] for m in c.list()};before=p.tensor_bytes
            start=perf_counter_ns();m=learn.stage('fuse','sum',[r['run'] for r in runs],family='combined');build_ns=perf_counter_ns()-start
            assert m['sha256'] not in hashes.values()
            holdout=[((-3,),(-13,)),((0,),(5,)),((11,),(71,))]
            start=perf_counter_ns();assert learn.promote('fuse',holdout,oracle_id='math-heldout');admit_ns=perf_counter_ns()-start
            timings={'two_parent_compute':[],'sum_compute':[]}
            # Equal outer source guard per query; no full Environment execution
            # in either timing. Alternate order to reduce systematic bias.
            for i in range(-50,50):
                for name in (list(timings) if i%2 else list(reversed(timings))):
                    start=perf_counter_ns()
                    with l._guard():
                        y=p.compute('sum',(i,)) if name=='sum_compute' else p.compute('a-second',p.compute('z-first',(i,)))
                    timings[name].append((perf_counter_ns()-start)/1000)
                    assert y==(6*i+5,)
            print(json.dumps({'contract':'group-direct-v1','event':event,'group':bus.groups(event),'runs':runs,
                'job':learn.job('fuse'),'new_artifact':m,'parent_parameters':sum(c.manifest(i)['parameters'] for i in m['parents']),
                'candidate_parameters':m['parameters'],'population_core_tensor_bytes_before':before,'population_core_tensor_bytes_after':p.tensor_bytes,
                'original_core_hashes_unchanged':all(c.manifest(i)['sha256']==v for i,v in hashes.items()),
                'build_ms':build_ns/1e6,'admission_ms':admit_ns/1e6,
                'benchmark':{'queries_per_path':100,'correct_per_path':100,'median_us':{name:median(v) for name,v in timings.items()},
                             'scope':'same outer source guard; resident compute only; no group scheduler, head inference or GPU timing'}},indent=2))
        finally:
            p.close();c.close();k.close()


if __name__=='__main__': main()
