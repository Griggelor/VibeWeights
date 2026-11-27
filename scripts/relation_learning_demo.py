"""Synthetic two-hop transfer, additive admission and source invalidation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from vibeweights.factory import Catalog
from vibeweights.population import Population
from vibeweights.knowledge import Access,KnowledgeLedger
from vibeweights.knowledge_learning import Domain,KnowledgeLearning
from vibeweights.relation_learning import derive_relation
from vibeweights.abi import ContractError


def main():
    with TemporaryDirectory() as tmp:
        root=Path(tmp); c=Catalog(root/'experts');p=Population(c,tensor_budget=65536)
        k=KnowledgeLedger(root/'knowledge.sqlite');a=Access('host');l=KnowledgeLearning(p,k,a)
        d1,d2,d3=Domain('object-material',6,4),Domain('material-class',4,3),Domain('object-class',6,3)
        t1,t2,t3={0:2,1:1,3:0,4:3},{0:1,2:0,3:2},{0:0,3:1,4:2}
        def learn(job,d,truth,method):
            ids=[k.store(a,scope='global',key=f'{d.topic}:entity:{i}',topic=d.topic,values=(v,),source='fixture',status='VERIFIED').id for i,v in truth.items()]
            l.prepare(job,job,d,ids,method=method);assert l.promote(job,truth.get,oracle_id='fixture-oracle');return ids
        ar=learn('first',d1,t1,'TRAINED');learn('second',d2,t2,'DIRECT')
        before=list(p.resident_ids());assert not k.search(a,topic=d3.topic)
        m,ids=derive_relation(l,first_job='first',second_job='second',result_domain=d3,job_id='joined',expert_id='joined',bridge_id='material-id-v1',oracle=t3.get,oracle_id='independent-join-fixture',method='TRAINED')
        assert l.promote('joined',t3.get,oracle_id='independent-join-fixture')
        result={'scope':'synthetic, declared two-hop identity join; no autonomous semantic discovery',
                'resident_before':before,'resident_after':list(p.resident_ids()),'derived_facts':len(ids),
                'derived_provenance':[list(k.read(i,a).parents) for i in ids],
                'manifest_parents':m['parents'],'parameters':m['parameters'],'learning':l.job('joined')['build'],
                'evaluation':l.job('joined')['report'],
                'known_without_weights':sum(d3.decode((0.0,)*d3.labels)==v for v in t3.values())}
        k.store(a,scope='global',key=d1.topic+':entity:0',topic=d1.topic,values=(3,),source='correction',status='VERIFIED',expected=ar[0])
        try: p.compute('joined',d3.encode(0));blocked=False
        except ContractError: blocked=True
        result.update(blocked_after_revision=blocked,states_after_revision={m['id']:m['state'] for m in c.list()},resident_after_revision=list(p.resident_ids()))
        p.close();c.close();k.close();print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
