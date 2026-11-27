"""Reproducible, synthetic knowledge-to-weights demonstration. No network/model server."""
import json
from pathlib import Path
import random
import tempfile
from vibeweights.factory import Catalog
from vibeweights.knowledge import Access, KnowledgeLedger
from vibeweights.knowledge_learning import Domain, KnowledgeLearning
from vibeweights.population import Population


def main():
    domain=Domain('demo-catalog',12,5)
    rng=random.Random(67)
    truth={i:rng.randrange(domain.labels) for i in (0,1,3,5,7,9)}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); c=Catalog(root/'experts'); p=Population(c,tensor_budget=65536)
        k=KnowledgeLedger(root/'knowledge.sqlite'); a=Access('host'); learner=KnowledgeLearning(p,k,a)
        ids=[k.store(a,scope='global',key=f'{domain.topic}:entity:{i}',topic=domain.topic,
                     values=(v,),source='synthetic-teacher',status='VERIFIED').id for i,v in truth.items()]
        results={'scope':'synthetic closed-domain recall; no generalization claim',
                 'entities':domain.entities,'classes':domain.labels,'facts':len(ids),
                 'baseline_resident':list(p.resident_ids()),'methods':{}}
        for method in ('DIRECT','TRAINED'):
            job=method.lower(); manifest=learner.prepare(job,job,domain,ids,method=method)
            assert learner.promote(job,truth.get,oracle_id='independent-fixture')
            record=learner.job(job)
            results['methods'][method]={'build':record['build'],'evaluation':record['report'],
                'parameters':manifest['parameters'],'sha256':manifest['sha256']}
        p.close(); c.close(); k.close(); (root/'knowledge.sqlite').unlink()
        c=Catalog(root/'experts'); p=Population(c,tensor_budget=65536)
        results['resident_after_restart']=list(p.resident_ids())
        results['knowledge_db_present_after_restart']=(root/'knowledge.sqlite').exists()
        for method,row in results['methods'].items():
            row['restart_correct']=sum(domain.decode(p.compute(method.lower(),domain.encode(i)))==truth.get(i) for i in range(-1,13))
            row['restart_cases']=14
            row['known_recall_without_candidate_zero_output']=sum(domain.decode((0.0,)*domain.labels)==v for v in truth.values())
        p.close(); c.close()
        print(json.dumps(results,indent=2,sort_keys=True))

if __name__=='__main__': main()
