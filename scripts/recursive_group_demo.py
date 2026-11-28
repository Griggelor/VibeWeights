"""Reproducible two-generation loop, using the same controlled numeric fixtures."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_recursive_group_v08 import ready, next_run, add_head
from test_knowledge_learning_v08 import close, fixture
from vibeweights.group_learning import GroupLearning
from vibeweights.group_work import GroupWork
from vibeweights.learning_events import LearningEvents
from vibeweights.perception import PerceptionHeads
from vibeweights.task_learning import TaskLearning


def demo(root):
    c,p,k,a,l,h,w,g,event,head=ready(root)
    initial={m['id']:m['sha256'] for m in c.list()}
    delivered=h.tick(delivery_budget=32)
    groups=h.bus.groups(event)
    runs=[next_run(w,event,'next-first',2),next_run(w,event,'next-second',5)]
    m=g.stage('second-gen','grandstudent',['next-first','next-second'],family='combined-next',method='TRAINED')
    assert g.promote('second-gen',[((-3,),(-53,)),((0,),(19,)),((11,),(283,))],oracle_id='second-proof')
    second_head=add_head(h,'grandstudent','third')
    new_event=h.bus.publish_group_knowledge('second-gen')
    preserved=all(c.manifest(i)['sha256']==sha for i,sha in initial.items())
    data={'first_job':g.job('first-gen'),'second_job':g.job('second-gen'),
          'first_head':head,'second_head':second_head,'first_event':h.bus.event(event),
          'second_event':h.bus.event(new_event),'deliveries':delivered,'groups':groups,
          'runs':runs,'prior_artifacts_unchanged':preserved,'new_artifact':m}
    close(c,p,k)
    c,p,k,a,l=fixture(root);h=PerceptionHeads(LearningEvents(TaskLearning(l)))
    h.tick(delivery_budget=64)
    data['second_event_groups_after_restart']=h.bus.groups(new_event)
    data['additional_inputs_after_restart']=[{'input':x,'actual':p.compute('grandstudent',(x,))[0],'expected':24*x+19} for x in range(-20,21) if x not in (2,5,-3,0,11)]
    assert all(r['actual']==r['expected'] for r in data['additional_inputs_after_restart'])
    close(c,p,k)
    return data


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:
        data=demo(Path(tmp))
    print(json.dumps(data,indent=2,sort_keys=True,allow_nan=False))
