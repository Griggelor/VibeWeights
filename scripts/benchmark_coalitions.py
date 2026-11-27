"""CPU experiment: broad discovery overhead, NOT 512 competent experts.
No external calls. Synthetic participants are permanently inactive placeholders.
"""
from pathlib import Path
import sys,json,time,statistics,platform
from dataclasses import replace
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
import torch
from vibeweights.application import Application,BaselineApplication
from vibeweights.coalition import CoalitionEngine,Participant,Gate,construct_gate
from vibeweights.coalition_arithmetic import FEATURES
from vibeweights.platform import DenseExpert

def main():
    torch.set_num_threads(1)
    results=[]; text='Ist 7 mal 6 mindestens 30?'
    with Application(root) as app, BaselineApplication(root) as baseline:
        req=app.encoder.encode_task(text); goal=app.runtime.goal(req); base=baseline.ask(text)
        active=list(app.runtime.engine.participants); template=active[0]
        for count in (5,8,32,128,512):
            extras=[]
            for i in range(count-len(active)):
                core=DenseExpert(replace(template.expert.spec,id=f'placeholder.e{i}'),template.expert.state_dict())
                def inactive():
                    # Contradiction via constant-zero output. Not a competency artifact.
                    t=construct_gate(len(FEATURES),(((0,1),),)); t['out'].zero_(); return Gate(t)
                extras.append(Participant(core,inactive(),inactive(),template.features,template.apply,template.allowed))
            for cache in (False,True):
                engine=CoalitionEngine(app.runtime.registry.schema,active+extras,app.cooperation,cache_heads=cache)
                timings=[]
                for i in range(5):
                    start=time.perf_counter(); r=engine.run_groups((goal,),recall=False)[0]; elapsed=time.perf_counter()-start
                    assert r.verified and r.frame.get('PRODUCT')==42 and len(r.members)==4
                    timings.append(elapsed*1000)
                results.append({'cache_heads':cache,'inventory':count,'real_participants':5,'inactive_placeholders':count-5,'members':len(r.members),'steps':r.steps,'parameter_visits':r.parameter_visits,'head_visits':r.head_visits,'events':len(r.events),'median_ms':round(statistics.median(timings[1:]),3),'samples_ms':[round(t,3) for t in timings]})
        data={'scope':'synthetic CPU broad discovery overhead; not GPU nor 100M-expert throughput','python':platform.python_version(),'torch':str(torch.__version__),'threads':torch.get_num_threads(),'baseline_v03':{'steps':base['steps'],'parameter_visits':base['parameter_visits']},'results':results}
    print(json.dumps(data,indent=2)); return data
if __name__=='__main__': main()
