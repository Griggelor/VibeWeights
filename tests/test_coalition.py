import itertools,random,json,subprocess,sys
from pathlib import Path
import pytest,torch
from vibeweights.application import Application
from vibeweights.coalition import Op,Gate,construct_gate,CoalitionEngine,Participant,Cooperation
from vibeweights.coalition_arithmetic import build,HEADS,FEATURES
from vibeweights.platform import Schema,Field,ExpertSpec,DenseExpert,Delta
ROOT=Path(__file__).resolve().parents[1]

def kinds(r,kind): return [e for e in r['events'] if e['kind']==kind]

def test_head_truth_tables():
    # independent Boolean oracle, exhaustive on all 12 feature bits
    for name,heads in HEADS.items():
        for clauses in heads:
            gate=Gate(construct_gate(len(FEATURES),clauses))
            for values in itertools.product((0,1),repeat=len(FEATURES)):
                expected=any(all(values[i]==v for i,v in c) for c in clauses)
                assert gate(torch.tensor(values,dtype=torch.float32))==expected

def test_head_reproduction(tmp_path):
    build(tmp_path)
    for p in (ROOT/'artifacts/coalition').iterdir(): assert p.read_bytes()==(tmp_path/'artifacts/coalition'/p.name).read_bytes()

def test_default_path_no_plan(monkeypatch):
    from vibeweights.platform import TaskRouter,Engine
    def forbidden(*args,**kwargs): raise AssertionError('central path reached')
    monkeypatch.setattr(TaskRouter,'run',forbidden); monkeypatch.setattr(Engine,'run',forbidden)
    with Application(ROOT) as app:
        r=app.ask('Ist 7 mal 6 mindestens 30?')
        assert r['verified'] and r['output']=='Das Produkt ist 42. Es ist mindestens 30.'
        assert len(r['members'])==4 and len(kinds(r,'JOIN'))==4 and len(kinds(r,'READY'))==1
        assert r['steps']==9 and r['head_visits']>0
        assert kinds(r,'JOIN')[-1]['revision']>kinds(r,'JOIN')[0]['revision']
        assert app.ask('Ist 3 mindestens 4?')['members']==['logic.ge']

def test_exhaustive_runtime():
    with Application(ROOT) as app:
        for x,y in itertools.product(range(32),repeat=2):
            t=(x*71+y*17)%962
            r=app.runtime.run(app.encoder.encode_task(f'Ist {x} mal {y} mindestens {t}?'),recall=False)
            assert r.verified,(x,y,r.error)
            assert r.frame.get('PRODUCT')==x*y and r.frame.get('GE')==int(x*y>=t)
            assert r.steps==y+3

def test_cooperation_persistent_and_ablation(tmp_path):
    db=tmp_path/'episodes.db'
    with Application(ROOT,cooperation_path=db) as app:
        r=app.ask('Ist 7 mal 6 mindestens 30?',session='a')
        assert r['verified'] and not kinds(r,'INVITE')
    with Application(ROOT,cooperation_path=db) as app:
        r=app.ask('Ist 8 mal 5 mindestens 41?',session='a')
        assert r['verified'] and r['output']=='Das Produkt ist 40. Es ist kleiner als 41.'
        assert kinds(r,'INVITE') and any(e['revision']==1 or e['revision']==0 for e in kinds(r,'JOIN') if e['source']!=2)
        cold=app.ask('Ist 8 mal 5 mindestens 41?',session='b'); disabled=app.ask('Ist 8 mal 5 mindestens 41?',session='a',recall=False)
        assert not kinds(cold,'INVITE') and not kinds(disabled,'INVITE') and cold['output']==disabled['output']==r['output']
        # Stored peer versions cannot match a different version.
        assert not app.cooperation.peers('a',2,'arith.init@99.0.0')

def test_mutated_activation_and_invitation():
    with Application(ROOT) as app:
        p=next(p for p in app.runtime.engine.participants if p.id=='arith.step')
        p.work.out.data.zero_(); r=app.ask('Ist 7 mal 6 mindestens 30?')
        assert not r['verified'] and r['status']=='STAGNATION' and not kinds(r,'READY')
    with Application(ROOT) as app:
        app.ask('Ist 7 mal 6 mindestens 30?')
        for p in app.runtime.engine.participants: p.join.out.data.zero_()
        r=app.ask('Ist 8 mal 5 mindestens 30?')
        assert r['verified'] and kinds(r,'INVITE')
        # No JOIN can be causally attributed to INVITE with zeroed join heads.
        invitations={e['seq'] for e in kinds(r,'INVITE')}
        assert all(e['cause'] not in invitations for e in kinds(r,'JOIN'))

def test_permutation_and_replay():
    with Application(ROOT) as app:
        req=app.encoder.encode_task('Ist 7 mal 6 mindestens 30?')
        ids=sorted(app.runtime.registry.experts); results=[]
        for seed in range(12):
            order=ids[:]; random.Random(seed).shuffle(order)
            r=app.runtime.run(req,recall=False,order=order); assert r.verified and r.frame.get('PRODUCT')==42
            replay=app.runtime.run(req,recall=False,order=order); assert replay.events==r.events and replay.frame==r.frame
            results.append(r.trace)

def test_memory_and_budgets():
    with Application(ROOT) as app:
        app.memory.write(17,9,'source'); app.memory.write(42,1,'distractor')
        r=app.ask('Multipliziere 3 mit dem Wert zu Schlüssel 17 und vergleiche mit 30.')
        assert r['verified'] and len(r['members'])==5 and all(a in r['trace'] for a in ('SEARCH','READ','EVICT','HALT'))
        assert app.memory.read(app.memory.search(17),17)==9
        for kwargs in ({'budget':2},{'work_budget':1}):
            r=app.ask('Ist 7 mal 6 mindestens 30?',**kwargs); assert r['status']=='BUDGET' and not kinds(r,'READY')
        r=app.ask('Multipliziere 3 mit dem Wert zu Schlüssel 99 und vergleiche mit 30.'); assert r['status']=='MEMORY' and not r['verified']

def test_conflict_generic_and_new_expert():
    schema=Schema('demo.conflict',(Field('A',0,10),Field('B',0,11)))
    def participant(name,bias):
        spec=ExpertSpec(name,'1.0.0','DIRECT',('A',),('B',),False,(1,1))
        core=DenseExpert(spec,{'w0':torch.tensor([[1.]]),'b0':torch.tensor([float(bias)])})
        gate=Gate(construct_gate(1,(((0,1),),)))
        return Participant(core,gate,gate,lambda f:torch.tensor([float(f.has('A'))]),lambda f,v:(Delta((('B',v[0]),)),None),('B',))
    c=Cooperation()
    try:
        e=CoalitionEngine(schema,[participant('demo.a',1),participant('demo.b',0)],c)
        r=e.run({'A':4},context=9,verify=lambda a,b:True,done=lambda f:f.has('B')); assert r.status=='CONFLICT' and not r.verified
        # Generic engine works with unrelated schema and only one expert; gate switches off when B exists.
        p=participant('demo.new',1); p.features=lambda f:torch.tensor([float(f.has('A') and not f.has('B'))])
        r=CoalitionEngine(schema,[p],c).run({'A':4},context=9,verify=lambda a,b:b.get('B')==5,done=lambda f:f.has('B'))
        assert r.verified and r.frame.get('B')==5
    finally:c.close()

def test_core_ablation_rejected():
    with Application(ROOT) as app:
        app.runtime.registry.experts['arith.step'].w0.data[0,1]=0
        r=app.ask('Ist 7 mal 6 mindestens 30?'); assert r['status']=='VERIFICATION' and not kinds(r,'READY') and '42' not in r['output']

def test_multiple_groups_shared_budget_and_isolation():
    with Application(ROOT) as app:
        text='Ist 7 mal 6 mindestens 30?; Ist 8 mal 5 mindestens 41?'
        r=app.ask(text,recall=False)
        assert r['verified'] and r['output']=='Das Produkt ist 42. Es ist mindestens 30. Das Produkt ist 40. Es ist kleiner als 41.'
        assert len(r['groups'])==2 and all(len(g['members'])==4 for g in r['groups'])
        events=r['events']; assert {e['group'] for e in events}=={1,2}
        assert [e['seq'] for e in events]==list(range(1,len(events)+1))
        byseq={e['seq']:e for e in events}; assert all(e['cause']==0 or (e['cause']<e['seq'] and byseq[e['cause']]['group']==e['group']) for e in events)
        proposals=[e['group'] for e in events if e['kind']=='PROPOSE']; assert proposals[:4]==[1,2,1,2]
        failed=app.ask(text,budget=8,recall=False); assert not failed['verified'] and failed['steps']<=8 and failed['status']=='BUDGET'
        failed=app.ask(text,work_budget=100,recall=False); assert not failed['verified'] and failed['parameter_visits']<=100

def test_production_http_decentralized():
    import threading,urllib.request
    from queue import Queue
    from vibeweights.local_api import make_server
    q=Queue(); stop=threading.Event()
    def worker():
        with Application(ROOT) as app,make_server(app,0) as server:
            server.timeout=.1; q.put(server.server_port)
            while not stop.is_set(): server.handle_request()
    th=threading.Thread(target=worker,daemon=True); th.start()
    try:
        port=q.get(timeout=10); url=f'http://127.0.0.1:{port}/v1/ask'
        req=urllib.request.Request(url,data=json.dumps({'text':'Ist 7 mal 6 mindestens 30?; Ist 3 mindestens 4?','recall':False}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as r: result=json.load(r)
        assert result['verified'] and len(result['groups'])==2 and {e['group'] for e in result['events']}=={1,2}
    finally: stop.set(); th.join(timeout=5)
    assert not th.is_alive()

def test_failed_cooperation_not_rewarded_and_unknown_peer():
    with Application(ROOT) as app:
        owner=next(p.identity for p in app.runtime.engine.participants if p.id=='arith.init')
        app.cooperation.feedback('a',2,[owner,'ghost.unknown@0.1.0'],True)
        app.runtime.registry.experts['arith.step'].w0.data[0,1]=0
        r=app.ask('Ist 7 mal 6 mindestens 30?',session='a')
        assert r['status']=='VERIFICATION' and not r['verified']
        assert app.cooperation.db.execute('SELECT COUNT(*) FROM cooperation_v1 WHERE session=? AND ok=0',('a',)).fetchone()[0]>0
        assert all(e['target'] in (0,*r['expert_ids']) for e in r['events'])

def test_head_cache_equivalence():
    with Application(ROOT) as app:
        req=app.encoder.encode_task('Ist 7 mal 6 mindestens 30?')
        cached=app.runtime.run(req,recall=False)
        app.runtime.engine.cache_heads=False
        raw=app.runtime.run(req,recall=False)
        assert cached.verified and raw.verified and cached.events==raw.events and cached.frame==raw.frame and cached.trace==raw.trace
        assert cached.head_visits<raw.head_visits
        assert cached.parameter_visits-cached.head_visits==raw.parameter_visits-raw.head_visits
