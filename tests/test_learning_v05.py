from pathlib import Path
import pytest
from vibeweights.application import Application
from vibeweights.abi import ContractError
ROOT=Path(__file__).resolve().parents[1]

def kinds(r,k): return [e for e in r['events'] if e['kind']==k]

def test_shots_seed_replay_and_accounting():
    with Application(ROOT) as app:
        text='Ist 3 mindestens 4?'
        control=app.ask(text,recall=False)
        a=app.ask(text,recall=False,shots=4,shot_seed=7)
        b=app.ask(text,recall=False,shots=4,shot_seed=7)
        assert a['verified'] and a['output']==control['output']==b['output']
        assert a['events']==b['events'] and a['steps']==1+len(kinds(a,'PROBE'))
        assert kinds(a,'SHOT') and len(kinds(a,'SHOT'))<=4
        targets=[e['target'] for e in kinds(a,'SHOT')]; assert len(set(targets))==len(targets)
        assert set(targets).isdisjoint(e['source'] for e in kinds(a,'PROPOSE'))
        for e in kinds(a,'SHOT'): assert any(j['source']==e['target'] and j['cause']==e['seq'] for j in kinds(a,'JOIN'))
        assert not kinds(control,'SHOT')
        assert a['parameter_visits']>=control['parameter_visits']

def test_shots_shared_budget_and_input_validation():
    with Application(ROOT) as app:
        r=app.ask('Ist 3 mindestens 4?; Ist 7 mindestens 6?',shots=3,shot_seed=1,recall=False)
        assert r['verified'] and len(kinds(r,'SHOT'))<=3
        r=app.ask('Ist 3 mindestens 4?',shots=4,budget=1,work_budget=500,recall=False)
        assert r['steps']<=1 and r['parameter_visits']<=500
        for shots in (-1,129,True): assert app.ask('Ist 3 mindestens 4?',shots=shots)['status']=='INVALID'

def test_feedback_persistence_and_separate_attribution(tmp_path):
    journal=tmp_path/'journal.sqlite'
    with Application(ROOT,journal_path=journal) as app:
        a=app.ask('Ist 3 mindestens 4?',session='a',shots=3,shot_seed=1)
        assert a['verified']
        f=app.feedback(a['answer_id'],'negative','Antwortstil',session='a')
        assert f['state']=='PENDING'
        with pytest.raises(ContractError): app.feedback(a['answer_id'],'positive',session='b')
        with pytest.raises(ContractError): app.ask('Ist 3 mindestens 4?',session='b',parent_answer_id=a['answer_id'])
        metrics=app.metrics(session='a'); assert sum(m['negative_or_corrected_answer_associations'] for m in metrics)==1
        assert all(m['negative_or_corrected_answer_associations']==0 for m in metrics if not m['proposals'])
        assert app.metrics(session='b')==[]
    with Application(ROOT,journal_path=journal) as app:
        assert app.reflections(session='a')[0]['answer_id']==a['answer_id']
        b=app.ask('Ist 3 mindestens 4?',session='a',parent_answer_id=a['answer_id'])
        assert b['answer_id']!=a['answer_id'] and b['parent_answer_id']==a['answer_id']
        app.feedback(b['answer_id'],'positive',session='a')
        assert sum(m['positive_answer_associations'] for m in app.metrics(session='a'))==1
        assert sum(m['verified_group_contributions'] for m in app.metrics(session='a'))==2
        app.feedback(a['answer_id'],'positive',session='a')
        assert sum(m['negative_or_corrected_answer_associations'] for m in app.metrics(session='a'))==0
        assert len(app.reflections(session='a'))==3

def test_variant_lineage_no_automatic_activation():
    with Application(ROOT) as app:
        l=app.learning; l.stage_variant('logic@1','logic','1'*64,'DIRECT'); l.stage_variant('logic@2','logic','2'*64,'SYNTHESIS','logic@1')
        assert [x['state'] for x in l.variants()]==['STAGED','STAGED']
        assert len(app.runtime.engine.participants)==5
        with pytest.raises(ContractError): l.stage_variant('other@1','other','3'*64,'DIRECT','logic@1')
        with pytest.raises(ContractError): l.stage_variant('logic@2','logic','3'*64,'DIRECT')

def test_http_learning_endpoints():
    import threading,urllib.request,json
    from queue import Queue
    from vibeweights.local_api import make_server
    q=Queue(); stop=threading.Event()
    def worker():
        with Application(ROOT) as app,make_server(app,0) as server:
            server.timeout=.1; q.put(server.server_port)
            while not stop.is_set(): server.handle_request()
    th=threading.Thread(target=worker,daemon=True); th.start()
    try:
        url=f'http://127.0.0.1:{q.get(timeout=10)}'
        def post(path,data):
            req=urllib.request.Request(url+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=5) as r: return json.load(r)
        a=post('/v1/ask',{'text':'Ist 3 mindestens 4?','shots':2})
        assert a['verified'] and post('/v1/feedback',{'answer_id':a['answer_id'],'kind':'positive'})['state']=='PENDING'
        assert post('/v1/metrics',{}) and post('/v1/reflections',{})
    finally: stop.set(); th.join(timeout=5)
    assert not th.is_alive()

def test_actual_shadow_forward_no_memory_action(monkeypatch):
    with Application(ROOT) as app:
        calls=[]
        def forbidden(*args,**kwargs):
            calls.append(args); raise AssertionError('probe invoked memory action')
        monkeypatch.setattr(app.memory,'search',forbidden)
        monkeypatch.setattr(app.memory,'read',forbidden)
        # Find a seed whose inactive memory action expert has a valid masked input.
        for seed in range(32):
            result=app.ask('Ist 3 mindestens 4?',recall=False,shots=4,shot_seed=seed)
            if kinds(result,'PROBE'): break
        assert result['verified'] and kinds(result,'PROBE') and not calls
        assert result['steps']==1+len(kinds(result,'PROBE'))
        assert result['contributors']==['logic.ge'] and result['output']=='3 ist kleiner als 4.'
        assert all(result['expert_ids'][e['source']]=='memory.action' for e in kinds(result,'PROBE'))
        assert all(m['verified_group_contributions']==0 for m in app.metrics() if m['expert'].startswith('memory.action@'))
