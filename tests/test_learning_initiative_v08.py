import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from vibeweights.abi import ContractError
from vibeweights.bootstrap import BootstrapExpert, BootstrapSpec
from vibeweights.knowledge_learning import Domain
from vibeweights.learning_initiative import LearningInitiative, Opportunity
from test_knowledge_learning_v08 import fixture, close, facts


class Model(BootstrapExpert):
    def __init__(self, response):
        super().__init__(BootstrapSpec('model','test-model'))
        self.response=response; self.prompts=[]
    def propose(self,text):
        self.prompts.append(text)
        return {'proposal':self.response,'verified':False}


def opportunities():
    return (Opportunity('a',Domain('first',4,3),'first skill','first teacher request','oracle-a'),
            Opportunity('b',Domain('second',5,3),'second skill','second teacher request','oracle-b'))


def test_model_selects_and_stages_only_then_host_promotes(tmp_path):
    c,p,k,a,l=fixture(tmp_path); ops=opportunities(); i=LearningInitiative(l)
    chooser=Model('{"action":"LEARN","target":"b"}')
    teacher=Model('{"facts":[{"entity":0,"label":2},{"entity":3,"label":1}]}')
    r=i.propose('cycle',ops,chooser)
    assert r['target']=='b' and r['eligible']==['a','b']
    assert 'teacher request' not in chooser.prompts[0] and 'oracle-b' not in chooser.prompts[0]
    assert p.resident_ids()==() and k.search(a,topic='second')==()
    m=i.stage('cycle',ops,teacher,oracles={'oracle-b':{0:2,3:1}.get})
    assert m['state']=='STAGED' and p.resident_ids()==()
    assert len(teacher.prompts)==1 and 'second teacher request' in teacher.prompts[0]
    assert k.search(a,topic='first')==()
    assert l.promote(r['job_id'],{0:2,3:1}.get,oracle_id='oracle-b')
    assert p.resident_ids()==(r['expert_id'],)
    assert ops[1].domain.decode(p.compute(r['expert_id'],ops[1].domain.encode(3)))==1
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path);i=LearningInitiative(l)
    assert i.propose('cycle',ops,chooser)==r
    assert i.stage('cycle',ops,teacher,oracles={'oracle-b':{0:2,3:1}.get})['state']=='ACTIVE'
    assert len(chooser.prompts)==len(teacher.prompts)==1
    assert len(l.intakes())==1 and len(c.list())==1
    close(c,p,k)


@pytest.mark.parametrize('bad',['{"action":"LEARN","target":"x"}', '{"action":"LEARN","target":"a","code":"exec"}',
    '{"action":"LEARN","target":"a","target":"b"}', '{"action":"PROMOTE","target":"a"}', '```json\n{}\n```'])
def test_untrusted_proposal_rejected(tmp_path,bad):
    c,p,k,a,l=fixture(tmp_path); i=LearningInitiative(l)
    with pytest.raises(ContractError): i.propose('cycle',opportunities(),Model(bad))
    assert not c.list() and not l.intakes()
    with pytest.raises(ContractError): i.receipt('cycle')
    close(c,p,k)


def test_skip_and_no_eligible(tmp_path):
    c,p,k,a,l=fixture(tmp_path);i=LearningInitiative(l);ops=opportunities(); chooser=Model('{"action":"SKIP"}')
    assert i.propose('skip',ops,chooser)['action']=='SKIP'
    assert i.stage('skip',ops,chooser,oracles={}) is None
    for op in ops: facts(k,a,op.domain,{0:1})
    assert i.propose('none',ops,chooser)['eligible']==[]
    assert len(chooser.prompts)==1 and not c.list()
    close(c,p,k)


def test_policy_and_stale_opportunity(tmp_path):
    c,p,k,a,l=fixture(tmp_path); i=LearningInitiative(l);ops=opportunities(); chooser=Model('{"action":"LEARN","target":"a"}')
    i.propose('cycle',ops,chooser)
    changed=(Opportunity('a',ops[0].domain,'changed','first teacher request','oracle-a'),ops[1])
    with pytest.raises(ContractError): i.propose('cycle',changed,chooser)
    with pytest.raises(ContractError): i.stage('cycle',changed,chooser,oracles={'oracle-a':{0:1}.get})
    facts(k,a,ops[0].domain,{0:1})
    with pytest.raises(ContractError): i.stage('cycle',ops,chooser,oracles={'oracle-a':{0:1}.get})
    assert len(chooser.prompts)==1 and not l.intakes()
    close(c,p,k)


def test_wrong_teacher_and_missing_oracle(tmp_path):
    c,p,k,a,l=fixture(tmp_path);i=LearningInitiative(l);ops=opportunities()
    i.propose('cycle',ops,Model('{"action":"LEARN","target":"a"}'))
    teacher=Model('{"facts":[{"entity":0,"label":2}]}')
    with pytest.raises(ContractError): i.stage('cycle',ops,teacher,oracles={})
    assert not teacher.prompts
    with pytest.raises(ContractError): i.stage('cycle',ops,teacher,oracles={'oracle-a':{0:1}.get})
    assert not l.intakes() and not c.list() and not k.search(a,topic='first')
    close(c,p,k)


def test_recovery_after_committed_intake(tmp_path,monkeypatch):
    c,p,k,a,l=fixture(tmp_path);i=LearningInitiative(l);ops=opportunities()
    i.propose('cycle',ops,Model('{"action":"LEARN","target":"a"}'))
    teacher=Model('{"facts":[{"entity":0,"label":1}]}')
    with monkeypatch.context() as mp:
        def stop(*args,**kwargs): raise RuntimeError('interrupted')
        mp.setattr(l,'resume_intake',stop)
        with pytest.raises(RuntimeError): i.stage('cycle',ops,teacher,oracles={'oracle-a':{0:1}.get})
    close(c,p,k);c,p,k,a,l=fixture(tmp_path);i=LearningInitiative(l)
    assert i.stage('cycle',ops,teacher,oracles={'oracle-a':{0:1}.get})['state']=='STAGED'
    assert len(teacher.prompts)==1 and len(l.intakes())==1 and not p.resident_ids()
    close(c,p,k)


def test_http_loopback_teacher_and_initiative(tmp_path):
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path,req))
            proposal='{"action":"LEARN","target":"b"}' if len(calls)==1 else '{"facts":[{"entity":0,"label":2}]}'
            payload=json.dumps({'choices':[{'message':{'content':proposal}}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    c,p,k,a,l=fixture(tmp_path)
    try:
        model=BootstrapExpert(BootstrapSpec('http','fixture-model',port=server.server_port,timeout=5))
        i=LearningInitiative(l);ops=opportunities();r=i.propose('http',ops,model)
        assert i.stage('http',ops,model,oracles={'oracle-b':{0:2}.get})['state']=='STAGED'
        assert l.promote(r['job_id'],{0:2}.get,oracle_id='oracle-b')
        assert len(calls)==2 and all(path=='/v1/chat/completions' for path,_ in calls)
    finally:
        close(c,p,k);server.shutdown();server.server_close();thread.join()


def test_calendar_cli_proposal_stage_promotion(tmp_path,monkeypatch,capsys):
    import calendar, runpy
    from pathlib import Path
    cli=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'learning_initiative_lmstudio.py'))['main']
    calls=[]
    def respond(self,text):
        calls.append(text)
        raw=('{"action":"LEARN","target":"calendar-2024"}' if len(calls)==1 else
            json.dumps({'facts':[{'entity':i,'label':calendar.monthrange(2024,i+1)[1]-28} for i in range(12)]}))
        return {'proposal':raw,'verified':False}
    monkeypatch.setattr(BootstrapExpert,'propose',respond)
    args=['--root',str(tmp_path),'--model','fixture-model']
    assert cli(['propose',*args])==0
    assert json.loads(capsys.readouterr().out)['target']=='calendar-2024'
    assert cli(['stage',*args])==0
    assert json.loads(capsys.readouterr().out)['state']=='STAGED'
    assert cli(['promote','--root',str(tmp_path)])==0
    out=json.loads(capsys.readouterr().out)
    assert out['job']['state']=='ACTIVE' and out['answers_days']==[calendar.monthrange(2024,i+1)[1] for i in range(12)]
    assert len(calls)==2
