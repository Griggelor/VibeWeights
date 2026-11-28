import pytest
from vibeweights.abi import ContractError
from vibeweights.knowledge_learning import Domain
from vibeweights.learning_initiative import Opportunity
from vibeweights.task_learning import TaskLearning
from vibeweights.population import ARCHIVED, SelectionEvidence
from test_knowledge_learning_v08 import fixture, close, facts
from test_learning_initiative_v08 import Model, opportunities


def test_failed_task_to_learned_answer(tmp_path):
    c,p,k,a,l=fixture(tmp_path); t=TaskLearning(l); ops=opportunities(); d=ops[1].domain
    assert t.observe('ask-1',d,3)['outcome']=='MISSING_DOMAIN'
    chooser=Model('{"action":"LEARN","target":"b"}')
    r=t.propose('cycle',ops,chooser)
    assert r['plan']['evidence']=={'b':'ask-1'} and r['decision']['eligible']==['b']
    assert 'first skill' not in chooser.prompts[0] and 'teacher request' not in chooser.prompts[0]
    teacher=Model('{"facts":[{"entity":3,"label":1}]}')
    truth={3:1}.get
    assert t.stage('cycle',ops,teacher,oracles={'oracle-b':truth})['state']=='STAGED'
    assert t.observe('ask-staged',d,3)['outcome']=='RESERVED_DOMAIN'
    assert not p.resident_ids()
    assert l.promote(r['decision']['job_id'],truth,oracle_id='oracle-b')
    assert t.observe('ask-2',d,3)['answer']==1
    assert t.observe('ask-3',d,0)['outcome']=='ABSTAINED'
    assert t.observation('ask-1')['answer'] is None
    close(c,p,k)
    c,p,k,a,l=fixture(tmp_path); t=TaskLearning(l)
    assert t.propose('cycle',ops,chooser)==r
    assert t.stage('cycle',ops,teacher,oracles={'oracle-b':truth})['state']=='ACTIVE'
    assert len(chooser.prompts)==len(teacher.prompts)==1
    assert t.observe('ask-4',d,3)['answer']==1
    assert t.propose('next',ops,chooser)['decision'] is None
    close(c,p,k)


def test_no_observation_no_model_and_frozen_plan(tmp_path):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities();model=Model('{"action":"LEARN","target":"a"}')
    r=t.propose('empty',ops,model)
    assert r['decision'] is None and not model.prompts
    t.observe('task',ops[0].domain,0)
    assert t.propose('empty',ops,model)==r and not model.prompts
    assert t.propose('new',ops,model)['decision']['target']=='a'
    close(c,p,k)


def test_wrong_domain_and_duplicate_task(tmp_path):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities()
    with pytest.raises(ContractError): t.observe('bad',ops[0].domain,4)
    with pytest.raises(ContractError): t.observe('bad',ops[0].domain,True)
    t.observe('task',Domain('first',5,3),0)
    with pytest.raises(ContractError): t.observe('task',ops[0].domain,0)
    assert t.propose('cycle',ops,Model('{"action":"SKIP"}'))['decision'] is None
    close(c,p,k)


def test_reject_unobserved_target_and_policy_change(tmp_path):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities()
    t.observe('task',ops[0].domain,0)
    with pytest.raises(ContractError): t.propose('cycle',ops,Model('{"action":"LEARN","target":"b"}'))
    with pytest.raises(ContractError): t.stage('cycle',ops,Model('{}'),oracles={})
    changed=(Opportunity('a',ops[0].domain,'changed','req','oracle-a'),ops[1])
    with pytest.raises(ContractError): t.propose('cycle',changed,Model('{"action":"SKIP"}'))
    assert not c.list() and not l.intakes()
    close(c,p,k)


def test_archived_and_existing_facts_not_missing(tmp_path):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities();d=ops[0].domain
    ids=facts(k,a,d,{0:1});l.prepare('job','expert',d,ids,method='DIRECT');assert l.promote('job',{0:1}.get,oracle_id='oracle-a')
    ev=SelectionEvidence('report',1,True,False,True,True,True,True)
    p.select('expert',ARCHIVED,expected_revision=0,evidence=ev)
    assert t.observe('archived',d,0)['outcome']=='RESERVED_DOMAIN'
    facts(k,a,ops[1].domain,{0:2})
    assert t.observe('ledger',ops[1].domain,0)['outcome']=='RESERVED_DOMAIN'
    assert t.propose('cycle',ops,Model('{}'))['decision'] is None
    close(c,p,k)


def test_stale_gap_rechecked_before_proposal(tmp_path):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities();d=ops[0].domain
    t.observe('gap',d,0);facts(k,a,d,{0:1})
    model=Model('{}');assert t.propose('cycle',ops,model)['decision'] is None and not model.prompts
    close(c,p,k)


def test_guard_error_not_a_gap(tmp_path,monkeypatch):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities()
    def fail(): raise ContractError('source guard failed')
    monkeypatch.setattr(l,'_audit_locked',fail)
    with pytest.raises(ContractError): t.observe('task',ops[0].domain,0)
    assert not t.db.execute('SELECT 1 FROM task_observations_v08').fetchone()
    close(c,p,k)


def test_calendar_cli_task_to_answer(tmp_path,monkeypatch,capsys):
    import calendar, json, runpy
    from pathlib import Path
    from vibeweights.bootstrap import BootstrapExpert
    cli=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'learning_initiative_lmstudio.py'))['main']
    calls=[]
    def respond(self,text):
        calls.append(text)
        raw=('{"action":"LEARN","target":"calendar-2024"}' if len(calls)==1 else
            json.dumps({'facts':[{'entity':i,'label':calendar.monthrange(2024,i+1)[1]-28} for i in range(12)]}))
        return {'proposal':raw,'verified':False}
    monkeypatch.setattr(BootstrapExpert,'propose',respond)
    root=['--root',str(tmp_path)]; model=['--model','fixture-model']; query=['--topic','calendar-2024','--entity','1']
    def run(action,*args):
        assert cli([action,*root,*args])==0
        return json.loads(capsys.readouterr().out)
    assert run('observe',*query,'--task','before')['outcome']=='MISSING_DOMAIN'
    assert run('task-propose',*model)['plan']['evidence']=={'calendar-2024':'before'}
    assert run('task-stage',*model)['state']=='STAGED'
    assert run('promote')['job']['state']=='ACTIVE'
    assert run('observe',*query,'--task','after')['answer']==1
    assert len(calls)==2


def test_skip_is_persistent(tmp_path):
    c,p,k,a,l=fixture(tmp_path);t=TaskLearning(l);ops=opportunities();m=Model('{"action":"SKIP"}')
    t.observe('gap',ops[0].domain,0)
    r=t.propose('cycle',ops,m)
    assert r['decision']['action']=='SKIP' and t.stage('cycle',ops,m,oracles={}) is None
    assert t.propose('cycle',ops,m)==r and len(m.prompts)==1
    close(c,p,k)
