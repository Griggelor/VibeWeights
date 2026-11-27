"""Independent characterization of v0.5.0. PASS means a limitation is reproduced,
not that the corresponding target requirement is satisfied.
Run from extracted v0.5.0 after editable install, with VIBEWEIGHTS_ROOT set if needed.
"""
import os, subprocess, sys
from pathlib import Path
import pytest
import torch
from vibeweights.application import Application
from vibeweights.abi import ContractError
from vibeweights.coalition import CoalitionEngine, Cooperation, Gate, Participant, construct_gate
from vibeweights.platform import Schema, Field, ExpertSpec, DenseExpert, Delta, Registry
ROOT=Path(os.environ.get('VIBEWEIGHTS_ROOT',Path.cwd())).resolve()


def test_R01_variants_same_logical_id_cannot_coexist():
    with Application(ROOT) as app:
        reg=app.runtime.registry
        v1=reg.experts['logic.ge']
        s=v1.spec
        v2=ExpertSpec(s.id,'2.0.0',s.origin,s.inputs,s.outputs,s.mask,s.dimensions,s.action_count)
        core=DenseExpert(v2,{n:t.detach().clone() for n,t in v1.named_parameters()})
        with pytest.raises(ContractError,match='duplicate expert'): reg.register(core)
        reg.register(core,replace=True)
        assert reg.experts[s.id].spec.version=='2.0.0'
        assert sum(e.spec.id==s.id for e in reg.experts.values())==1
        with pytest.raises(ContractError,match='registry limits'): Registry(reg.schema,max_experts=513)
        assert reg.max_parameters==1_000_000


def test_R02_order_decides_between_competing_proposals():
    schema=Schema('review.alternatives',(Field('A',0,10),Field('B',0,11)))
    def p(name,bias):
        spec=ExpertSpec(name,'1.0.0','DIRECT',('A',),('B',),False,(1,1))
        expert=DenseExpert(spec,{'w0':torch.tensor([[1.]]),'b0':torch.tensor([float(bias)])})
        g=Gate(construct_gate(1,(((0,1),),)))
        return Participant(expert,g,g,lambda f:torch.tensor([float(f.has('A') and not f.has('B'))]),lambda f,v:(Delta((('B',v[0]),)),None),('B',))
    c=Cooperation()
    try:
        engine=CoalitionEngine(schema,[p('review.good',1),p('review.bad',0)],c)
        args=dict(context=1,verify=lambda a,b:b.get('B')==5,done=lambda f:f.has('B'),recall=False)
        assert engine.run({'A':4},order=['review.good','review.bad'],**args).status=='OK'
        r=engine.run({'A':4},order=['review.bad','review.good'],**args)
        assert r.status=='VERIFICATION' and r.frame.get('B')==4
        assert r.contributors==('review.bad',)
    finally: c.close()


def test_R03_shot_can_consume_last_required_step():
    with Application(ROOT) as app:
        text='Ist 7 mal 6 mindestens 30?'
        assert app.ask(text,budget=9,shots=0,recall=False)['verified']
        r=app.ask(text,budget=9,shots=4,shot_seed=8,recall=False)
        assert r['status']=='BUDGET' and r['steps']==9
        assert any(e['kind']=='PROBE' for e in r['events'])


def test_R04_failed_shadow_probe_breaks_good_inference():
    with Application(ROOT) as app:
        app.runtime.registry.experts['memory.action'].w0.data.fill_(float('nan'))
        text='Ist 7 mal 6 mindestens 30?'
        assert app.ask(text,shots=0,recall=False)['verified']
        r=app.ask(text,shots=1,shot_seed=8,recall=False)
        assert r['status']=='INVALID' and not r['verified']
        assert 'invalid shot probe' in r['error']


def test_R05_feedback_does_not_change_recruitment_or_trigger_revision():
    with Application(ROOT) as app:
        text='Ist 7 mal 6 mindestens 30?'
        r=app.ask(text,session='review')
        owner=next(p.identity for p in app.runtime.engine.participants if p.id=='arith.init')
        peers=app.cooperation.peers('review',2,owner)
        assert peers
        app.feedback(r['answer_id'],'correction','Das Ergebnis sei falsch.',session='review')
        assert app.cooperation.peers('review',2,owner)==peers
        assert len(app.reflections(session='review'))==1
        assert len(app.journal.history('review'))==1
        assert app.ask(text,session='review')['output']==r['output']


def test_R06_preference_erases_rating_from_metrics():
    with Application(ROOT) as app:
        r=app.ask('Ist 3 mindestens 4?')
        app.feedback(r['answer_id'],'positive')
        assert sum(m['positive_answer_associations'] for m in app.metrics())==1
        app.feedback(r['answer_id'],'preference','Bitte kürzer.')
        assert sum(m['positive_answer_associations'] for m in app.metrics())==0
        assert len(app.reflections())==2


def test_R07_output_failure_loses_contributor_trace():
    with Application(ROOT) as app:
        app.decoder.weights.relation_w.data.zero_()
        app.decoder.weights.relation_b.data.copy_(torch.tensor([0.,1.]))
        r=app.ask('Ist 7 mal 6 mindestens 99?',session='review')
        assert r['status']=='INVALID' and not r['verified']
        assert 'language output verification' in r['error']
        assert r['trace'] and 'events' not in r and app.metrics(session='review')==[]
        assert app.cooperation.db.execute("SELECT COUNT(*) FROM cooperation_v1 WHERE session='review' AND ok=1").fetchone()[0]==12


def test_R08_default_application_forgets_on_restart():
    with Application(ROOT) as app:
        r=app.ask('Ist 7 mal 6 mindestens 30?')
        app.feedback(r['answer_id'],'positive')
        assert app.metrics()
    with Application(ROOT) as app:
        assert app.metrics()==[] and app.reflections()==[] and app.journal.history('default')==[]
        assert not any(e['kind']=='INVITE' for e in app.ask('Ist 7 mal 6 mindestens 30?')['events'])


def test_R09_installed_cli_cannot_find_weights_from_other_cwd(tmp_path):
    p=subprocess.run([sys.executable,'-m','vibeweights.chat','--ask','Ist 3 mindestens 4?'],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert p.returncode!=0 and 'empty registry' in p.stderr


def test_R10_recall_disabled_still_learns_cooperation():
    with Application(ROOT) as app:
        r=app.ask('Ist 7 mal 6 mindestens 30?',session='review',recall=False)
        assert r['verified'] and not any(e['kind']=='INVITE' for e in r['events'])
        owner=next(p.identity for p in app.runtime.engine.participants if p.id=='arith.init')
        assert app.cooperation.peers('review',2,owner)
