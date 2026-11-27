import json, threading, subprocess, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import pytest
from vibeweights.abi import ContractError
from vibeweights.environment import Signal, Environment, Limits, EventLedger
from vibeweights.factory import Catalog
from vibeweights.bootstrap import BootstrapSpec, BootstrapExpert
from vibeweights.next import demo
from vibeweights.application import Application

ROOT=Path(__file__).resolve().parents[1]

class Expert:
    def __init__(self,id,value,active=True): self.id=id; self.value=value; self.active=active
    def relevant(self,s): return self.active
    def forward(self,s):
        if isinstance(self.value,Exception): raise self.value
        return [Signal('global','result',(self.value,))]

@pytest.mark.parametrize('reverse',[False,True])
def test_competing_variants_order(reverse):
    ex=[Expert('logic@1',-1),Expert('logic@2',2)]
    if reverse: ex.reverse()
    r=Environment(ex,{('global','result'):lambda s,v:v==(2,)}).run([Signal('global','input',(1,))],('global','result'))
    assert r['status']=='OK' and r['value']==[2]
    assert len(r['hypotheses'])==2

def test_ambiguous_evidence_is_not_majority_vote():
    r=Environment([Expert('a',1),Expert('b',2),Expert('c',2)],{('global','result'):lambda s,v:True}).run([],('global','result'))
    assert r['status']=='CONFLICT' and r['value'] is None

def test_optional_probe_error_isolated():
    ex=[Expert('a',1),Expert('b',ValueError('bad'),False)]
    env=Environment(ex,{('global','result'):lambda s,v:v==(1,)})
    a=env.run([],('global','result'),limits=Limits(steps=2))
    b=env.run([],('global','result'),limits=Limits(steps=2,probes=1))
    assert a['status']==b['status']=='OK' and a['value']==b['value'] and a['steps']==b['steps']
    assert b['faults']==[('b','probe','ValueError')] and b['probes']==1

def test_incomplete_round_and_unverified_result():
    ex=[Expert('a',1),Expert('b',2)]
    assert Environment(ex,{}).run([],('global','result'))['status']=='STAGNATION'
    assert Environment(ex,{('global','result'):lambda s,v:True}).run([],('global','result'),limits=Limits(steps=1))['status']=='BUDGET'
    with pytest.raises(ContractError): Environment(ex,{}).run([Signal('global','result',(1,))],('global','result'))

def test_numeric_signal_contract():
    for value in (float('nan'),float('inf'),True,'hello'):
        with pytest.raises(ContractError): Signal('global','input',(value,))

def test_large_population_without_fixed_cap():
    ex=[Expert(f'e{i}',0,False) for i in range(600)]+[Expert('active',2)]
    r=Environment(ex,{('global','result'):lambda s,v:v==(2,)}).run([],('global','result'))
    assert r['status']=='OK' and r['steps']==2

def test_factory_lifecycle_and_hash(tmp_path):
    c=Catalog(tmp_path)
    opts=dict(source_topic='input',target_scope='global',target_topic='result')
    m=c.stage_affine('scale@1','scale',[[2,1]],[1],**opts)
    c.stage_affine('scale@2','scale',[[3,1]],[1],parents=('scale@1',),**opts)
    with pytest.raises(ContractError): c.load('scale@1')
    assert not c.admit('scale@2',[((3,4),(11,))])
    assert c.admit('scale@1',[((3,4),(11,)),((-2,5),(2,))])
    assert c.admit('scale@2',[((3,4),(14,))])
    assert c.load('scale@1').compute((7,8))==(23,)
    assert c.load('scale@2').compute((7,8))==(30,)
    with pytest.raises(ContractError): c.load('scale@1',byte_budget=1)
    c.close(); c=Catalog(tmp_path)
    assert len(c.list())==2 and c.manifest('scale@2')['parents']==['scale@1']
    (tmp_path/(m['sha256']+'.safetensors')).write_bytes(b'corrupt')
    with pytest.raises(ContractError): c.load('scale@1')
    c.close()

def test_demo_persistence_and_cross_scope(tmp_path):
    for x in (-7,0,9):
        r=demo(tmp_path,x); assert r['status']=='OK' and r['value']==[2*x+1]
        l=EventLedger(tmp_path/'Data'/'environment.sqlite'); assert l.read(r['run_id'])['value']==[2*x+1];l.close()

def test_bootstrap_http_real_transport_test_double():
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            data=json.dumps({'choices':[{'message':{'content':'candidate'}}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*a): pass
    with HTTPServer(('127.0.0.1',0),Handler) as server:
        t=threading.Thread(target=server.handle_request);t.start()
        try: r=BootstrapExpert(BootstrapSpec('qwen@1','local-model',server.server_port,32,3)).propose('test')
        finally: t.join(5)
    assert not t.is_alive() and not r['verified'] and r['proposal']=='candidate'
    assert requests[0][0]=='/v1/chat/completions' and requests[0][1]['max_tokens']==32

def test_rating_not_overwritten_by_preference():
    with Application(ROOT) as app:
        r=app.ask('Ist 3 mindestens 4?'); app.feedback(r['answer_id'],'positive');app.feedback(r['answer_id'],'preference','kurz')
        assert sum(m['positive_answer_associations'] for m in app.metrics())==1

def test_cli_from_other_directory(tmp_path):
    import os
    env=dict(os.environ,VIBEWEIGHTS_ROOT=str(ROOT))
    p=subprocess.run([sys.executable,'-m','vibeweights.chat','--memory',':memory:','--journal',':memory:','--ask','Ist 3 mindestens 4?'],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['verified']

def test_analytic_consolidation_on_independent_cases(tmp_path):
    c=Catalog(tmp_path)
    a=c.stage_affine('a@1','a',[[2],[3]],[1,-1],source_topic='input',target_scope='work',target_topic='mid')
    b=c.stage_affine('b@1','b',[[4,-1]],[2],source_topic='mid',target_scope='global',target_topic='result')
    assert c.admit('a@1',[((2,),(5,5)),((-1,),(-1,-4))])
    assert c.admit('b@1',[((2,3),(7,)),((-1,4),(-6,))])
    m=c.compose_affine('fused@1','fused','a@1','b@1')
    assert m['parents']==['a@1','b@1'] and m['parameters']==2<a['parameters']+b['parameters']==7
    assert c.admit('fused@1',[((0,),(7,)),((2,),(17,))])
    ea,eb,ef=(c.load(x) for x in ('a@1','b@1','fused@1'))
    for x in range(-100,101):
        assert ef.compute((x,))==eb.compute(ea.compute((x,)))==(5*x+7,)
    assert len(c.list())==3
    c.close()

def test_verifier_error_fails_closed():
    def broken(s,v): raise RuntimeError('invalid verifier')
    r=Environment([Expert('a',1)],{('global','result'):broken}).run([],('global','result'))
    assert r['status']=='INVALID' and r['value'] is None

def test_decoder_error_does_not_erase_core_observations(monkeypatch):
    with Application(ROOT) as app:
        def fail(view): raise ContractError('decoder fault')
        monkeypatch.setattr(app.decoder,'decode',fail)
        r=app.ask('Ist 3 mindestens 4?')
        assert r['status']=='INVALID' and not r['verified'] and r['core_verified'] and r['events']
        assert sum(m['verified_group_contributions'] for m in app.metrics())==1
