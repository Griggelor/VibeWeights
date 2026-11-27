from dataclasses import replace
from pathlib import Path
import hashlib,itertools,json,random,subprocess,sys,threading,urllib.request,urllib.error
import pytest,torch
from safetensors.torch import save_file
from vibeweights.abi import ContractError,Slot
from vibeweights.platform import Field,Schema,Frame,Registry,ExpertSpec,DenseExpert,Delta,Rule,Plan,Engine,ActionHandler
from vibeweights.arithmetic_plan import PlatformRuntime,SCHEMA,make_plan
from vibeweights.runtime import Status
from vibeweights.memory import Memory
from vibeweights.language_weights import LexicalWeights,build,WeightedInput
from vibeweights.application import BaselineApplication as Application,FinalView
from vibeweights.local_api import make_server
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def runtime(): return PlatformRuntime.load(ROOT)

def test_platform_exhaustive(runtime):
    for x,y in itertools.product(range(32),repeat=2):
        t=(x*71+y*17)%962
        r=runtime.run(x,t,y=y)
        assert r.status==Status.OK and r.state.get(Slot.PRODUCT)==x*y and r.state.get(Slot.GE)==int(x*y>=t)
        assert r.steps==y+3 and r.trace.count('arith.step')==y

def test_platform_all_indexed():
    m=Memory(':memory:')
    try:
        rt=PlatformRuntime.load(ROOT,m)
        for y in range(32): m.write(y+1,y,'original')
        for y in range(32):
            r=rt.run(31,960,key=y+1); assert r.status==Status.OK and r.state.get(Slot.PRODUCT)==31*y and not r.state.valid[Slot.Y]
            assert r.steps==y+8 and r.trace.count('READ')==1 and r.trace.count('EVICT')==1
        assert rt.run(1,1,key=999).status==Status.MEMORY
    finally: m.close()

def test_platform_corruption_fails_closed():
    rt=PlatformRuntime.load(ROOT); rt.registry.experts['arith.step'].w0.data[0,1]=0
    r=rt.run(7,30,y=6); assert r.status==Status.VERIFICATION and FinalView.from_result(r).product is None
    rt=PlatformRuntime.load(ROOT); rt.registry.experts['arith.step'].b0.data[1]=0
    assert rt.run(7,30,y=6).status==Status.STAGNATION
    rt=PlatformRuntime.load(ROOT); rt.registry.experts['logic.ge'].w1.data.zero_()
    assert rt.run(7,30,y=6).status==Status.VERIFICATION

def test_budget_and_routing(runtime):
    assert runtime.run(7,30,y=6,budget=2).status==Status.BUDGET
    assert runtime.run(7,30,y=6,budget=True).status==Status.INVALID
    plan=make_plan(); inputs={'X':7,'Y':6,'THRESHOLD':30}
    rules=plan.rules+(replace(plan.rules[3],id='duplicate'),)
    assert Engine(runtime.registry).run(replace(plan,rules=rules),inputs).status=='ROUTING'
    assert Engine(runtime.registry).run(replace(plan,rules=tuple(replace(r,ready=lambda f:False) for r in plan.rules)),inputs).status=='ROUTING'
    assert Engine(runtime.registry).run(replace(plan,requirements=()),inputs).status=='INVALID'

def test_generic_manifest_new_schema_no_core_patch(tmp_path):
    schema=Schema('demo.increment',(Field('A',0,10),Field('B',0,11)))
    name='demo.increment'; d=tmp_path/'catalog/manifests'; d.mkdir(parents=True); w=tmp_path/'artifacts/weights'; w.mkdir(parents=True)
    path=w/f'{name}.safetensors'; save_file({'w0':torch.tensor([[1.]]),'b0':torch.tensor([1.])},str(path))
    manifest=dict(id=name,version='1.0.0',weight_origin='DIRECT',latent_abi='0.2',schema_id=schema.id,schema_sha256=schema.fingerprint,architecture='affine_relu',parameters=2,input_contract=['A'],output_contract=['B'],mask_input=False,layer_dimensions=[1,1],weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    mp=d/f'{name}.json'; mp.write_text(json.dumps(manifest),encoding='utf-8')
    reg=Registry.discover(tmp_path,schema,max_parameters=2)
    plan=Plan('increment',schema,(Rule('increment',name,lambda f:not f.has('B')),),('A',),('B',),lambda f:f.has('B'),lambda a,b:b.get('B')==a.get('A')+1,((name,('A',),('B',),False,0),))
    for a in range(11):
        r=Engine(reg).run(plan,{'A':a}); assert r.verified and r.frame.get('B')==a+1
    old=reg.experts[name]
    new=DenseExpert(replace(old.spec,version='1.0.1'),{'w0':torch.tensor([[1.]]),'b0':torch.tensor([1.])})
    reg.register(new,replace=True); assert reg.experts[name] is new and Engine(reg).run(plan,{'A':10}).verified
    with pytest.raises(ContractError): reg.register(DenseExpert(replace(old.spec,outputs=('A',)),{'w0':torch.tensor([[1.]]),'b0':torch.tensor([1.])}),replace=True)
    with pytest.raises(ContractError): Registry.discover(tmp_path,schema,max_parameters=1)
    with pytest.raises(ContractError): Registry.discover(tmp_path,Schema(schema.id,(Field('A',0,9),Field('B',0,11))))
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ContractError): Registry.discover(tmp_path,schema)

def test_frame_invalid():
    f=Frame.empty(SCHEMA)
    with pytest.raises(ContractError): f.update({'X':True},('X',))
    with pytest.raises(ContractError): f.update({'X':1},('Y',))
    with pytest.raises(ContractError): Delta((('X',1),('X',2))).apply(f,('X',))
    with pytest.raises(ContractError): f.get('X')
    assert f.update({'X':0},('X',)).has('X') and not f.has('X')

def test_memory_cas(tmp_path):
    path=tmp_path/'m.db'; a=Memory(path); b=Memory(path)
    try:
        old=a.write(42,6,'original'); latest=b.update(42,9,'revision',expected_id=old)
        with pytest.raises(ContractError): a.update(42,3,'stale',expected_id=old)
        assert a.search(42)==latest and a.read(old,42)==6 and a.provenance(latest)=={'key':42,'version':2,'source':'revision'}
    finally: a.close(); b.close()

def test_language_weight_reproduction_and_oracle(tmp_path):
    build(tmp_path)
    assert (tmp_path/'artifacts/language/lexical.safetensors').read_bytes()==(ROOT/'artifacts/language/lexical.safetensors').read_bytes()
    w=LexicalWeights.load(tmp_path); assert sum(p.numel() for p in w.parameters())==85
    rng=random.Random(2301)
    for i in list(range(1000))+[2**24-1,2**24-2,10000000]+[rng.randrange(2**24) for _ in range(1000)]: assert w.decimal(str(i))==i
    assert w.decimal('00000042')==42 and w.relation(0)=='kleiner als' and w.relation(1)=='mindestens'
    for bad in ['16777216','-1','1.2','abc','123456789']:
        with pytest.raises(ContractError): w.decimal(bad)
    w.digits_w.data.zero_(); assert w.decimal('7')==0
    with pytest.raises(ContractError): WeightedInput(w).encode('Ist 7 mal 6 mindestens 30?')
    w.relation_w.data.zero_(); assert w.relation(1)=='kleiner als' # mutation causes output error, no symbolic correction

def test_application_persistence_and_final_boundary(tmp_path):
    m=tmp_path/'m.db'; j=tmp_path/'j.db'
    with Application(ROOT,memory_path=m,journal_path=j) as app:
        app.memory.write(42,6,'source'); r=app.ask('Multipliziere 7 mit dem Wert zu Schlüssel 42 und vergleiche mit 30.',session='a')
        assert r['verified'] and r['output']=='Das Produkt ist 42. Es ist mindestens 30.'
        assert app.ask('Ist 7 mal 6 mindestens 30?',session='b',budget=1)['status']=='BUDGET'
        assert app.ask('Was ist sieben mal sechs?',session='b')['status']=='INVALID'
        app.runtime.registry.experts['arith.step'].w0.data[0,1]=0
        r=app.ask('Ist 7 mal 6 mindestens 30?',session='a'); assert not r['verified'] and r['output']=='Kein Ergebnis (VERIFICATION).'
    with Application(ROOT,memory_path=m,journal_path=j) as app:
        assert len(app.journal.history('a'))==2 and len(app.journal.history('b'))==2
        assert app.memory.read(app.memory.search(42),42)==6
        assert app.journal.history('unknown')==[]

def test_local_http():
    # SQLite and HTTPServer constructed and used in same worker thread.
    from queue import Queue
    q=Queue(); stop=threading.Event()
    def worker():
        with Application(ROOT) as app,make_server(app,0) as server:
            server.timeout=.1; q.put(server.server_port)
            while not stop.is_set(): server.handle_request()
    th=threading.Thread(target=worker,daemon=True); th.start()
    try:
        port=q.get(timeout=10); base=f'http://127.0.0.1:{port}'
        with urllib.request.urlopen(base+'/health',timeout=5) as r: assert json.load(r)['status']=='OK'
        req=urllib.request.Request(base+'/v1/ask',data=json.dumps({'text':'Ist 7 mal 6 mindestens 30?'}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as r: assert json.load(r)['verified']
        req.add_header('Origin','https://example.com')
        with pytest.raises(urllib.error.HTTPError) as err: urllib.request.urlopen(req,timeout=5)
        assert err.value.code==403
    finally: stop.set(); th.join(timeout=5)
    assert not th.is_alive()

def test_chat_cli():
    r=subprocess.run([sys.executable,'-m','vibeweights.chat','--ask','Ist 7 mal 6 mindestens 30?'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['verified']


def test_multi_task_and_language_verification():
    with Application(ROOT) as app:
        assert app.router.task_ids==('bounded.compare','bounded.multiply_compare')
        r=app.ask('Ist 42 mindestens 30?'); assert r['verified'] and r['output']=='42 ist mindestens 30.' and r['trace']==['logic.ge'] and r['steps']==1
        r=app.ask('Ist 3 mindestens 4?'); assert r['verified'] and r['output']=='3 ist kleiner als 4.'
        r=app.ask('Ist 7 mal 6 mindestens 30?'); assert r['verified'] and r['steps']==9
        app.decoder.weights.relation_w.data.zero_()
        r=app.ask('Ist 42 mindestens 30?'); assert not r['verified'] and r['status']=='INVALID' and 'mindestens' not in r['output']
        app.encoder.weights.digits_w.data.zero_()
        r=app.ask('Ist 42 mindestens 30?'); assert not r['verified'] and r['status']=='INVALID'


def test_backend_and_work_budget(runtime):
    from vibeweights.platform import CPUBackend
    class CountBackend(CPUBackend):
        def __init__(self): self.calls=[]
        def forward(self,expert,tensor): self.calls.append(expert.spec.id); return super().forward(expert,tensor)
    backend=CountBackend(); engine=Engine(runtime.registry,backend); plan=make_plan()
    r=engine.run(plan,{'X':7,'Y':6,'THRESHOLD':30},work_budget=62)
    assert r.status=='BUDGET' and r.parameter_visits==54 and r.steps==8
    r=engine.run(plan,{'X':7,'Y':6,'THRESHOLD':30},work_budget=63)
    assert r.verified and r.parameter_visits==63 and backend.calls[-9:]==list(r.trace)
    assert engine.run(plan,{'X':7,'Y':6,'THRESHOLD':30},work_budget=True).status=='INVALID'
    assert engine.run(plan,{'X':7,'THRESHOLD':30}).status=='INVALID'
