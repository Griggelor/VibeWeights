from pathlib import Path
import itertools,json
import pytest,torch
from vibeweights.abi import State, Slot as S, ContractError
from vibeweights.weights import load,build,sha,SPECS
from vibeweights.runtime import Runtime,Status,verbalize
from vibeweights.memory import Memory
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def pool(): return load(ROOT)

def test_reproducibility(tmp_path):
    build(tmp_path)
    for name in SPECS:
        f=f'artifacts/weights/{name}.safetensors'
        assert sha(tmp_path/f)==sha(ROOT/f)
    assert sum(sum(p.numel() for p in e.parameters()) for e in load(tmp_path).values())==43

def test_corrupted_artifact(tmp_path):
    build(tmp_path); p=tmp_path/'artifacts/weights/arith.step.safetensors'; p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ContractError): load(tmp_path)

@pytest.mark.parametrize('bad',[True,-1,32,1.5,float('nan')])
def test_invalid_slot(bad):
    with pytest.raises(ContractError): State().change({S.X:bad},[S.X])

def test_mask_and_permissions():
    state=State(); s=state.change({S.X:0},[S.X]); assert s.get(S.X)==0 and not state.valid[S.X]
    with pytest.raises(ContractError): state.get(S.X)
    with pytest.raises(ContractError): state.change({S.Y:1},[S.X])
    with pytest.raises(ContractError): State(version='9')
    with pytest.raises(ContractError): state.apply([S.X],torch.tensor([0.5]))
    with pytest.raises(ContractError): state.apply([S.X],torch.tensor([float('inf')]))
    with pytest.raises(ContractError): state.apply([S.X],torch.tensor([[1.]]))

def test_exhaustive_comparison(pool):
    # Independent integer oracle over every ABI-permitted pair.
    a=torch.arange(962); x,y=torch.meshgrid(a,a,indexing='ij'); inputs=torch.stack((x.flatten(),y.flatten()),dim=-1).float()
    actual=pool['logic.ge'](inputs).flatten(); assert torch.equal(actual,(x>=y).flatten().float())

def test_all_products_production_path(pool):
    rt=Runtime(pool)
    for x,y in itertools.product(range(32),repeat=2):
        expected=x*y; threshold=(x*71+y*17)%962
        r=rt.run(x,threshold,y=y)
        assert r.status==Status.OK,(x,y,r)
        assert r.state.get(S.PRODUCT)==expected and r.state.get(S.GE)==int(expected>=threshold)
        assert r.trace.count('arith.step')==y and r.steps==y+3

@pytest.mark.parametrize('x,y,t',[(0,31,0),(31,31,961),(31,31,960),(1,0,1),(3,7,22),(7,3,21)])
def test_boundary_and_verbalizer(pool,x,y,t):
    r=Runtime(pool).run(x,t,y=y); assert r.status==Status.OK
    assert r.state.get(S.PRODUCT)==x*y
    assert verbalize(r)==f'Das Produkt ist {x*y}. Es ist {"mindestens" if x*y>=t else "kleiner als"} {t}.'

def test_budget_and_invalid(pool):
    r=Runtime(pool).run(3,5,y=7,budget=4); assert r.status==Status.BUDGET and r.steps==4 and not r.state.valid[S.PRODUCT]
    assert Runtime(pool).run(3,5,y=7,key=1).status==Status.INVALID
    assert Runtime(pool).run(3,5,y=7,budget=True).status==Status.INVALID

def test_weight_ablation(tmp_path):
    p=load(ROOT); e=p['arith.step']; e.w0.data[0,1]=0
    r=Runtime(p).run(7,30,y=6); assert r.status==Status.OK and r.state.get(S.PRODUCT)==0 and r.state.get(S.GE)==0
    p=load(ROOT); p['arith.step'].b0.data[1]=0
    r=Runtime(p).run(7,30,y=6); assert r.status==Status.STAGNATION
    p=load(ROOT); p['logic.ge'].w1.data.zero_(); r=Runtime(p).run(7,30,y=6); assert r.state.get(S.GE)==0

def test_expert_swap(pool):
    p=dict(pool); p['arith.step']=load(ROOT)['arith.step']; assert Runtime(p).run(8,56,y=7)==Runtime(pool).run(8,56,y=7)

def test_memory_persistence_and_versions(pool,tmp_path):
    path=tmp_path/'db.sqlite'; m=Memory(path); original=m.write(42,1,'original'); m.write(19,31,'distractor'); latest=m.write(42,6,'revision'); m.close()
    m=Memory(path)
    try:
        assert m.read(original,42)==1 and m.search(42)==latest
        with pytest.raises(ContractError): m.read(latest,19)
        for _ in range(2):
            r=Runtime(pool,m).run(7,30,key=42); assert r.status==Status.OK and r.state.get(S.PRODUCT)==42 and r.state.get(S.RECORD)==latest
            assert not r.state.valid[S.Y] and r.trace.count('READ')==1 and r.trace.count('EVICT')==1 and r.steps==14
        assert m.read(latest,42)==6
        assert Runtime(pool,m).run(7,30,key=100).status==Status.MEMORY
    finally: m.close()

def test_memory_permutation(pool,tmp_path):
    results=[]
    for i,entries in enumerate([[(42,6),(19,31),(10,0)],[(10,0),(19,31),(42,6)]]):
        m=Memory(tmp_path/f'{i}.db')
        try:
            for k,v in entries: m.write(k,v,'source')
            r=Runtime(pool,m).run(7,30,key=42); results.append((r.status,r.state.get(S.PRODUCT),r.state.get(S.GE),r.trace))
        finally: m.close()
    assert results[0]==results[1]

def test_memory_action_weight_dependency():
    p=load(ROOT); p['memory.action'].w0.data.zero_(); p['memory.action'].b0.data.zero_()
    m=Memory(':memory:')
    try:
        m.write(42,6,'source'); r=Runtime(p,m).run(7,30,key=42); assert r.status==Status.INVALID and 'SEARCH' not in r.trace
    finally: m.close()

def test_wrong_memory_action_rejected():
    p=load(ROOT); p['memory.action'].w0.data.zero_(); p['memory.action'].b0.data.copy_(torch.tensor([0.,1.,0.,0.,0.]))
    m=Memory(':memory:')
    try: assert Runtime(p,m).run(7,30,key=42).status==Status.INVALID
    finally: m.close()

def test_indexed_boundaries_and_budget(pool):
    m=Memory(':memory:')
    try:
        for y in range(32): m.write(y+1,y,'factor')
        rt=Runtime(pool,m)
        for y in range(32):
            r=rt.run(31,961,key=y+1); assert r.status==Status.OK and r.state.get(S.PRODUCT)==31*y and not r.state.valid[S.Y]
        r=rt.run(31,961,key=32,budget=2); assert r.status==Status.BUDGET and not r.state.valid[S.PRODUCT]
        assert verbalize(r)=='Kein Ergebnis (BUDGET).'
    finally: m.close()

def test_output_only_formats_state(pool):
    from dataclasses import replace
    r=Runtime(pool).run(7,30,y=6)
    altered=r.state.change({S.PRODUCT:11,S.GE:0},[S.PRODUCT,S.GE])
    assert verbalize(replace(r,state=altered))=='Das Produkt ist 11. Es ist kleiner als 30.'

def test_closed_memory_fails(pool):
    m=Memory(':memory:'); m.close(); assert Runtime(pool,m).run(7,30,key=42).status==Status.MEMORY
