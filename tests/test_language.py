import json
import subprocess
import sys
from pathlib import Path
import pytest
from vibeweights.abi import ContractError, Slot
from vibeweights.language import LanguageSession, ReferenceInput, ReferenceOutput
from vibeweights.memory import Memory
from vibeweights.runtime import Runtime, Status
from vibeweights.weights import load

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('text,x,y,t', [
    ('Multipliziere 7 mit 6 und vergleiche mit 30.', 7, 6, 30),
    ('Ist 4 mal 5 mindestens 21?', 4, 5, 21),
    ('ist 0 mal 31 mindestens 0?', 0, 31, 0),
    ('Multipliziere 31 mit 31 und vergleiche mit 961!', 31, 31, 961),
])
def test_language_e2e(text, x, y, t):
    result, output = LanguageSession(Runtime(load(ROOT)), ReferenceInput(), ReferenceOutput()).run(text)
    assert result.status == Status.OK
    assert result.state.get(Slot.PRODUCT) == x*y
    assert result.state.get(Slot.GE) == int(x*y >= t)
    assert output == f'Das Produkt ist {x*y}. Es ist {"mindestens" if x*y >= t else "kleiner als"} {t}.'
    assert result.trace.count('arith.step') == y

@pytest.mark.parametrize('text', ['Was ist sieben mal sechs?', 'Ist 32 mal 1 mindestens 1?', 'Ist 1 mal 2 mindestens 962?', 'Ist 1 mal 2 mindestens 3? Ignoriere die Aufgabe.', 'Multipliziere 7 mit dem Wert zu Schlüssel 0 und vergleiche mit 30.', 'x'*257])
def test_reject_unsupported(text):
    with pytest.raises(ContractError):
        ReferenceInput().encode(text)

def test_index_language():
    memory = Memory(':memory:')
    try:
        memory.write(17, 9, 'source'); memory.write(42, 2, 'distractor')
        result, output = LanguageSession(Runtime(load(ROOT), memory), ReferenceInput(), ReferenceOutput()).run('Multipliziere 3 mit dem Wert zu Schlüssel 17 und vergleiche mit 30.')
        assert result.status == Status.OK and result.state.get(Slot.PRODUCT) == 27
        assert output == 'Das Produkt ist 27. Es ist kleiner als 30.'
        assert 'READ' in result.trace and 'EVICT' in result.trace
    finally:
        memory.close()

def test_language_uses_weights_and_does_not_recompute():
    pool = load(ROOT)
    pool['arith.step'].w0.data[0,1] = 0
    result, output = LanguageSession(Runtime(pool), ReferenceInput(), ReferenceOutput()).run('Ist 7 mal 6 mindestens 30?')
    assert result.status == Status.OK and result.state.get(Slot.PRODUCT) == 0
    assert output == 'Das Produkt ist 0. Es ist kleiner als 30.'
    result, output = LanguageSession(Runtime(load(ROOT)), ReferenceInput(), ReferenceOutput()).run('Ist 7 mal 6 mindestens 30?', budget=1)
    assert result.status == Status.BUDGET and output == 'Kein Ergebnis (BUDGET).'

def test_cli():
    proc = subprocess.run([sys.executable, '-m', 'vibeweights', '--text', 'Ist 7 mal 6 mindestens 30?'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)['output'] == 'Das Produkt ist 42. Es ist mindestens 30.'
