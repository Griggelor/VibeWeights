"""Bounded symbolic reference language boundary; not learned language understanding.
Only the input adapter sees the request string. The output adapter receives the
result, not the original prompt or the runtime/pool. No task arithmetic here.
"""
from dataclasses import dataclass
from typing import Protocol
import re
from .abi import Slot, checked, ContractError
from .runtime import Result, Runtime, verbalize

@dataclass(frozen=True)
class Request:
    x: int
    threshold: int
    y: int | None = None
    key: int | None = None

    def __post_init__(self):
        checked(Slot.X, self.x)
        checked(Slot.THRESHOLD, self.threshold)
        if (self.y is None) == (self.key is None):
            raise ContractError('exactly one operand source required')
        if self.y is not None:
            checked(Slot.Y, self.y)
        else:
            checked(Slot.KEY, self.key)

class InputEncoder(Protocol):
    def encode(self, text: str) -> Request: ...

class OutputDecoder(Protocol):
    def decode(self, result: Result) -> str: ...

class ReferenceInput:
    origin = 'SYMBOLIC'
    _forms = (
        (re.compile(r'Multipliziere ([0-9]{1,2}) mit ([0-9]{1,2}) und vergleiche mit ([0-9]{1,3})[.!]?', re.IGNORECASE), False),
        (re.compile(r'Ist ([0-9]{1,2}) mal ([0-9]{1,2}) mindestens ([0-9]{1,3})\?', re.IGNORECASE), False),
        (re.compile(r'Multipliziere ([0-9]{1,2}) mit dem Wert zu Schlüssel ([0-9]{1,8}) und vergleiche mit ([0-9]{1,3})[.!]?', re.IGNORECASE), True),
    )
    def encode(self, text):
        if type(text) is not str or len(text) > 256:
            raise ContractError('invalid language input')
        for form, indexed in self._forms:
            match = form.fullmatch(text.strip())
            if match:
                x, operand, threshold = map(int, match.groups())
                return Request(x, threshold, **({'key': operand} if indexed else {'y': operand}))
        raise ContractError('unsupported language request; no guessed interpretation')

class ReferenceOutput:
    origin = 'SYMBOLIC'
    def decode(self, result):
        return verbalize(result)

class LanguageSession:
    def __init__(self, runtime: Runtime, encoder: InputEncoder, decoder: OutputDecoder):
        self.runtime, self.encoder, self.decoder = runtime, encoder, decoder

    def run(self, text: str, *, budget: int = 48) -> tuple[Result, str]:
        request = self.encoder.encode(text)
        result = self.runtime.run(request.x, request.threshold, y=request.y, key=request.key, budget=budget)
        return result, self.decoder.decode(result)
