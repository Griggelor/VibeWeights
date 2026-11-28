"""Atomic verified REMEMBER and historical run receipt on the source ledger.

Single trusted owner; catalog safety changes retain their own transactions.
No automatic learning admission, external event-ledger dual write or retry.
"""
import json
import uuid
from .abi import ContractError
from .environment import Environment, identifier
from .knowledge_learning import _hash
from .population import RESIDENT


class MemoryRunTransaction:
    def __init__(self, learning):
        self.learning = learning
        self.db = learning.knowledge.db
        self.db.execute('''CREATE TABLE IF NOT EXISTS memory_runs_v08(
          id TEXT PRIMARY KEY, receipt TEXT NOT NULL, sha TEXT NOT NULL)''')

    def receipt(self, run_id):
        identifier(run_id)
        row = self.db.execute('SELECT receipt,sha FROM memory_runs_v08 WHERE id=?',(run_id,)).fetchone()
        if row is None:
            raise ContractError('unknown memory run')
        receipt = json.loads(row[0])
        if _hash(receipt) != row[1]:
            raise ContractError('memory run receipt integrity')
        return receipt

    def _write(self, result, access, **fields):
        l = self.learning
        scope = fields['scope']
        # Conservative overapproximation: all proposals and all loaded records
        # in this run, not only the final expert's current working set.
        if any(f['scope'] not in ('global',scope) for f in result['facts']) or any(
                h['scope'] not in ('global',scope) for h in result['hypotheses']):
            raise ContractError('memory run scope cannot be widened')
        roots = set(fields['parents'])
        for entry in result['memory_trace']:
            roots.update(entry[4])
        direct = {h['source'] for h in result['hypotheses']}
        pending = list(direct)
        seen = set()
        bindings = []
        while pending:
            expert = pending.pop()
            if expert in seen:
                continue
            if len(seen) >= 512:
                raise ContractError('memory expert provenance bound')
            m = l.catalog.manifest(expert)
            if expert in direct and m['state'] not in RESIDENT:
                raise ContractError('memory contributor not resident')
            if m['state'] not in RESIDENT | {'ARCHIVED'}:
                raise ContractError('memory contributor dependency invalid')
            seen.add(expert)
            bindings.append({'expert':expert,'sha256':m['sha256']})
            if 'learning_job' in m:
                j = l.job(m['learning_job'])
                l._check_sources(j)
                roots.update(r['id'] for r in j['spec']['sources'])
            pending.extend(m['parents'])
        if len(roots) > 64:
            raise ContractError('memory source parent bound')
        for rid in roots:
            l.knowledge.current_verified(rid,access)
        fields['parents'] = tuple(sorted(roots))
        record = l.knowledge._store_locked(access,**fields)
        # Preserve the exact core provenance in the historical run receipt.
        result['memory_core_provenance'] = sorted(bindings,key=lambda x:x['expert'])
        return record

    def save(self, result, run_id):
        """Environment ledger protocol; no independent commit."""
        receipt = dict(result,run_id=run_id)
        raw = json.dumps(receipt,sort_keys=True,allow_nan=False)
        self.db.execute('INSERT INTO memory_runs_v08 VALUES(?,?,?)',(run_id,raw,_hash(receipt)))
        return run_id

    def execute(self, experts, inputs, goal, validators, *, run_id=None, **options):
        l = self.learning
        if l._guard_depth != 1 or l._locked_depth != 1 or not self.db.in_transaction:
            raise ContractError('memory run requires outer source transaction')
        run_id = uuid.uuid4().hex if run_id is None else identifier(run_id)
        if len(run_id)>100:
            raise ContractError('memory run identity bound')
        if self.db.execute('SELECT 1 FROM memory_runs_v08 WHERE id=?',(run_id,)).fetchone():
            raise ContractError('memory run already recorded; read historical receipt')
        result = Environment(experts,validators,self).run(inputs,goal,
            _remember_writer=self._write,_run_id=run_id,**options)
        # One commit contains the verified record, head, index events, and receipt.
        # Any exception before it is rolled back by the outer source guard.
        self.db.commit()
        return result
