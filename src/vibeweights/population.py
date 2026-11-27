"""Opt-in resident population for trusted host code, not a cognitive router.
One owner per catalog while running; no physical deletion, no RAM cache tier.
CPU is a reference emulation of VRAM residency, not a GPU measurement.
"""
from dataclasses import asdict, dataclass
import json
from .abi import ContractError
from .environment import Environment, identifier

ACTIVE = 'ACTIVE'
LOW_ACTIVE_IMPORTANT = 'LOW_ACTIVE_IMPORTANT'
ARCHIVED = 'ARCHIVED'
RESIDENT = frozenset((ACTIVE, LOW_ACTIVE_IMPORTANT))
STATES = RESIDENT | {ARCHIVED}


@dataclass(frozen=True)
class SelectionEvidence:
    """Host-attested evaluation; NOT generated or independently verified here."""
    report_id: str
    observations: int
    low_activity: bool
    important: bool
    coverage: bool
    ablation_ok: bool
    dependencies_clear: bool
    beneficial: bool

    def __post_init__(self):
        identifier(self.report_id)
        if type(self.observations) is not int or self.observations < 1:
            raise ContractError('selection observations')
        for name in ('low_activity', 'important', 'coverage', 'ablation_ok',
                     'dependencies_clear', 'beneficial'):
            if type(getattr(self, name)) is not bool:
                raise ContractError('selection evidence must be boolean')


class Population:
    """Own resident affine tensors; explicit selection, never automatic eviction.

    Call between inference runs from one trusted owner. The catalog is persistent;
    resident objects are reconstructed on restart. A byte budget counts tensor
    payloads, not allocator overhead, temporary tensors or other processes.
    Legacy direct Catalog APIs are not a security or resource isolation boundary.
    """
    def __init__(self, catalog, *, tensor_budget, artifact_budget=64*1024*1024,
                 min_observations=1, device='cpu'):
        for value in (tensor_budget, artifact_budget, min_observations):
            if type(value) is not int or value < 1:
                raise ContractError('invalid population budget or policy')
        self.catalog = catalog
        self.tensor_budget = tensor_budget
        self.artifact_budget = artifact_budget
        self.min_observations = min_observations
        self.device = device
        self._resident = {}
        self._closed = False
        self.catalog.db.execute('''CREATE TABLE IF NOT EXISTS selection_events(
            expert_id TEXT NOT NULL, revision INTEGER NOT NULL, old_state TEXT NOT NULL,
            new_state TEXT NOT NULL, report TEXT NOT NULL,
            PRIMARY KEY(expert_id, revision))''')
        try:
            for m in self.catalog.list():
                if m['state'] in RESIDENT:
                    self._resident[m['id']] = self._prepare(m, experimental=False)
        except BaseException:
            self.close()
            raise

    @property
    def tensor_bytes(self):
        return sum(t.numel()*t.element_size() for e in self._resident.values()
                   for t in (e.w, e.b))

    def _check(self):
        if self._closed:
            raise ContractError('population closed')
        expected = {m['id'] for m in self.catalog.list() if m['state'] in RESIDENT}
        if expected != set(self._resident):
            raise ContractError('catalog changed outside population; reopen')

    def _prepare(self, manifest, *, experimental):
        # Affine catalog stores FP32 parameters; file bytes have a separate limit.
        needed = 4*manifest['parameters']
        if needed + self.tensor_bytes > self.tensor_budget:
            raise ContractError('resident tensor budget; no automatic eviction')
        return self.catalog.load(manifest['id'], byte_budget=self.artifact_budget,
                                 experimental=experimental, device=self.device)

    def resident_ids(self):
        self._check()
        return tuple(sorted(self._resident))

    def compute(self, id, values):
        self._check()
        if id not in self._resident:
            raise ContractError('expert is not resident')
        return self._resident[id].compute(values)

    def run(self, inputs, goal, validators, *, ledger=None, **options):
        """Numerical runtime sees only resident, approved experts."""
        self._check()
        return Environment(tuple(self._resident.values()), validators, ledger).run(inputs, goal, **options)

    def admit(self, id, cases, *, learning_job=None):
        """Add only; a full budget leaves the candidate STAGED."""
        self._check()
        m = self.catalog.manifest(id)
        if m['state'] != 'STAGED':
            raise ContractError('not staged')
        expert = self._prepare(m, experimental=True)
        if not self.catalog.admit(id, cases, learning_job=learning_job):
            return False
        self._resident[id] = expert
        return True

    def revision(self, id):
        self.catalog.manifest(id)
        return self.catalog.db.execute('SELECT COALESCE(MAX(revision),0) FROM selection_events WHERE expert_id=?', (id,)).fetchone()[0]

    def history(self, id):
        self.catalog.manifest(id)
        return [dict(revision=n, old_state=a, new_state=b, evidence=json.loads(r))
                for n,a,b,r in self.catalog.db.execute('SELECT revision,old_state,new_state,report FROM selection_events WHERE expert_id=? ORDER BY revision',(id,))]

    def select(self, id, target, *, expected_revision, evidence):
        """Separate host selection operation; validates evidence gates and CAS.
        No claim that evidence supplied by a caller is objectively true.
        """
        self._check()
        if not isinstance(evidence, SelectionEvidence) or evidence.observations < self.min_observations:
            raise ContractError('insufficient selection evidence')
        if type(expected_revision) is not int or expected_revision < 0 or target not in STATES:
            raise ContractError('invalid selection target or revision')
        m = self.catalog.manifest(id)
        old = m['state']
        if old not in STATES or old == target or self.revision(id) != expected_revision:
            raise ContractError('invalid or stale selection')
        if target == ARCHIVED and (evidence.important or not all((evidence.coverage, evidence.ablation_ok, evidence.dependencies_clear))):
            raise ContractError('archive evidence insufficient')
        if target == LOW_ACTIVE_IMPORTANT and not (evidence.low_activity and evidence.important):
            raise ContractError('rare-important evidence required')
        if old == ARCHIVED and not evidence.beneficial:
            raise ContractError('reactivation evidence required')
        expert = self._prepare(m, experimental=True) if old == ARCHIVED else None
        db = self.catalog.db
        with db:
            db.execute('BEGIN IMMEDIATE')
            if self.revision(id) != expected_revision:
                raise ContractError('stale selection revision')
            if db.execute('UPDATE experts SET state=? WHERE id=? AND state=?',(target,id,old)).rowcount != 1:
                raise ContractError('stale expert state')
            db.execute('INSERT INTO selection_events VALUES(?,?,?,?,?)',
                       (id, expected_revision+1, old, target, json.dumps(asdict(evidence),sort_keys=True)))
        if target == ARCHIVED:
            del self._resident[id]
        elif expert is not None:
            self._resident[id] = expert
        return expected_revision+1

    def quarantine(self, ids, *, reason):
        """Independent safety withdrawal, never an evolutionary archive/delete.

        Trusted host supplies a reason. Quarantined artifacts cannot be reactivated
        via SELECT; reconstruct a new candidate from valid sources instead.
        """
        self._check()
        identifier(reason)
        ids=tuple(dict.fromkeys(ids))
        if not ids: return
        db=self.catalog.db
        with db:
            db.execute('BEGIN IMMEDIATE')
            for id in ids:
                old=self.catalog.manifest(id)['state']
                if old=='QUARANTINED': continue
                if old not in STATES|{'STAGED'}: raise ContractError('invalid quarantine state')
                revision=self.revision(id)+1
                if db.execute('UPDATE experts SET state=? WHERE id=? AND state=?',('QUARANTINED',id,old)).rowcount!=1:
                    raise ContractError('stale quarantine state')
                db.execute('INSERT INTO selection_events VALUES(?,?,?,?,?)',
                           (id,revision,old,'QUARANTINED',json.dumps({'safety_reason':reason})))
        for id in ids: self._resident.pop(id,None)

    def close(self):
        self._resident.clear()
        self._closed = True
