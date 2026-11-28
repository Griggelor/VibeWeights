"""Task-driven learning for approved numeric domains, not a cognitive router.
Only observed misses in unused domains trigger the existing initiative path.
Host supplies the domain contract, teacher and independent approval. No raw user
text, test labels or teacher instructions enter the task journal or proposer.
"""
from dataclasses import asdict
import json
from .abi import ContractError
from .environment import identifier
from .knowledge_learning import Domain, _hash
from .learning_initiative import LearningInitiative
from .population import RESIDENT
from .conflict_learning import open_case, record_conflict


class TaskLearning:
    def __init__(self, learning):
        self.learning = learning
        self.initiative = LearningInitiative(learning)
        self.db = learning.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS task_observations_v08(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL UNIQUE,
          topic TEXT NOT NULL, domain_sha TEXT NOT NULL, outcome TEXT NOT NULL,
          receipt TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS task_gap_idx_v08 ON task_observations_v08
          (topic,domain_sha,outcome,seq);
        CREATE TABLE IF NOT EXISTS task_learning_cycles_v08(
          cycle TEXT PRIMARY KEY, policy_sha TEXT NOT NULL, plan TEXT NOT NULL);
        ''')

    def observe(self, task, domain, entity):
        """Execute a typed knowledge query against approved resident jobs.

        The caller supplies the task, not its expected answer. Abstention, a
        conflict, a reserved namespace and a missing domain are distinct. A
        runtime/guard error propagates and is never turned into a learning gap.
        An observation is historical, not a verified fact or fresh answer.
        """
        identifier(task)
        if not isinstance(domain, Domain) or type(entity) is not int or not 0 <= entity < domain.entities:
            raise ContractError('task domain/entity contract')
        l = self.learning
        new_conflict=False
        with l._guard():
            if (self.db.execute('SELECT 1 FROM task_observations_v08 WHERE task=?',(task,)).fetchone() or
                l.knowledge.db.execute('SELECT 1 FROM conflict_cases_v08 WHERE task=?',(task,)).fetchone()):
                raise ContractError('duplicate task observation')
            jobs = []
            reserved = False
            for m in l.catalog.list():
                if m['family'] != domain.topic: continue
                reserved = True
                if m['state'] not in RESIDENT or 'learning_job' not in m: continue
                j = l.job(m['learning_job'])
                if j['spec']['domain'] != asdict(domain):
                    raise ContractError('task domain mismatch')
                if j['state'] not in RESIDENT or not j['report'] or not j['report']['accepted']:
                    raise ContractError('task expert not approved')
                jobs.append(j)
            case=open_case(l,domain,entity)
            outputs = {} if case else {j['expert_id']:domain.decode(l.population.compute(j['expert_id'],domain.encode(entity))) for j in jobs}
            known = {v for v in outputs.values() if v is not None}
            if case is not None: outcome, answer = 'CONFLICT', None
            elif len(known) > 1:
                case=record_conflict(l,task,domain,entity,outputs)
                new_conflict=True
                outcome, answer = 'CONFLICT', None
            elif known: outcome, answer = 'ANSWERED', next(iter(known))
            elif jobs: outcome, answer = 'ABSTAINED', None
            else:
                # _eligible is deliberately namespace-wide, including facts,
                # archived experts, staged jobs and pending intakes.
                from .learning_initiative import Opportunity
                probe = Opportunity('probe',domain,'eligibility probe','not sent','probe')
                outcome = 'MISSING_DOMAIN' if not reserved and self.initiative._eligible(probe) else 'RESERVED_DOMAIN'
                answer = None
            receipt = {'task':task,'domain':asdict(domain),'entity':entity,'outcome':outcome,
                       'answer':answer,'experts':[j['expert_id'] for j in jobs]}
            if case is not None:
                receipt['conflict_case']=case['task']
                receipt['outputs']=case['outputs']
            with self.db:
                self.db.execute('INSERT INTO task_observations_v08(task,topic,domain_sha,outcome,receipt) VALUES(?,?,?,?,?)',
                    (task,domain.topic,_hash(asdict(domain)),outcome,json.dumps(receipt,sort_keys=True)))
        if new_conflict: l.audit_sources()
        return receipt

    def observation(self, task):
        row=self.db.execute('SELECT receipt FROM task_observations_v08 WHERE task=?',(task,)).fetchone()
        if row is None:
            row=self.learning.knowledge.db.execute('SELECT receipt FROM conflict_cases_v08 WHERE task=?',(task,)).fetchone()
        if row is None: raise ContractError('unknown task observation')
        return json.loads(row[0])

    def _plan(self, cycle, opportunities, *, create=False):
        mapping, policy = self.initiative._policy(cycle,opportunities)
        with self.learning._guard():
            row=self.db.execute('SELECT policy_sha,plan FROM task_learning_cycles_v08 WHERE cycle=?',(cycle,)).fetchone()
            if row:
                if row[0]!=policy: raise ContractError('task learning policy changed')
                plan=json.loads(row[1])
            else:
                if not create: raise ContractError('unknown task learning cycle')
                if self.db.execute('SELECT 1 FROM learning_initiatives_v08 WHERE cycle=?',(cycle,)).fetchone():
                    raise ContractError('initiative cycle already reserved')
                evidence={}
                for key,op in sorted(mapping.items()):
                    row=self.db.execute('''SELECT task FROM task_observations_v08
                      WHERE topic=? AND domain_sha=? AND outcome='MISSING_DOMAIN'
                      ORDER BY seq DESC LIMIT 1''',(op.domain.topic,_hash(asdict(op.domain)))).fetchone()
                    if row and self.initiative._eligible(op): evidence[key]=row[0]
                plan={'cycle':cycle,'policy_sha256':policy,'evidence':evidence}
                with self.db:
                    self.db.execute('INSERT INTO task_learning_cycles_v08 VALUES(?,?,?)',(cycle,policy,json.dumps(plan,sort_keys=True)))
        return plan,tuple(mapping[k] for k in sorted(plan['evidence']))

    def propose(self, cycle, opportunities, proposer):
        """At most one model selection per completed cycle, over observed gaps.

        Freeze the evidence before model I/O. Empty plan: no model call. A
        crash before the initiative receipt may require explicit retry; after
        the receipt the original selection is reused. No auto-admission.
        """
        plan,ops=self._plan(cycle,opportunities,create=True)
        if not ops: return {'plan':plan,'decision':None}
        return {'plan':plan,'decision':self.initiative.propose(cycle,ops,proposer)}

    def stage(self, cycle, opportunities, teacher, *, oracles):
        plan,ops=self._plan(cycle,opportunities)
        if not ops: return None
        return self.initiative.stage(cycle,ops,teacher,oracles=oracles)
