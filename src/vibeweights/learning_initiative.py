"""Bounded model-initiated learning, not autonomous authorization.
The host supplies approved gaps, source rights, teacher and independent oracle.
One model proposal per cycle; one selected learning job; explicit promotion.
"""
from dataclasses import asdict, dataclass
import json
from .abi import ContractError
from .bootstrap import BootstrapExpert
from .environment import identifier
from .knowledge_learning import Domain, _hash
from .relation_learning import stage_teacher_knowledge


@dataclass(frozen=True)
class Opportunity:
    id: str
    domain: Domain
    description: str
    teacher_request: str
    oracle_id: str
    method: str = 'TRAINED'
    steps: int = 80

    def __post_init__(self):
        identifier(self.id); identifier(self.oracle_id)
        if not isinstance(self.domain, Domain): raise ContractError('initiative domain')
        for value, bound in ((self.description, 512), (self.teacher_request, 8000)):
            if type(value) is not str or not value.strip() or len(value)>bound:
                raise ContractError('initiative text bounds')
        if self.method not in ('DIRECT','TRAINED') or type(self.steps) is not int or not 1<=self.steps<=256:
            raise ContractError('initiative learning budget')


def _choice(raw, eligible):
    def no_duplicates(pairs):
        result={}
        for k,v in pairs:
            if k in result: raise ContractError('duplicate initiative key')
            result[k]=v
        return result
    if type(raw) is not str or len(raw)>1024: raise ContractError('initiative proposal bounds')
    try: obj=json.loads(raw,object_pairs_hook=no_duplicates)
    except (ValueError,TypeError,RecursionError) as exc: raise ContractError('invalid initiative JSON') from exc
    if type(obj) is not dict: raise ContractError('initiative proposal object')
    if obj=={'action':'SKIP'}: return None
    if set(obj)!={'action','target'} or obj['action']!='LEARN' or type(obj['target']) is not str or obj['target'] not in eligible:
        raise ContractError('initiative target not eligible')
    return obj['target']


class LearningInitiative:
    """Trusted single-owner host adapter. A model can select or decline a gap.

    Gaps are currently entirely new, host-approved numeric domains, detected
    from the knowledge and catalog namespaces; not arbitrary semantic deficits.
    Stored decisions bind the exact opportunity policy and are replayable.
    No hidden teacher request generation, no oracle access by the proposer.
    """
    def __init__(self, learning):
        self.learning=learning; self.db=learning.db
        self.db.execute('''CREATE TABLE IF NOT EXISTS learning_initiatives_v08(
          cycle TEXT PRIMARY KEY, policy_sha TEXT NOT NULL, decision TEXT NOT NULL,
          receipt TEXT NOT NULL)''')
        self.db.commit()

    def _policy(self, cycle, opportunities):
        identifier(cycle); identifier('teacher:initiative:'+cycle); identifier('expert:'+cycle)
        ops=tuple(opportunities)
        if not 1<=len(ops)<=8 or any(not isinstance(x,Opportunity) for x in ops):
            raise ContractError('initiative opportunity bounds')
        if len({x.id for x in ops})!=len(ops) or len({x.domain.topic for x in ops})!=len(ops):
            raise ContractError('duplicate initiative opportunity')
        mapping={x.id:x for x in ops}
        return mapping,_hash([asdict(mapping[k]) for k in sorted(mapping)])

    def _eligible(self, op):
        l=self.learning; topic=op.domain.topic
        if l.knowledge.db.execute('SELECT 1 FROM knowledge_v1 WHERE scope=? AND topic=? LIMIT 1',('global',topic)).fetchone(): return False
        # All existing variants, including archived/staged/quarantined, reserve
        # their domain. Correction and extension require a different contract.
        if any(m['family']==topic for m in l.catalog.list()): return False
        for row in l.db.execute('SELECT spec FROM knowledge_jobs_v08'):
            if json.loads(row[0])['domain']['topic']==topic: return False
        return not any(x['spec']['domain']['topic']==topic for x in l.intakes())

    def receipt(self, cycle):
        row=self.db.execute('SELECT receipt FROM learning_initiatives_v08 WHERE cycle=?',(cycle,)).fetchone()
        if row is None: raise ContractError('unknown initiative cycle')
        return json.loads(row[0])

    def propose(self, cycle, opportunities, proposer):
        mapping,policy=self._policy(cycle,opportunities)
        if not isinstance(proposer,BootstrapExpert): raise ContractError('initiative proposer')
        with self.learning._guard():
            prior=self.db.execute('SELECT policy_sha,receipt FROM learning_initiatives_v08 WHERE cycle=?',(cycle,)).fetchone()
            if prior:
                if prior[0]!=policy: raise ContractError('initiative policy changed')
                return json.loads(prior[1])
            eligible={k:v for k,v in mapping.items() if self._eligible(v)}
        # Do not keep SQLite write lock across potentially slow model I/O.
        if eligible:
            prompt=json.dumps({'instruction':'Choose one approved missing knowledge domain worth learning, or decline. Return only JSON {"action":"LEARN","target":"id"} or {"action":"SKIP"}.',
                'opportunities':[{'id':k,'description':v.description,'observed_gap':'NO_DOMAIN_EXPERT_OR_FACTS'} for k,v in sorted(eligible.items())]},ensure_ascii=False)
            result=proposer.propose(prompt)
            if not isinstance(result,dict) or result.get('verified') is not False: raise ContractError('initiative unverified proposal required')
            selected=_choice(result.get('proposal'),eligible)
            raw_hash=_hash(result['proposal'])
        else: selected=None; raw_hash=None
        with self.learning._guard():
            # Recheck after the model returns; never execute a stale selection.
            if selected is not None and not self._eligible(mapping[selected]): raise ContractError('initiative opportunity no longer eligible')
            receipt={'cycle':cycle,'policy_sha256':policy,'target':selected,'eligible':sorted(eligible),
                     'action':'LEARN' if selected is not None else 'SKIP',
                     'proposer':proposer.spec.id,'model':proposer.spec.model,'proposal_sha256':raw_hash,
                     'job_id':'initiative:'+cycle if selected is not None else None,
                     'expert_id':'expert:'+cycle if selected is not None else None}
            with self.db:
                self.db.execute('INSERT INTO learning_initiatives_v08 VALUES(?,?,?,?)',
                    (cycle,policy,receipt['action'],json.dumps(receipt,sort_keys=True)))
        return receipt

    def stage(self, cycle, opportunities, teacher, *, oracles):
        """Host explicitly authorizes staging with independent oracle registry.

        Resume a committed intake rather than re-call the teacher. Before an
        intake commits, an explicit retry may call the teacher again. Never
        promotes; no autonomous approval or hidden retry loop.
        """
        mapping,policy=self._policy(cycle,opportunities); receipt=self.receipt(cycle)
        if receipt['policy_sha256']!=policy: raise ContractError('initiative policy changed')
        if receipt['action']=='SKIP': return None
        op=mapping[receipt['target']]; l=self.learning
        oracle=oracles.get(op.oracle_id)
        if not callable(oracle) or not isinstance(teacher,BootstrapExpert): raise ContractError('independent oracle/teacher required')
        job,expert=receipt['job_id'],receipt['expert_id']
        with l._guard():
            pending=[x for x in l.intakes() if x['job_id']==job]
            if pending:
                spec=pending[0]['spec']; derivation=spec['derivation']
                if (pending[0]['expert_id']!=expert or spec['domain']!=asdict(op.domain)
                    or spec['method']!=op.method or spec['steps']!=op.steps
                    or not isinstance(derivation,dict) or derivation.get('initiative_sha256')!=_hash(receipt)):
                    raise ContractError('initiative intake mismatch')
                return l.resume_intake(job)
            if not self._eligible(op): raise ContractError('initiative opportunity no longer eligible')
            if l.db.execute('SELECT 1 FROM experts WHERE id=?',(expert,)).fetchone() or l.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job,expert)).fetchone():
                raise ContractError('initiative identity conflict')
        manifest,_=stage_teacher_knowledge(l,teacher,domain=op.domain,request=op.teacher_request,
            job_id=job,expert_id=expert,oracle=oracle,oracle_id=op.oracle_id,
            method=op.method,steps=op.steps,initiative_sha256=_hash(receipt))
        return manifest
