"""Local application service, minimal result-only decoder boundary and session journal."""
from dataclasses import dataclass
from pathlib import Path
import sqlite3,json,re
from .abi import ContractError,Slot
from .arithmetic_plan import PlatformRuntime
from .language_weights import LexicalWeights,WeightedInput,WeightedOutput
from .memory import Memory
from .learning import LearningLedger

@dataclass(frozen=True)
class FinalView:
    status: str
    product: int | None = None
    ge: int | None = None
    threshold: int | None = None
    task_id: str = "bounded.multiply_compare"
    @classmethod
    def from_result(cls,result):
        if result.status.name!='OK': return cls(result.status.name)
        return cls('OK',*(result.state.get(s) for s in (Slot.PRODUCT,Slot.GE,Slot.THRESHOLD)))
    @classmethod
    def from_execution(cls,task_id,execution):
        if not execution.verified or execution.status!='OK': return cls(execution.status,task_id=task_id)
        return cls('OK',*(execution.frame.get(n) for n in ('PRODUCT','GE','THRESHOLD')),task_id=task_id)

class SessionJournal:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=2)
        self.db.execute('CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, session TEXT NOT NULL, request TEXT NOT NULL, response TEXT NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS session_turn ON turns(session,id)')
    @staticmethod
    def check(session):
        if type(session) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',session): raise ContractError('invalid session id')
    def append(self,session,request,response):
        self.check(session)
        with self.db: self.db.execute('INSERT INTO turns(session,request,response) VALUES(?,?,?)',(session,request,json.dumps(response,ensure_ascii=False)))
    def history(self,session,limit=20):
        self.check(session)
        if type(limit) is not int or not 1<=limit<=100: raise ContractError('history limit')
        rows=self.db.execute('SELECT request,response FROM turns WHERE session=? ORDER BY id DESC LIMIT ?',(session,limit)).fetchall()
        return [{'request':request,'response':json.loads(response)} for request,response in reversed(rows)]
    def close(self): self.db.close()

class BaselineApplication:
    def __init__(self,root,*,memory_path=':memory:',journal_path=':memory:'):
        self.memory=Memory(memory_path)
        try:
            self.runtime=PlatformRuntime.load(root,self.memory); weights=LexicalWeights.load(root)
            self.encoder=WeightedInput(weights); self.decoder=WeightedOutput(weights); self.journal=SessionJournal(journal_path)
            self.router=self.runtime.task_router()
        except Exception:
            self.memory.close(); raise
    def close(self):
        try: self.journal.close()
        finally: self.memory.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def ask(self,text,*,session='default',budget=128,work_budget=100_000_000):
        self.journal.check(session)
        if type(text) is not str or len(text)>256: raise ContractError('invalid language input')
        result=None
        try:
            request=self.encoder.encode_task(text)
            result=self.router.run(request,budget=budget,work_budget=work_budget)
            view=FinalView.from_execution(request.task_id,result)
            response={'status':view.status,'output':self.decoder.decode(view),'verified':view.status=='OK','steps':result.steps,'trace':list(result.trace),'task':request.task_id,'parameter_visits':result.parameter_visits}
        except ContractError as e:
            response={'status':'INVALID','output':'Kein Ergebnis (INVALID).','verified':False,'steps':result.steps if result is not None else 0,'trace':list(result.trace) if result is not None else [],'error':str(e),'parameter_visits':result.parameter_visits if result is not None else 0}
        self.journal.append(session,text,response)
        return response


class Application:
    """Default v0.4 path; v0.3 remains an explicit comparison reference."""
    def __init__(self,root,*,memory_path=':memory:',journal_path=':memory:',cooperation_path=None):
        from .coalition import Cooperation
        from .coalition_arithmetic import ArithmeticCoalitions
        self.memory=Memory(memory_path)
        try:
            self.cooperation=Cooperation(journal_path if cooperation_path is None else cooperation_path)
            self.runtime=ArithmeticCoalitions(root,self.memory,self.cooperation)
            weights=LexicalWeights.load(root); self.encoder=WeightedInput(weights); self.decoder=WeightedOutput(weights)
            self.journal=SessionJournal(journal_path)
            self.learning=LearningLedger(self.journal.db)
        except Exception:
            if hasattr(self,'cooperation'): self.cooperation.close()
            self.memory.close(); raise
    def close(self):
        try: self.journal.close()
        finally:
            try: self.cooperation.close()
            finally: self.memory.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def ask(self,text,*,session='default',budget=128,work_budget=100_000_000,recall=True,shots=0,shot_seed=0,parent_answer_id=None):
        self.journal.check(session)
        if parent_answer_id is not None: self.learning.answer(session,parent_answer_id)
        if type(text) is not str or len(text)>256 or type(recall) is not bool: raise ContractError('invalid language input')
        result=None
        try:
            # Explicit semicolon-separated independent goal scopes; not general language decomposition.
            requests=tuple(self.encoder.encode_task(part.strip()) for part in text.split(';'))
            results=self.runtime.run_groups(requests,session=session,budget=budget,work_budget=work_budget,recall=recall,shots=shots,shot_seed=shot_seed)
            result=results[0]
            def project(request,result):
                view=FinalView.from_execution(request.task_id,result)
                try:
                    output=self.decoder.decode(view); status=view.status; error=result.error
                except ContractError as exc:
                    output='Kein Ergebnis (INVALID).'; status='INVALID'; error=str(exc)
                return {'status':status,'output':output,'verified':status=='OK','core_verified':view.status=='OK','steps':result.steps,'trace':list(result.trace),'task':request.task_id,'parameter_visits':result.parameter_visits,'head_visits':result.head_visits,'members':list(result.members),'contributors':list(result.contributors),'events':result.event_dicts(),'error':error}
            groups=[project(q,r) for q,r in zip(requests,results)]
            if len(groups)==1: response=groups[0]
            else:
                ok=all(g['verified'] for g in groups); status='OK' if ok else next(g['status'] for g in groups if not g['verified'])
                response={'status':status,'verified':ok,'output':' '.join(g['output'] for g in groups) if ok else f'Kein Gesamtergebnis ({status}).','groups':groups,'steps':sum(g['steps'] for g in groups),'parameter_visits':sum(g['parameter_visits'] for g in groups),'head_visits':sum(g['head_visits'] for g in groups),'events':sorted((e for g in groups for e in g['events']),key=lambda e:e['seq'])}
            response['expert_ids']={i+1:n for i,n in enumerate(sorted(self.runtime.registry.experts))}
            response['exploration']={'shots_budget':shots,'seed':shot_seed}
        except ContractError as e:
            response={'status':'INVALID','output':'Kein Ergebnis (INVALID).','verified':False,'steps':result.steps if result is not None else 0,'trace':list(result.trace) if result is not None else [],'error':str(e),'parameter_visits':result.parameter_visits if result is not None else 0}
        identities={i+1:p.identity for i,p in enumerate(sorted(self.runtime.engine.participants,key=lambda p:p.id))}
        with self.journal.db:
            self.learning.record(session,text,response,identities,parent_answer_id)
            self.journal.db.execute('INSERT INTO turns(session,request,response) VALUES(?,?,?)',(session,text,json.dumps(response,ensure_ascii=False)))
        return response

    def feedback(self,answer_id,kind,note='',*,session='default'):
        self.journal.check(session)
        return self.learning.feedback(session,answer_id,kind,note)

    def metrics(self,*,session='default'):
        self.journal.check(session)
        return self.learning.metrics(session)

    def reflections(self,*,session='default'):
        self.journal.check(session)
        return self.learning.pending(session)
