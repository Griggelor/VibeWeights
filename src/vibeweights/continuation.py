"""Bounded task continuation through locally emitted recruitment events.

Transport only: no expert ranking, route selection or new learning authority.
One trusted owner, same attested oracle on resume, catalog receipts are history.
"""
import json
from dataclasses import asdict
from .abi import ContractError
from .environment import Limits, Signal, identifier
from .group_work import GroupWork
from .knowledge_learning import _hash


class RecruitmentContinuation:
    def __init__(self, work):
        if not isinstance(work,GroupWork):raise ContractError('group work required')
        self.work=work;self.learning=work.learning;self.bus=work.bus;self.db=work.db
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS continuation_sessions_v08(
          id TEXT PRIMARY KEY, spec TEXT NOT NULL, sha TEXT NOT NULL,
          deliveries INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS continuation_runs_v08(
          session TEXT NOT NULL, event TEXT NOT NULL, run TEXT NOT NULL UNIQUE,
          PRIMARY KEY(session,event));
        ''')

    def _session(self,session,spec):
        raw=json.dumps(spec,sort_keys=True,allow_nan=False);sha=_hash(spec)
        with self.db:
            row=self.db.execute('SELECT spec,sha FROM continuation_sessions_v08 WHERE id=?',(session,)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO continuation_sessions_v08(id,spec,sha) VALUES(?,?,?)',(session,raw,sha))
            elif row!=(raw,sha):
                raise ContractError('continuation task identity changed')

    def _stats(self,session):
        deliveries=self.db.execute('SELECT deliveries FROM continuation_sessions_v08 WHERE id=?',(session,)).fetchone()[0]
        runs=self.db.execute('SELECT COUNT(*) FROM continuation_runs_v08 WHERE session=?',(session,)).fetchone()[0]
        return {'runs_reserved':runs,'deliveries_reserved':deliveries}

    def _reserve_run(self,session,event,target,group,max_runs):
        run='continuation:'+_hash([session,event,target,group])
        with self.db:
            row=self.db.execute('SELECT run FROM continuation_runs_v08 WHERE session=? AND event=?',(session,event)).fetchone()
            if row is not None:
                if row!=(run,):raise ContractError('continuation run identity changed')
                return run
            if self._stats(session)['runs_reserved']>=max_runs:return None
            if self.db.execute('SELECT 1 FROM group_runs_v08 WHERE id=?',(run,)).fetchone():
                raise ContractError('unreserved continuation receipt')
            self.db.execute('INSERT INTO continuation_runs_v08 VALUES(?,?,?)',(session,event,run))
        return run

    def _ready(self,event):
        """Check exact event delivery to all currently resident bound heads."""
        l=self.learning
        with l._guard():
            e=self.bus.event(event)
            if not self.bus._live(e,self.bus._conflicts()):return None
            residents=set(l.population.resident_ids())
            eligible={x for (x,) in self.db.execute('SELECT expert FROM perception_bindings_v08') if x in residents}
            responded={x for (x,) in self.bus.db.execute('SELECT expert FROM learning_event_decisions_v08 WHERE event=?',(event,))}
            return len(eligible-responded)

    def _delivery(self,session,event,max_deliveries):
        while True:
            pending=self._ready(event)
            if pending is None:return 'STALE'
            if not pending:return 'READY'
            remaining=max_deliveries-self._stats(session)['deliveries_reserved']
            if remaining<=0:return 'DELIVERY_BUDGET'
            charge=min(pending,remaining,64)
            # Charge before calling the external delivery step. A crash may
            # overcharge, never replenish this persistent whole-session budget.
            with self.db:
                self.db.execute('UPDATE continuation_sessions_v08 SET deliveries=deliveries+? WHERE id=?',(charge,session))
            receipts=self.work.heads.tick(delivery_budget=charge)
            if not any(r['status']=='RECORDED' for r in receipts):return 'WAITING'

    def run(self,session,event,target,group,inputs,goal,validators,*,oracle_id,
            max_runs=16,max_deliveries=128,recruitment_budget=2,recruitment_policy='learned',
            limits=Limits(steps=64,rounds=16,signals=256)):
        """Execute the same task through a bounded breadth-first child-event sweep.

        The original group key is the task-data boundary. A child event cannot
        authorize another group to see the inputs. No goal scores select peers.
        Retrying the same session requires the same task/budgets/oracle identity.
        Validator callables are trusted attestations, not serialized or hashed.
        """
        for x in (session,event,group,goal,oracle_id):identifier(x)
        if target!='explore':raise ContractError('continuation explore target required')
        if type(max_runs) is not int or not 1<=max_runs<=64 or type(max_deliveries) is not int or not 1<=max_deliveries<=1024:
            raise ContractError('continuation budget')
        if type(recruitment_budget) is not int or not 1<=recruitment_budget<=64 or recruitment_policy not in ('typed','learned'):
            raise ContractError('continuation recruitment policy')
        if recruitment_policy=='learned' and self.work.request_heads is None:
            raise ContractError('learned continuation requires request heads')
        if not isinstance(limits,Limits) or limits.steps>1024 or limits.rounds>64 or limits.signals>2048 or limits.probes:
            raise ContractError('continuation limits')
        if type(inputs) is not dict or not 1<=len(inputs)<=64 or goal in inputs:
            raise ContractError('continuation inputs')
        if type(validators) is not dict or not 1<=len(validators)<=64 or goal not in validators or any(not callable(v) for v in validators.values()):
            raise ContractError('continuation validators')
        for topic in validators:identifier(topic)
        inputs={k:Signal('continuation',k,v).values for k,v in inputs.items()}
        spec={'contract':'task-continuation-v1','event':event,'target':target,'group':group,
              'inputs':inputs,'goal':goal,'validator_topics':sorted(validators),'oracle_id':oracle_id,
              'max_runs':max_runs,'max_deliveries':max_deliveries,'recruitment_budget':recruitment_budget,
              'recruitment_policy':recruitment_policy,'limits':asdict(limits)}
        self.bus.event(event)
        self._session(session,spec)
        queue=[event];seen=set()
        def outcome(status,**details):
            return {'session':session,'status':status,**self._stats(session),**details}
        while queue:
            current=queue.pop(0)
            if current in seen:continue
            seen.add(current)
            delivery=self._delivery(session,current,max_deliveries)
            if delivery!='READY':return outcome(delivery,event=current)
            members=next((g['experts'] for g in self.bus.groups(current)
                          if (g['target'],g['group'])==(target,group)),())
            if not members:continue
            run=self._reserve_run(session,current,target,group,max_runs)
            if run is None:return outcome('RUN_BUDGET',event=current)
            historical=bool(self.db.execute('SELECT 1 FROM group_runs_v08 WHERE id=?',(run,)).fetchone())
            if historical:
                data=self.work.receipt(run)
            else:
                data=self.work.execute(run,current,target,group,inputs,goal,validators,oracle_id=oracle_id,
                                       limits=limits,recruitment_budget=recruitment_budget,recruitment_policy=recruitment_policy)
            if (data['run']!=run or data['event']!=current or data['target']!=target or data['group']!=group or
                data['inputs']!={k:list(v) for k,v in inputs.items()} or data['goal']!=goal or
                data['oracle_id']!=oracle_id or data['validator_topics']!=sorted(validators) or
                data['limits']!=asdict(limits)):
                raise ContractError('continuation receipt does not match task')
            result=data['result']
            if result['status']=='OK':
                # Historical receipts are not returned as newly verified answers.
                if historical:return outcome('HISTORICAL',run=run,event=current)
                return outcome('OK',run=run,event=current,value=result['value'])
            if result['status']!='STAGNATION':
                return outcome(result['status'],run=run,event=current)
            children=self.work.dispatch_recruitment(run)
            for child in children:
                e=self.bus.event(child)
                if e['kind']!='RECRUIT' or e['payload']['run']!=run or e['payload']['parent_event']!=current:
                    raise ContractError('continuation child provenance')
                if child not in seen:queue.append(child)
        return outcome('STAGNATION')
