"""Durable learning notifications and voluntary, bounded group participation.

Trusted single-owner host. Expert-local callbacks are proposals, not trained
perception heads; this transport never authorizes learning or admission.
"""
import json
from .abi import ContractError
from .environment import identifier
from .global_relearning import GlobalRelearning
from .knowledge_learning import _hash
from .population import RESIDENT


class LearningEvents:
    def __init__(self, tasks, *, membership_budget=64, memberships_per_expert=4):
        if type(membership_budget) is not int or not 1 <= membership_budget <= 4096:
            raise ContractError('membership budget')
        if type(memberships_per_expert) is not int or not 1 <= memberships_per_expert <= min(8, membership_budget):
            raise ContractError('expert membership budget')
        self.global_learning = GlobalRelearning(tasks)
        self.learning = tasks.learning
        self.db = self.learning.knowledge.db
        self.membership_budget = membership_budget
        self.memberships_per_expert = memberships_per_expert
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS learning_events_v08(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
          kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS learning_event_decisions_v08(
          event TEXT NOT NULL REFERENCES learning_events_v08(id),
          expert TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(event,expert));
        CREATE TABLE IF NOT EXISTS learning_event_members_v08(
          event TEXT NOT NULL, target TEXT NOT NULL, name TEXT NOT NULL,
          expert TEXT NOT NULL, PRIMARY KEY(event,target,name,expert),
          FOREIGN KEY(event,expert) REFERENCES learning_event_decisions_v08(event,expert));
        CREATE TABLE IF NOT EXISTS learning_event_cursor_v08(
          singleton INTEGER PRIMARY KEY CHECK(singleton=1), expert TEXT NOT NULL);
        ''')

    @staticmethod
    def _limit(n):
        if type(n) is not int or not 1 <= n <= 1024:
            raise ContractError('event limit')
        return n

    def _insert(self, kind, payload):
        event = _hash({'kind':kind,'payload':payload})
        self.db.execute('INSERT OR IGNORE INTO learning_events_v08(id,kind,payload) VALUES(?,?,?)',
                        (event,kind,json.dumps(payload,sort_keys=True)))
        return event

    def _conflicts(self):
        snapshots={}
        for impulse in self.global_learning.impulses():
            # Include attempt history so a reopened dependency does not reuse
            # an already acknowledged pre-repair event with identical targets.
            attempts=self.db.execute('''SELECT job_id,completed FROM global_repair_attempts_v08
                WHERE task=? ORDER BY rowid''',(impulse['case'],)).fetchall()
            snapshot={**impulse,'repair_revision':_hash(attempts)}
            snapshots[_hash(snapshot)]=snapshot
        return snapshots

    def sync_conflicts(self, *, limit=32):
        """Publish changed global repair snapshots, idempotently; bounded inserts.

        The original durable conflict case remains the source of truth. This
        is a pull/outbox transport, not a network push or a global scheduler.
        """
        self._limit(limit)
        new=[]
        with self.learning._guard():
            for impulse in self._conflicts().values():
                targets = (['local'] if impulse['local_open'] else []) + [
                    'repair/'+x for x in impulse['pending_reconstruction']]
                payload={'scope':'global','impulse':impulse,'targets':targets}
                event=_hash({'kind':'RELEARN','payload':payload})
                if self.db.execute('SELECT 1 FROM learning_events_v08 WHERE id=?',(event,)).fetchone():
                    continue
                new.append(self._insert('RELEARN',payload))
                if len(new)>=limit: break
            self.db.commit()
        return tuple(new)

    def publish_knowledge(self, job_id):
        """Explicit host notification of an independently admitted new expert."""
        identifier(job_id)
        l=self.learning
        with l._guard():
            j=l.job(job_id)
            if j['state'] not in RESIDENT or not j['report'] or not j['report']['accepted']:
                raise ContractError('verified resident knowledge required')
            l._check_sources(j)
            m=l.catalog.manifest(j['expert_id'])
            payload={'scope':'global','job':job_id,'expert':j['expert_id'],
                     'domain':j['spec']['domain'],'artifact_sha256':m['sha256'],
                     'targets':['explore']}
            event=self._insert('KNOWLEDGE',payload)
            self.db.commit()
        return event

    def _group_knowledge_payload(self, job_id, *, resident_only=False):
        """Bind an announcement to an independently admitted group artifact.

        The caller holds the source guard, which checks declared ancestors.
        An archived result may remain historically relevant but never executes
        unless separately reactivated. No new learning authority is granted.
        """
        l=self.learning; db=l.db
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='group_learning_jobs_v08'").fetchone():
            raise ContractError('group learning required')
        row=db.execute('SELECT expert_id,spec,sha FROM group_learning_jobs_v08 WHERE id=?',(job_id,)).fetchone()
        report=db.execute('SELECT passed,case_sha,artifact_sha,receipt FROM group_learning_reports_v08 WHERE job_id=?',(job_id,)).fetchone()
        if row is None or report is None: raise ContractError('admitted group competency required')
        expert,raw,sha=row; spec=json.loads(raw); receipt=json.loads(report[3])
        if _hash(spec)!=sha or spec.get('expert')!=expert: raise ContractError('group job integrity')
        m=l.catalog.manifest(expert)
        allowed=RESIDENT if resident_only else RESIDENT|{'ARCHIVED'}
        if (m['state'] not in allowed or m.get('group_job')!=job_id or m['parents']!=spec['parents'] or
            m['family']!=spec['family'] or m['source_topic']!=spec['source'] or m['target_topic']!=spec['goal'] or
            report[:3]!=(1,receipt.get('case_sha256'),m['sha256']) or receipt.get('accepted') is not True or
            receipt.get('artifact_sha256')!=m['sha256'] or receipt.get('observed_passed')!=receipt.get('observed_cases') or
            receipt.get('holdout_passed')!=receipt.get('holdout_cases')):
            raise ContractError('admitted group competency required')
        return {'scope':'global','group_job':job_id,'expert':expert,'domain':{'topic':m['family']},
                'artifact_sha256':m['sha256'],'spec_sha256':sha,'report_sha256':_hash(receipt),
                'source_topic':m['source_topic'],'target_topic':m['target_topic'],'targets':['explore']}

    def publish_group_knowledge(self, job_id):
        """Announce newly learned group competencies through the same event bus."""
        identifier(job_id)
        with self.learning._guard():
            payload=self._group_knowledge_payload(job_id,resident_only=True)
            event=self._insert('KNOWLEDGE',payload)
            self.db.commit()
            return event

    def _recruitment_payload(self, run, requester, conflicts, *, expected_depth=None):
        """Derive a missing-input impulse from a current, stagnated group receipt.

        Only the requested topic is global; no task values or private signals
        leave the originating group. This first reference uses core topic
        contracts, not a learned request-generation policy.
        """
        identifier(run); identifier(requester)
        # Run receipts and admitted heads belong to the catalog, while
        # event memberships belong to the source/knowledge database.
        catalog_db=self.learning.db
        if not catalog_db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='group_runs_v08'").fetchone():
            raise ContractError('group work required')
        row=catalog_db.execute('SELECT receipt,sha FROM group_runs_v08 WHERE id=?',(run,)).fetchone()
        if row is None: raise ContractError('unknown recruitment run')
        receipt=json.loads(row[0])
        if _hash(receipt)!=row[1] or receipt.get('run')!=run or receipt.get('contract')!='group-work-v1':
            raise ContractError('recruitment receipt integrity')
        result=receipt['result']
        if result['status']!='STAGNATION' or result['faults']:
            raise ContractError('clean stagnated group required')
        parent=self.event(receipt['event'])
        depth=parent['payload'].get('depth',0) if parent['kind']=='RECRUIT' else 0
        if type(depth) is not int or not 0<=depth<4 or (expected_depth is not None and depth+1!=expected_depth):
            raise ContractError('recruitment depth budget')
        if not self._live(parent,conflicts): raise ContractError('stale recruitment parent')
        binding=next((m for m in receipt['members'] if m['expert']==requester),None)
        if binding is None: raise ContractError('requester must be a group member')
        residents=set(self.learning.population.resident_ids())
        # Every originating member must still have its recorded core/head and
        # membership; no requests from a superseded or quarantined group.
        for member in receipt['members']:
            expert=member['expert']
            if expert not in residents:
                raise ContractError('stale recruitment member')
            m=self.learning.catalog.manifest(expert)
            current=catalog_db.execute('''SELECT b.head,h.sha FROM perception_bindings_v08 b
                JOIN perception_heads_v08 h ON h.id=b.head WHERE b.expert=?''',(expert,)).fetchone()
            joined=self.db.execute('''SELECT 1 FROM learning_event_members_v08
                WHERE event=? AND target=? AND name=? AND expert=?''',
                (parent['id'],receipt['target'],receipt['group'],expert)).fetchone()
            if m['sha256']!=member['core_sha256'] or current!=(member['head'],member['head_sha256']) or not joined:
                raise ContractError('stale recruitment binding')
        core=self.learning.catalog.manifest(requester)
        topic=identifier(core['source_topic'])
        if any(f['scope']==receipt['scope'] and f['topic']==topic for f in result['facts']):
            raise ContractError('requester input not missing')
        return {'scope':'global','parent_event':parent['id'],'run':run,'run_sha256':row[1],
                'requester':requester,'artifact_sha256':core['sha256'],
                'domain':{'topic':topic},'requested_topic':topic,'depth':depth+1,
                'targets':['explore']}

    def publish_recruitment(self, run, requester):
        """Host-authorized publication; recipients still choose locally."""
        return self.publish_recruitments(run,(requester,))[0]

    def publish_recruitments(self, run, requesters):
        """Validate all requests before one source-DB commit; idempotent retry.

        The caller authorizes publication of topic metadata, not task values.
        The source guard revalidates the historical run and all its ancestors.
        No semantic ranking or selection of the requested experts.
        """
        identifier(run)
        if type(requesters) not in (tuple,list) or not 1<=len(requesters)<=64:
            raise ContractError('recruitment batch budget')
        requesters=tuple(identifier(x) for x in requesters)
        if len(set(requesters))!=len(requesters):
            raise ContractError('duplicate recruitment requester')
        with self.learning._guard():
            conflicts=self._conflicts()
            payloads=tuple(self._recruitment_payload(run,x,conflicts) for x in requesters)
            events=tuple(self._insert('RECRUIT',p) for p in payloads)
            self.db.commit()
        return events

    def event(self, event):
        identifier(event)
        row=self.db.execute('SELECT kind,payload FROM learning_events_v08 WHERE id=?',(event,)).fetchone()
        if row is None: raise ContractError('unknown learning event')
        return {'id':event,'kind':row[0],'payload':json.loads(row[1])}

    def _live(self, e, conflicts):
        l=self.learning; p=e['payload']
        if e['kind']=='RELEARN':
            return _hash(p['impulse']) in conflicts
        if e['kind']=='RECRUIT':
            try:
                if _hash({'kind':'RECRUIT','payload':p})!=e['id']:
                    return False
                return p==self._recruitment_payload(p['run'],p['requester'],conflicts,expected_depth=p['depth'])
            except (ContractError, KeyError, TypeError, ValueError):
                return False
        if e['kind']!='KNOWLEDGE': return False
        if 'group_job' in p:
            try: return p==self._group_knowledge_payload(p['group_job'])
            except ContractError: return False
        j=l.job(p['job'])
        if j['state'] not in RESIDENT|{'ARCHIVED'} or not j['report'] or not j['report']['accepted']:
            return False
        if l.catalog.manifest(p['expert'])['sha256']!=p['artifact_sha256']: return False
        try: l._check_sources(j)
        except ContractError: return False
        return True

    def poll(self, expert, *, limit=16):
        """Every live resident can see every current unacknowledged global event."""
        identifier(expert); self._limit(limit)
        l=self.learning; result=[]
        with l._guard():
            if expert not in l.population.resident_ids(): raise ContractError('resident participant required')
            conflicts=self._conflicts()
            for (event,) in self.db.execute('''SELECT e.id FROM learning_events_v08 e
                WHERE NOT EXISTS(SELECT 1 FROM learning_event_decisions_v08 d
                  WHERE d.event=e.id AND d.expert=?) ORDER BY e.seq''',(expert,)):
                e=self.event(event)
                if self._live(e,conflicts): result.append(e)
                if len(result)>=limit: break
        return tuple(result)

    def respond(self, event, expert, joins=()):
        """Persist an expert's voluntary group proposals or explicit decline.

        The group key is chosen by the local proposer. Several experts can
        independently choose the same key; one may join overlapping groups.
        No score ranking, task assignment, weight changes or implicit admission.
        """
        identifier(event); identifier(expert); l=self.learning
        if type(joins) not in (list,tuple) or len(joins)>self.memberships_per_expert:
            raise ContractError('expert membership budget')
        normalized=[]
        for pair in joins:
            if type(pair) not in (tuple,list) or len(pair)!=2 or type(pair[0]) is not str:
                raise ContractError('group proposal contract')
            normalized.append((pair[0],identifier(pair[1])))
        if len(set(normalized))!=len(normalized): raise ContractError('duplicate group proposal')
        normalized.sort()
        receipt={'event':event,'expert':expert,'joins':[list(x) for x in normalized]}
        encoded=json.dumps(receipt,sort_keys=True)
        with l._guard():
            prior=self.db.execute('SELECT receipt FROM learning_event_decisions_v08 WHERE event=? AND expert=?',(event,expert)).fetchone()
            if prior:
                if prior!=(encoded,): raise ContractError('immutable event response')
                return receipt
            e=self.event(event)
            if expert not in l.population.resident_ids(): raise ContractError('resident participant required')
            if not self._live(e,self._conflicts()): raise ContractError('stale learning event')
            if any(target not in e['payload']['targets'] for target,_ in normalized):
                raise ContractError('unknown event target')
            used=self.db.execute('SELECT COUNT(*) FROM learning_event_members_v08 WHERE event=?',(event,)).fetchone()[0]
            if used+len(normalized)>self.membership_budget: raise ContractError('event membership budget')
            self.db.execute('INSERT INTO learning_event_decisions_v08 VALUES(?,?,?)',(event,expert,encoded))
            self.db.executemany('INSERT INTO learning_event_members_v08 VALUES(?,?,?,?)',
                                ((event,target,name,expert) for target,name in normalized))
            self.db.commit()
        return receipt

    def groups(self, event):
        """Live overlapping groups; stale/archived participants are not executable."""
        l=self.learning; result={}
        with l._guard():
            e=self.event(event)
            if not self._live(e,self._conflicts()): return ()
            residents=set(l.population.resident_ids())
            for target,name,expert in self.db.execute('''SELECT target,name,expert FROM learning_event_members_v08
                WHERE event=? ORDER BY target,name,expert''',(event,)):
                if expert in residents: result.setdefault((target,name),[]).append(expert)
        return tuple({'target':target,'group':name,'experts':tuple(members)}
                     for (target,name),members in result.items())

    def tick(self, policies, *, delivery_budget=16):
        """Bounded host-driven delivery to independent local proposal callbacks.

        No DB lock during callback. A crash before response commit may repeat
        delivery; callbacks must be side-effect-free or idempotent. Count calls,
        not wall time, token use, or GPU compute. No implicit learning execution.
        """
        self._limit(delivery_budget)
        if type(policies) is not dict or any(not callable(p) for p in policies.values()):
            raise ContractError('local proposal adapters required')
        for expert in policies: identifier(expert)
        self.sync_conflicts(limit=delivery_budget)
        delivered=[]
        # Round-robin within this tick, no cognitive ranking of experts.
        with self.learning._guard():
            residents=set(self.learning.population.resident_ids())
        ordered=sorted(set(policies)&residents)
        last=self.db.execute('SELECT expert FROM learning_event_cursor_v08 WHERE singleton=1').fetchone()
        if last:
            ordered=[x for x in ordered if x>last[0]]+[x for x in ordered if x<=last[0]]
        pending={expert:list(self.poll(expert,limit=delivery_budget)) for expert in ordered}
        while any(pending.values()) and len(delivered)<delivery_budget:
            for expert,events in pending.items():
                if not events or len(delivered)>=delivery_budget: continue
                e=events.pop(0); event=e['id']
                try:
                    joins=policies[expert](e)
                    receipt=self.respond(event,expert,joins)
                except Exception as exc:
                    delivered.append({'event':event,'expert':expert,'status':'DEFERRED','error':type(exc).__name__})
                else:
                    delivered.append({'event':event,'expert':expert,'status':'RECORDED','receipt':receipt})
                with self.learning._guard():
                    self.db.execute('INSERT INTO learning_event_cursor_v08 VALUES(1,?) ON CONFLICT(singleton) DO UPDATE SET expert=excluded.expert',(expert,))
                    self.db.commit()
        return tuple(delivered)
