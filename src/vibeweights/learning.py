"""Symbolic learning infrastructure: immutable answer/feedback lineage and observed telemetry.
No external model calls, weight learning, or causal attribution from ratings.
"""
import json, re, sqlite3, uuid
from .abi import ContractError

class LearningLedger:
    def __init__(self,db):
        self.db=db
        db.executescript('''
        CREATE TABLE IF NOT EXISTS answer_v05(id TEXT PRIMARY KEY, session TEXT NOT NULL, parent TEXT, request TEXT NOT NULL, result TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS answer_session_v05 ON answer_v05(session,id);
        CREATE TABLE IF NOT EXISTS expert_observation_v05(answer TEXT NOT NULL, group_no INTEGER NOT NULL, expert TEXT NOT NULL, context TEXT NOT NULL, joined INTEGER NOT NULL, shots INTEGER NOT NULL, probes INTEGER NOT NULL, invitations INTEGER NOT NULL, offers INTEGER NOT NULL, proposals INTEGER NOT NULL, verified INTEGER NOT NULL, PRIMARY KEY(answer,group_no,expert));
        CREATE TABLE IF NOT EXISTS feedback_v05(seq INTEGER PRIMARY KEY, answer TEXT NOT NULL, kind TEXT NOT NULL, note TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reflection_v05(id TEXT PRIMARY KEY, feedback INTEGER NOT NULL UNIQUE, state TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS variant_v05(id TEXT PRIMARY KEY, family TEXT NOT NULL, parent TEXT, digest TEXT NOT NULL, origin TEXT NOT NULL, state TEXT NOT NULL);
        ''')
    def answer(self,session,id):
        row=self.db.execute('SELECT request,result FROM answer_v05 WHERE session=? AND id=?',(session,id)).fetchone()
        if row is None: raise ContractError('unknown answer in session')
        return row[0],json.loads(row[1])
    def record(self,session,request,response,identities,parent=None):
        # Caller owns transaction, including journal write.
        if parent is not None: self.answer(session,parent)
        id=uuid.uuid4().hex; response['answer_id']=id; response['parent_answer_id']=parent
        self.db.execute('INSERT INTO answer_v05 VALUES(?,?,?,?,?)',(id,session,parent,request,json.dumps(response,ensure_ascii=False)))
        for group,g in enumerate(response.get('groups',[response]),1):
            events=g.get('events',[])
            for i,identity in identities.items():
                counts=lambda kind,field='source':sum(e['kind']==kind and e[field]==i for e in events)
                row=(counts('JOIN'),counts('SHOT','target'),counts('PROBE'),counts('INVITE','target'),counts('OFFER'),counts('PROPOSE'))
                if any(row): self.db.execute('INSERT INTO expert_observation_v05 VALUES(?,?,?,?,?,?,?,?,?,?,?)',(id,group,identity,g.get('task','unknown'),*row,int(bool(g.get('core_verified',g.get('verified'))))))
        return id
    def feedback(self,session,answer,kind,note=''):
        self.answer(session,answer)
        if kind not in ('positive','negative','correction','preference') or type(note) is not str or len(note)>2000: raise ContractError('invalid feedback')
        with self.db:
            seq=self.db.execute('INSERT INTO feedback_v05(answer,kind,note) VALUES(?,?,?)',(answer,kind,note)).lastrowid
            # Positive feedback is an observation too, not verification.
            rid=uuid.uuid4().hex
            self.db.execute('INSERT INTO reflection_v05 VALUES(?,?,?)',(rid,seq,'PENDING'))
        return {'feedback_id':seq,'reflection_id':rid,'state':'PENDING','answer_id':answer}
    def pending(self,session):
        rows=self.db.execute('''SELECT r.id,f.seq,f.answer,f.kind,f.note,r.state FROM reflection_v05 r JOIN feedback_v05 f ON r.feedback=f.seq JOIN answer_v05 a ON a.id=f.answer WHERE a.session=? AND r.state='PENDING' ORDER BY f.seq''',(session,))
        return [dict(zip(('reflection_id','feedback_id','answer_id','kind','note','state'),row)) for row in rows]
    def metrics(self,session):
        # Latest feedback per answer only. Group observations are not causal credit.
        rows=self.db.execute('''SELECT o.expert,o.context,COUNT(*),SUM(o.joined),SUM(o.shots),SUM(o.probes),SUM(o.invitations),SUM(o.offers),SUM(o.proposals),SUM(CASE WHEN o.proposals>0 AND o.verified=1 THEN 1 ELSE 0 END),SUM(CASE WHEN o.proposals>0 AND f.kind='positive' THEN 1 ELSE 0 END),SUM(CASE WHEN o.proposals>0 AND f.kind IN ('negative','correction') THEN 1 ELSE 0 END) FROM expert_observation_v05 o JOIN answer_v05 a ON a.id=o.answer LEFT JOIN feedback_v05 f ON f.seq=(SELECT MAX(seq) FROM feedback_v05 WHERE answer=a.id AND kind IN ('positive','negative','correction')) WHERE a.session=? GROUP BY o.expert,o.context ORDER BY o.expert,o.context''',(session,))
        keys=('expert','context','observed_groups','joins','shots','probes','invitations','offers','proposals','verified_group_contributions','positive_answer_associations','negative_or_corrected_answer_associations')
        return [dict(zip(keys,row)) for row in rows]
    def stage_variant(self,id,family,digest,origin,parent=None):
        # Metadata only. Does not load or promote weights; immutable lineage.
        if any(type(x) is not str or not re.fullmatch(r'[A-Za-z0-9_.@-]{1,128}',x) for x in (id,family)) or type(digest) is not str or not re.fullmatch(r'[0-9a-f]{64}',digest) or origin not in ('DIRECT','SYNTHESIS','PRETRAINED','TRAINED','SYMBOLIC'):
            raise ContractError('invalid variant metadata')
        if parent is not None:
            row=self.db.execute('SELECT family FROM variant_v05 WHERE id=?',(parent,)).fetchone()
            if row is None or row[0]!=family: raise ContractError('invalid variant parent')
        try:
            with self.db: self.db.execute('INSERT INTO variant_v05 VALUES(?,?,?,?,?,?)',(id,family,parent,digest,origin,'STAGED'))
        except sqlite3.IntegrityError as e: raise ContractError('duplicate variant') from e
    def variants(self):
        return [dict(zip(('id','family','parent','sha256','origin','state'),r)) for r in self.db.execute('SELECT * FROM variant_v05 ORDER BY id')]
