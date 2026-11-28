"""Indexed reverse provenance and change journals for the trusted single owner.

Each DB records changes in its own transaction. Checkpoints are process-local;
restart always revalidates the full catalog. No cross-DB atomicity claim.
"""
import json
from .abi import ContractError


class DependencyIndex:
    def __init__(self, learning):
        self.l=learning; self.catalog=learning.db; self.source=learning.knowledge.db
        self.checkpoint=None
        # Rebuild derived indexes at owner startup, including pre-dev14 rows.
        self.catalog.executescript('''
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS dep_sources_v08(record_id INTEGER, expert TEXT, PRIMARY KEY(record_id,expert));
        CREATE TABLE IF NOT EXISTS dep_parents_v08(parent TEXT, child TEXT, PRIMARY KEY(parent,child));
        CREATE INDEX IF NOT EXISTS dep_child_v08 ON dep_parents_v08(child,parent);
        CREATE INDEX IF NOT EXISTS dep_source_expert_v08 ON dep_sources_v08(expert,record_id);
        CREATE INDEX IF NOT EXISTS dep_state_v08 ON experts(state,id);
        CREATE TABLE IF NOT EXISTS dep_catalog_events_v08(seq INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT NOT NULL DEFAULT (hex(randomblob(16))), expert TEXT);
        DELETE FROM dep_sources_v08;
        INSERT INTO dep_sources_v08 SELECT json_extract(s.value,'$.id'),j.expert_id FROM knowledge_jobs_v08 j,json_each(j.spec,'$.sources') s;
        DELETE FROM dep_parents_v08;
        INSERT INTO dep_parents_v08 SELECT p.value,e.id FROM experts e,json_each(e.manifest,'$.parents') p;
        CREATE TRIGGER IF NOT EXISTS dep_job_insert_v08 AFTER INSERT ON knowledge_jobs_v08 BEGIN
          INSERT INTO dep_sources_v08 SELECT json_extract(value,'$.id'),new.expert_id FROM json_each(new.spec,'$.sources');
          INSERT INTO dep_catalog_events_v08(expert) VALUES(new.expert_id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_job_update_v08 AFTER UPDATE ON knowledge_jobs_v08 BEGIN
          DELETE FROM dep_sources_v08 WHERE expert=old.expert_id;
          INSERT INTO dep_sources_v08 SELECT json_extract(value,'$.id'),new.expert_id FROM json_each(new.spec,'$.sources');
          INSERT INTO dep_catalog_events_v08(expert) VALUES(NULL);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_job_delete_v08 AFTER DELETE ON knowledge_jobs_v08 BEGIN
          DELETE FROM dep_sources_v08 WHERE expert=old.expert_id;
          INSERT INTO dep_catalog_events_v08(expert) VALUES(NULL);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_expert_insert_v08 AFTER INSERT ON experts BEGIN
          INSERT INTO dep_parents_v08 SELECT value,new.id FROM json_each(new.manifest,'$.parents');
          INSERT INTO dep_catalog_events_v08(expert) VALUES(new.id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_expert_manifest_v08 AFTER UPDATE OF manifest ON experts BEGIN
          DELETE FROM dep_parents_v08 WHERE child=old.id;
          INSERT INTO dep_parents_v08 SELECT value,new.id FROM json_each(new.manifest,'$.parents');
          INSERT INTO dep_catalog_events_v08(expert) VALUES(NULL);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_expert_delete_v08 AFTER DELETE ON experts BEGIN
          DELETE FROM dep_parents_v08 WHERE child=old.id;
          INSERT INTO dep_catalog_events_v08(expert) VALUES(NULL);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_expert_state_v08 AFTER UPDATE OF state ON experts BEGIN
          INSERT INTO dep_catalog_events_v08(expert) VALUES(new.id);
        END;
        COMMIT;
        ''')
        self.source.executescript('''
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS dep_source_events_v08(seq INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT NOT NULL DEFAULT (hex(randomblob(16))), record_id INTEGER, conflict TEXT);
        CREATE TRIGGER IF NOT EXISTS dep_head_insert_v08 AFTER INSERT ON knowledge_heads_v1 BEGIN
          INSERT INTO dep_source_events_v08(record_id) VALUES(new.id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_head_update_v08 AFTER UPDATE ON knowledge_heads_v1 BEGIN
          INSERT INTO dep_source_events_v08(record_id) VALUES(old.id),(new.id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_head_delete_v08 AFTER DELETE ON knowledge_heads_v1 BEGIN
          INSERT INTO dep_source_events_v08(record_id) VALUES(old.id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_record_update_v08 AFTER UPDATE ON knowledge_v1 BEGIN
          INSERT INTO dep_source_events_v08(record_id) VALUES(old.id),(new.id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_record_delete_v08 AFTER DELETE ON knowledge_v1 BEGIN
          INSERT INTO dep_source_events_v08(record_id) VALUES(old.id);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_conflict_insert_v08 AFTER INSERT ON conflict_cases_v08 BEGIN
          INSERT INTO dep_source_events_v08(conflict) VALUES(new.task);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_conflict_edit_v08 AFTER UPDATE OF receipt ON conflict_cases_v08 BEGIN
          INSERT INTO dep_source_events_v08(record_id,conflict) VALUES(NULL,NULL);
        END;
        CREATE TRIGGER IF NOT EXISTS dep_conflict_delete_v08 AFTER DELETE ON conflict_cases_v08 BEGIN
          INSERT INTO dep_source_events_v08(record_id,conflict) VALUES(NULL,NULL);
        END;
        COMMIT;
        ''')

    def consumers(self, rid):
        return {r[0] for r in self.catalog.execute('SELECT expert FROM dep_sources_v08 WHERE record_id=?',(rid,))}

    def descendants(self, seeds):
        seen=set(seeds); pending=list(seen)
        while pending:
            parent=pending.pop()
            for (child,) in self.catalog.execute('SELECT child FROM dep_parents_v08 WHERE parent=?',(parent,)):
                if child not in seen:
                    seen.add(child); pending.append(child)
        return seen

    @staticmethod
    def _tip(db, table):
        return db.execute(f'SELECT seq,token FROM {table} ORDER BY seq DESC LIMIT 1').fetchone() or (0,'')

    @staticmethod
    def _changes(db, table, anchor, tip, fields):
        # A rollback can reuse a sequence; random token distinguishes histories.
        if anchor[0] and db.execute(f'SELECT seq,token FROM {table} WHERE seq=?',(anchor[0],)).fetchone()!=anchor:
            return None
        if tip[0]<anchor[0]: return None
        return db.execute(f'SELECT {fields} FROM {table} WHERE seq>? AND seq<=? ORDER BY seq',(anchor[0],tip[0])).fetchall()

    def audit(self, *, full=False):
        l=self.l; c=self.catalog; s=self.source
        st='dep_source_events_v08'; ct='dep_catalog_events_v08'
        tips=(self._tip(s,st),self._tip(c,ct))
        full=full or self.checkpoint is None
        if not full:
            sc=self._changes(s,st,self.checkpoint[0],tips[0],'record_id,conflict')
            cc=self._changes(c,ct,self.checkpoint[1],tips[1],'expert')
            full=sc is None or cc is None or any(r is None and t is None for r,t in sc) or any(e is None for e, in cc)
        invalid=set(); candidates=set()
        if full:
            candidates.update(e for e, in c.execute('SELECT id FROM experts'))
            conflicts=s.execute('SELECT receipt FROM conflict_cases_v08')
        else:
            candidates.update(e for e, in cc)
            for rid,task in sc:
                if rid is not None: candidates.update(self.consumers(rid))
            conflicts=(row for _,task in sc if task is not None for row in s.execute('SELECT receipt FROM conflict_cases_v08 WHERE task=?',(task,)))
        for raw, in conflicts: invalid.update(json.loads(raw)['suspects'])
        checked=0
        for expert in sorted(candidates):
            row=c.execute('SELECT state,manifest FROM experts WHERE id=?',(expert,)).fetchone()
            if row is None: continue # job without staged artifact
            state,raw=row; m=json.loads(raw)
            if state=='QUARANTINED': invalid.add(expert); continue
            if c.execute("SELECT 1 FROM dep_parents_v08 p LEFT JOIN experts e ON e.id=p.parent WHERE p.child=? AND (e.id IS NULL OR e.state='QUARANTINED') LIMIT 1",(expert,)).fetchone(): invalid.add(expert)
            binding=c.execute('SELECT id FROM knowledge_jobs_v08 WHERE expert_id=?',(expert,)).fetchone()
            if binding is not None or 'learning_job' in m:
                checked+=1
                try:
                    if binding!=(m.get('learning_job'),): raise ContractError('learning manifest identity changed')
                    j=l.job(m['learning_job'])
                    if j['expert_id']!=expert: raise ContractError('learning job identity changed')
                    l._check_sources(j)
                except ContractError: invalid.add(expert)
        affected=tuple(sorted(e for e in self.descendants(invalid) if c.execute("SELECT 1 FROM experts WHERE id=? AND state!='QUARANTINED'",(e,)).fetchone()))
        if affected: l.population.quarantine(affected,reason='knowledge-source-revision')
        # Keep pre-audit tips: quarantine-generated events are seen next time.
        self.checkpoint=tips
        self.last_audit={'full':full,'candidates':len(candidates),'source_checks':checked,'quarantined':len(affected)}
        return affected
