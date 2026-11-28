"""Versioned numeric knowledge ledger. Trusted host owns write/verification policy.
Scope checks are logical boundaries for trusted adapters, not a Python sandbox.
"""
from dataclasses import dataclass
import json, math, sqlite3, time
from .abi import ContractError
from .environment import identifier

KINDS = ('semantic','episodic','procedural','cooperation','reflection','hypothesis')
STATES = ('OBSERVED','HYPOTHESIS','VERIFIED','UNCERTAIN','CONFLICTED','INVALIDATED')

def integer(value, low=1, high=2**63-1):
    if type(value) is not int or not low <= value <= high:
        raise ContractError('invalid integer bound')
    return value

def vector(values, *, unit=False):
    if type(values) not in (tuple,list) or not 1 <= len(values) <= 4096:
        raise ContractError('vector shape')
    if any(type(x) not in (float,int) or abs(x)>1e100 or not math.isfinite(x) for x in values):
        raise ContractError('numeric vector required')
    values=tuple(values)
    if unit:
        norm=math.hypot(*values)
        if norm==0: raise ContractError('zero embedding')
        return tuple(v/norm for v in values)
    return values

@dataclass(frozen=True)
class Record:
    id: int
    scope: str
    key: str
    version: int
    topic: str
    kind: str
    status: str
    values: tuple
    source: str
    previous: int | None
    parents: tuple
    space: str | None
    embedding: tuple | None
    created_ns: int

@dataclass(frozen=True)
class Access:
    expert: str
    groups: tuple = ()
    def __post_init__(self):
        identifier(self.expert)
        if type(self.groups) is not tuple or len(self.groups)>64 or len(set(self.groups))!=len(self.groups): raise ContractError('group access')
        for g in self.groups:
            identifier(g)
            if not g.startswith('group:'): raise ContractError('group scope')
        identifier('private:'+self.expert)
    @property
    def scopes(self): return ('global','private:'+self.expert,*self.groups)

class KnowledgeLedger:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=2)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS knowledge_v1(
          id INTEGER PRIMARY KEY, scope TEXT NOT NULL, key TEXT NOT NULL,
          version INTEGER NOT NULL, topic TEXT NOT NULL, kind TEXT NOT NULL,
          status TEXT NOT NULL, payload TEXT NOT NULL, source TEXT NOT NULL,
          previous INTEGER REFERENCES knowledge_v1(id), parents TEXT NOT NULL,
          space TEXT, embedding TEXT, created_ns INTEGER NOT NULL, UNIQUE(scope,key,version));
        CREATE TABLE IF NOT EXISTS knowledge_heads_v1(
          scope TEXT NOT NULL, key TEXT NOT NULL, id INTEGER NOT NULL REFERENCES knowledge_v1(id),
          PRIMARY KEY(scope,key));
        CREATE INDEX IF NOT EXISTS knowledge_topic_v1 ON knowledge_v1(scope,topic,kind,status,id);
        CREATE INDEX IF NOT EXISTS knowledge_time_v1 ON knowledge_v1(scope,created_ns,id);
        CREATE INDEX IF NOT EXISTS knowledge_source_v1 ON knowledge_v1(scope,source,id);
        CREATE INDEX IF NOT EXISTS knowledge_embedding_v1 ON knowledge_v1(scope,space,id);
        CREATE TABLE IF NOT EXISTS knowledge_spaces_v1(space TEXT PRIMARY KEY, width INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_links_v1(
          src INTEGER NOT NULL REFERENCES knowledge_v1(id), relation TEXT NOT NULL,
          dst INTEGER NOT NULL REFERENCES knowledge_v1(id), PRIMARY KEY(src,relation,dst));
        CREATE INDEX IF NOT EXISTS knowledge_links_reverse_v1 ON knowledge_links_v1(dst,relation,src);
        ''')
    def close(self): self.db.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    @staticmethod
    def _record(row):
        return Record(*row[:7],tuple(json.loads(row[7])),*row[8:10],tuple(json.loads(row[10])),row[11],None if row[12] is None else tuple(json.loads(row[12])),row[13])
    def read(self,rid,access):
        integer(rid)
        row=self.db.execute('SELECT * FROM knowledge_v1 WHERE id=?',(rid,)).fetchone()
        if row is None or row[1] not in access.scopes: raise ContractError('record unavailable')
        return self._record(row)
    def current_verified(self,rid,access):
        """Trusted live read: historical revisions remain inspectable via read()."""
        pending=[rid]; seen=set(); result=None
        while pending:
            item=pending.pop()
            if item in seen: continue
            if len(seen)>=512: raise ContractError('source dependency bound')
            rec=self.read(item,access)
            if rec.status!='VERIFIED' or self.db.execute(
                'SELECT id FROM knowledge_heads_v1 WHERE scope=? AND key=?',
                (rec.scope,rec.key)).fetchone()!=(item,):
                raise ContractError('current verified knowledge required')
            if item==rid: result=rec
            seen.add(item); pending.extend(rec.parents)
        return result

    def store(self,access,*,scope,key,topic,values,source,kind='semantic',status='OBSERVED',expected=None,parents=(),space=None,embedding=None):
        """Host-only append in its own transaction; never roll back a caller."""
        if self.db.in_transaction: raise ContractError('nested public knowledge write')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            return self._store_locked(access,scope=scope,key=key,topic=topic,values=values,
                source=source,kind=kind,status=status,expected=expected,parents=parents,
                space=space,embedding=embedding)

    def _store_locked(self,access,*,scope,key,topic,values,source,kind='semantic',status='OBSERVED',expected=None,parents=(),space=None,embedding=None):
        """Internal append; caller owns an IMMEDIATE transaction."""
        if not self.db.in_transaction: raise ContractError('knowledge transaction required')
        for x in (scope,key,topic,source): identifier(x)
        if scope not in access.scopes or kind not in KINDS or status not in STATES: raise ContractError('record contract')
        values=vector(values)
        if expected is not None: integer(expected)
        if type(parents) is not tuple or len(parents)>64 or len(set(parents))!=len(parents): raise ContractError('parent contract')
        if (space is None)!=(embedding is None): raise ContractError('embedding space required')
        if space is not None: identifier(space); embedding=vector(embedding,unit=True)
        row=self.db.execute('SELECT id FROM knowledge_heads_v1 WHERE scope=? AND key=?',(scope,key)).fetchone()
        current=None if row is None else row[0]
        if current!=expected: raise ContractError('stale knowledge revision')
        # A derived record must not widen visibility of private/group sources.
        for parent in parents+(() if current is None else (current,)):
            rec=self.read(parent,access)
            if rec.scope not in ('global',scope): raise ContractError('source scope cannot be widened')
        version=1 if current is None else self.read(current,access).version+1
        if space is not None:
            self.db.execute('INSERT OR IGNORE INTO knowledge_spaces_v1 VALUES(?,?)',(space,len(embedding)))
            width=self.db.execute('SELECT width FROM knowledge_spaces_v1 WHERE space=?',(space,)).fetchone()[0]
            if width!=len(embedding): raise ContractError('embedding width mismatch')
        cur=self.db.execute('INSERT INTO knowledge_v1(scope,key,version,topic,kind,status,payload,source,previous,parents,space,embedding,created_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (scope,key,version,topic,kind,status,json.dumps(values),source,current,json.dumps(parents),space,None if embedding is None else json.dumps(embedding),time.time_ns()))
        rid=cur.lastrowid
        self.db.execute('INSERT INTO knowledge_heads_v1 VALUES(?,?,?) ON CONFLICT(scope,key) DO UPDATE SET id=excluded.id',(scope,key,rid))
        return self.read(rid,access)
    def search(self,access,*,topic=None,kind=None,source=None,status='VERIFIED',space=None,query=None,limit=8,scan_limit=4096,since_ns=None,until_ns=None):
        """Current heads only. Exact indexed filters; optional bounded exact cosine scan.
        No claim of semantic alignment: caller supplies compatible versioned embeddings.
        """
        integer(limit,1,128); integer(scan_limit,1,100_000)
        if status is not None and status not in STATES: raise ContractError('status filter')
        if kind is not None and kind not in KINDS: raise ContractError('kind filter')
        if (space is None)!=(query is None): raise ContractError('query space required')
        scopes=access.scopes; where=['k.scope IN ('+','.join('?' for _ in scopes)+')']; args=list(scopes)
        for col,val in [('topic',topic),('kind',kind),('source',source),('status',status),('space',space)]:
            if val is not None: identifier(val); where.append('k.'+col+'=?'); args.append(val)
        if since_ns is not None: where.append('k.created_ns>=?'); args.append(integer(since_ns,0))
        if until_ns is not None: where.append('k.created_ns<=?'); args.append(integer(until_ns,0))
        if since_ns is not None and until_ns is not None and since_ns>until_ns: raise ContractError('invalid time range')
        sql='SELECT k.id,k.embedding FROM knowledge_v1 k JOIN knowledge_heads_v1 h ON h.id=k.id WHERE '+' AND '.join(where)
        if query is None:
            return tuple(r[0] for r in self.db.execute(sql+' ORDER BY k.id DESC LIMIT ?',(*args,limit)))
        q=vector(query,unit=True)
        row=self.db.execute('SELECT width FROM knowledge_spaces_v1 WHERE space=?',(space,)).fetchone()
        if row is None: return ()
        if row[0]!=len(q): raise ContractError('query width mismatch')
        rows=self.db.execute(sql+' ORDER BY k.id LIMIT ?',(*args,scan_limit+1)).fetchall()
        if len(rows)>scan_limit: raise ContractError('embedding scan budget exceeded')
        scores=[(math.fsum(a*b for a,b in zip(q,json.loads(emb))),rid) for rid,emb in rows]
        return tuple(rid for _,rid in sorted(scores,key=lambda t:(-t[0],t[1]))[:limit])
    def link(self,access,src,relation,dst):
        if self.db.in_transaction: raise ContractError('nested public knowledge write')
        identifier(relation); a=self.read(src,access); b=self.read(dst,access)
        if a.scope!=b.scope and b.scope!='global': raise ContractError('relation scope cannot be widened')
        with self.db: self.db.execute('INSERT OR IGNORE INTO knowledge_links_v1 VALUES(?,?,?)',(src,relation,dst))
    def related(self,access,src,relation,*,limit=8):
        self.read(src,access); identifier(relation); integer(limit,1,128)
        scopes=access.scopes
        sql='SELECT l.dst FROM knowledge_links_v1 l JOIN knowledge_v1 k ON k.id=l.dst WHERE l.src=? AND l.relation=? AND k.scope IN ('+','.join('?' for _ in scopes)+') ORDER BY l.dst LIMIT ?'
        return tuple(r[0] for r in self.db.execute(sql,(src,relation,*scopes,limit)))
