"""Exact-key persistent index with immutable version rows. No semantic retrieval claim."""
import sqlite3
from .abi import Slot, checked, ContractError
class Memory:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=2)
        self.db.executescript('''CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, key INTEGER NOT NULL, version INTEGER NOT NULL, value INTEGER NOT NULL, source TEXT NOT NULL, UNIQUE(key,version)); CREATE INDEX IF NOT EXISTS lookup_key ON records(key,version DESC);''')
    def write(self,key,value,source):
        checked(Slot.KEY,key); checked(Slot.Y,value)
        if not isinstance(source,str) or not source.strip(): raise ContractError('source required')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            cur=self.db.execute('INSERT INTO records(key,version,value,source) SELECT ?,COALESCE(MAX(version),0)+1,?,? FROM records WHERE key=?',(key,value,source,key)); rid=cur.lastrowid; checked(Slot.RECORD,rid)
        return rid
    def update(self,key,value,source,*,expected_id):
        """Append-only compare-and-swap; prevents a stale writer from overwriting latest."""
        checked(Slot.KEY,key); checked(Slot.Y,value); checked(Slot.RECORD,expected_id)
        if not isinstance(source,str) or not source.strip(): raise ContractError('source required')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.search(key)!=expected_id: raise ContractError('stale memory version')
            cur=self.db.execute('INSERT INTO records(key,version,value,source) SELECT ?,MAX(version)+1,?,? FROM records WHERE key=?',(key,value,source,key))
            rid=cur.lastrowid; checked(Slot.RECORD,rid)
        return rid
    def provenance(self,rid):
        checked(Slot.RECORD,rid)
        row=self.db.execute('SELECT key,version,source FROM records WHERE id=?',(rid,)).fetchone()
        if row is None: raise ContractError('record missing')
        return {'key':row[0],'version':row[1],'source':row[2]}
    def search(self,key):
        checked(Slot.KEY,key); row=self.db.execute('SELECT id FROM records WHERE key=? ORDER BY version DESC LIMIT 1',(key,)).fetchone()
        if row is None: raise ContractError('memory key not found')
        return checked(Slot.RECORD,row[0])
    def read(self,rid,key):
        checked(Slot.RECORD,rid); checked(Slot.KEY,key); row=self.db.execute('SELECT value FROM records WHERE id=? AND key=?',(rid,key)).fetchone()
        if row is None: raise ContractError('record/key mismatch')
        return checked(Slot.Y,row[0])
    def close(self): self.db.close()
