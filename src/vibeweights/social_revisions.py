"""Append-only population revisions; genesis and historical receipts unchanged."""
import json
import re
from .abi import ContractError
from .qwen3_fragments import _hash
from .environment import identifier


def create_revision_table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS social_population_revisions_v1(
        session TEXT NOT NULL, revision INTEGER NOT NULL, previous_sha TEXT NOT NULL,
        spec TEXT NOT NULL, spec_sha TEXT NOT NULL, PRIMARY KEY(session,revision),
        UNIQUE(session,spec_sha))""")


def _decode(raw,sha):
    try:
        s=json.loads(raw)
        if (_hash(s)!=sha or s['contract']!='social-episodes-v1'
            or set(s)-{'contract','width','output_width','cores','credit_policy'}
            or s.get('credit_policy','leave-one-out-v1') not in ('leave-one-out-v1','isolated-activation-v2')
            or any(type(s[k]) is not int or not 1<=s[k]<=64 for k in ('width','output_width'))
            or type(s['cores']) is not dict or not 1<=len(s['cores'])<=32
            or any(identifier(k)!=k or type(v) is not str or not re.fullmatch('[0-9a-f]{64}',v)
                   for k,v in s['cores'].items())):
            raise ContractError('social population revision integrity')
    except (ValueError,TypeError,KeyError) as e:
        raise ContractError('social population revision integrity') from e
    return s


def additive(previous,current):
    """No mutation, deletion, ABI change, or replacement of any existing ID."""
    return ({k:v for k,v in previous.items() if k!='cores'}==
            {k:v for k,v in current.items() if k!='cores'}
            and len(current['cores'])>len(previous['cores'])
            and all(current['cores'].get(k)==v for k,v in previous['cores'].items()))


def history(db,session):
    """Genesis plus fully validated chain, at most 31 additive transitions."""
    identifier(session)
    row=db.execute('SELECT contract,spec_sha FROM social_population_v1 WHERE session=?',(session,)).fetchone()
    if row is None:return []
    first=_decode(*row);result=[(first,row[1])]
    # Legacy databases need no migration of their immutable genesis.
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='social_population_revisions_v1'").fetchone() is None:
        return result
    rows=db.execute('SELECT revision,previous_sha,spec,spec_sha FROM social_population_revisions_v1 '
                    'WHERE session=? ORDER BY revision LIMIT 32',(session,)).fetchall()
    if len(rows)>31:raise ContractError('social population revision bound')
    for n,(rev,parent,raw,sha) in enumerate(rows,1):
        spec=_decode(raw,sha)
        if rev!=n or parent!=result[-1][1] or not additive(result[-1][0],spec):
            raise ContractError('social population revision ancestry')
        result.append((spec,sha))
    return result


def historical(chain,receipt):
    """Legacy receipts predate growth; bind those to immutable genesis."""
    if not chain:raise ContractError('social source population missing')
    sha=receipt.get('population_sha256',chain[0][1])
    for spec,digest in chain:
        if sha==digest:return spec,digest
    raise ContractError('social receipt population revision missing')


def append(db,session,expected_sha,current):
    """Caller owns BEGIN IMMEDIATE; compare-and-swap latest revision."""
    chain=history(db,session)
    if not chain or chain[-1][1]!=expected_sha:
        raise ContractError('social population stale revision')
    sha=_hash(current);raw=json.dumps(current,sort_keys=True,separators=(',',':'))
    _decode(raw,sha)
    if not additive(chain[-1][0],current):
        raise ContractError('social population growth must be additive')
    create_revision_table(db)
    db.execute('INSERT INTO social_population_revisions_v1 VALUES(?,?,?,?,?)',
               (session,len(chain),expected_sha,raw,sha))
    return sha
