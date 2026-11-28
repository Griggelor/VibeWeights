"""Bounded relational knowledge transfer. Host declares a typed identity bridge.
No free-form semantic alignment, autonomous group formation or teacher calls.
"""
from .abi import ContractError
from .environment import identifier
from .knowledge_learning import Domain
from .population import RESIDENT


def derive_relation(learning, *, first_job, second_job, result_domain, job_id,
                    expert_id, bridge_id, oracle, oracle_id, method='DIRECT', steps=80):
    """Infer a two-hop relation from resident weights and learn it additively.

    Oracle is independently supplied and never participates in the forward pass.
    The declared bridge identifies equal intermediate ID spaces (not just widths).
    All positive derived records are committed together after exhaustive checking.
    The subsequent candidate job is separate and may fail without undoing valid facts.
    """
    for value in (job_id,expert_id,bridge_id,oracle_id): identifier(value)
    source=identifier('derived:'+job_id)
    if method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256:
        raise ContractError('learning method/budget')
    if not isinstance(result_domain,Domain) or first_job==second_job:
        raise ContractError('relation contract')
    with learning._guard():
        a,b=learning.job(first_job),learning.job(second_job)
        if any(j['state'] not in RESIDENT or not j['report'] or not j['report']['accepted'] for j in (a,b)):
            raise ContractError('relation parents not approved and resident')
        da,db=(learning._check_sources(j) for j in (a,b))
        d=result_domain
        if len({da.topic,db.topic,d.topic})!=3 or da.labels!=db.entities or (d.entities,d.labels)!=(da.entities,db.labels):
            raise ContractError('relation bridge shape or namespace mismatch')
        if learning.db.execute('SELECT 1 FROM experts WHERE id=?',(expert_id,)).fetchone() or learning.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job_id,expert_id)).fetchone():
            raise ContractError('duplicate learning identity')
        if learning.knowledge.db.execute('SELECT 1 FROM knowledge_v1 WHERE scope=? AND topic=?',('global',d.topic)).fetchone():
            raise ContractError('derived namespace already used')
        # Read only record identity keys for provenance; never read label values
        # to compute the relation. Source checks have independently checked hashes.
        def roots(j):
            return {int(learning.knowledge.read(r,learning.access).key.rsplit(':',1)[1]):r for r in j['spec']['roots']}
        ra,rb=roots(a),roots(b)
        parents=(a['expert_id'],b['expert_id'])
        inferred=[]
        for i in range(-1,d.entities+1):
            mid=da.decode(learning.population.compute(parents[0],da.encode(i)))
            value=None if mid is None else db.decode(learning.population.compute(parents[1],db.encode(mid)))
            expected=oracle(i) if 0<=i<d.entities else None
            d.target(expected)  # reject non-integer or out-of-range oracle labels
            if value!=expected: raise ContractError('independent relation evaluation failed')
            if value is not None:
                if i not in ra or mid not in rb: raise ContractError('derived provenance missing')
                inferred.append((i,value,tuple(dict.fromkeys((ra[i],rb[mid])))))
        if not inferred: raise ContractError('no verified derived facts')
        ids=[]
        for i,value,refs in inferred:
            rec=learning.knowledge._store_locked(learning.access,scope='global',key=f'{d.topic}:entity:{i}',topic=d.topic,values=(value,),source=source,status='VERIFIED',parents=refs)
            ids.append(rec.id)
        metadata={'contract':'two-hop-identity-v1','bridge_id':bridge_id,'parent_jobs':[first_job,second_job],
                  'oracle_id':oracle_id,'evaluated_cases':d.entities+2,'derived_facts':len(ids)}
        learning.record_intake(job_id=job_id,expert_id=expert_id,domain=d,ids=ids,method=method,
                               steps=steps,parents=parents,derivation=metadata)
        learning.knowledge.db.commit()
    m=learning.resume_intake(job_id)
    return m,tuple(ids)


def stage_teacher_knowledge(learning, teacher, *, domain, request, job_id, expert_id,
                            oracle, oracle_id, method='TRAINED', steps=80, initiative_sha256=None):
    """Use a local text-bridge proposal as data, never as verification authority.

    Host authorizes request and supplies independent oracle. Only strict JSON
    with known integer IDs is accepted; no model instructions are executed.
    No remote connection, live-model availability or source license is inferred.
    """
    import json
    import hashlib
    from .bootstrap import BootstrapExpert
    if not isinstance(teacher,BootstrapExpert) or not isinstance(domain,Domain):
        raise ContractError('teacher/domain contract')
    for value in (job_id,expert_id,oracle_id): identifier(value)
    source=identifier('teacher:'+job_id)
    if initiative_sha256 is not None and (type(initiative_sha256) is not str or len(initiative_sha256)!=64 or any(c not in '0123456789abcdef' for c in initiative_sha256)):
        raise ContractError('initiative receipt digest')
    if type(request) is not str or not request.strip() or len(request)>8000:
        raise ContractError('teacher request bounds')
    if method not in ('DIRECT','TRAINED') or type(steps) is not int or not 1<=steps<=256:
        raise ContractError('learning method/budget')
    result=teacher.propose(request+'\nReturn only JSON: {"facts":[{"entity":0,"label":0}]}. '
        f'Entity IDs 0..{domain.entities-1}, label IDs 0..{domain.labels-1}. Omit unknown facts.')
    raw=result.get('proposal')
    if type(raw) is not str or len(raw)>32000 or result.get('verified') is not False:
        raise ContractError('teacher proposal contract')
    def no_duplicates(pairs):
        obj={}
        for k,v in pairs:
            if k in obj: raise ContractError('duplicate JSON key')
            obj[k]=v
        return obj
    try: data=json.loads(raw,object_pairs_hook=no_duplicates)
    except (ValueError,TypeError,RecursionError) as exc: raise ContractError('invalid teacher JSON') from exc
    if type(data) is not dict or set(data)!={'facts'} or type(data['facts']) is not list or not 1<=len(data['facts'])<=domain.entities:
        raise ContractError('teacher facts contract')
    facts={}
    for fact in data['facts']:
        if type(fact) is not dict or set(fact)!={'entity','label'}: raise ContractError('teacher fact shape')
        i,v=fact['entity'],fact['label']
        if type(i) is not int or not 0<=i<domain.entities or i in facts or type(v) is not int:
            raise ContractError('teacher fact identity')
        domain.target(v);facts[i]=v
    with learning._guard():
        if learning.db.execute('SELECT 1 FROM experts WHERE id=?',(expert_id,)).fetchone() or learning.db.execute('SELECT 1 FROM knowledge_jobs_v08 WHERE id=? OR expert_id=?',(job_id,expert_id)).fetchone():
            raise ContractError('duplicate learning identity')
        if learning.knowledge.db.execute('SELECT 1 FROM knowledge_v1 WHERE scope=? AND topic=?',('global',domain.topic)).fetchone():
            raise ContractError('teacher namespace already used')
        # Check whole finite domain. Missing required facts do not become false
        # abstentions; independent oracle defines the scope of this learning job.
        for i in range(domain.entities):
            expected=oracle(i);domain.target(expected)
            if facts.get(i)!=expected: raise ContractError('independent teacher verification failed')
        ids=[]
        for i,value in sorted(facts.items()):
            rec=learning.knowledge._store_locked(learning.access,scope='global',key=f'{domain.topic}:entity:{i}',topic=domain.topic,values=(value,),source=source,status='VERIFIED')
            ids.append(rec.id)
        metadata={'contract':'local-teacher-json-v1','teacher_id':teacher.spec.id,'model':teacher.spec.model,
                  'oracle_id':oracle_id,'request_sha256':hashlib.sha256(request.encode()).hexdigest(),
                  'proposal_sha256':hashlib.sha256(raw.encode()).hexdigest(),'verified_facts':len(ids)}
        if initiative_sha256 is not None: metadata['initiative_sha256']=initiative_sha256
        learning.record_intake(job_id=job_id,expert_id=expert_id,domain=domain,ids=ids,method=method,
                               steps=steps,derivation=metadata)
        learning.knowledge.db.commit()
    return learning.resume_intake(job_id),tuple(ids)
