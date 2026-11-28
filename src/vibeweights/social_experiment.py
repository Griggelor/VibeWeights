"""Prespecified social-head reference: five opaque affine cores, two tasks.

Numeric impulse has two orthogonal goal features, not expert IDs or ideal
membership labels. Outputs are additive 2-vectors. The independent oracle
knows the desired task result; participants never receive that target.
"""
import random
from .coalition import Cooperation
from .social_learning import AffineSocialCore,SocialPopulation


def population(store):
    coefficients=((1.,0.),(0.,.25),(-1.,0.),(0.,-.25),(0.,.75))
    return SocialPopulation(store,[
        AffineSocialCore(f'e{i}',[[a],[b]],[0.,0.]) for i,(a,b) in enumerate(coefficients)
    ],width=2)


def task(kind,magnitude):
    impulse=(1.,0.) if kind==0 else (0.,1.)
    payload=(magnitude,)
    sign=1 if kind==0 else -1
    target=(sign*magnitude,sign*magnitude*.25)
    def oracle(value):
        error=sum(abs(a-b) for a,b in zip(value,target))
        return value==target, max(-1.,1.-error/(2.*magnitude))
    return impulse,payload,oracle


def episode(pop,session,episode_id,kind,magnitude,**kw):
    z,p,oracle=task(kind,magnitude)
    return pop.run(session,episode_id,z,p,oracle,oracle_id='analytic-vector-v1',**kw)


def train(pop,session,seed,*,learn=True,shots=True,count=600):
    rng=random.Random(seed);stats=[]
    for i in range(count):
        kind=rng.randrange(2);shot_seed=rng.randrange(2**63)
        r=episode(pop,session,f'train-{i}',kind,1+i%9,shots=(i%3 if shots else 0),
                  seed=shot_seed,learn=learn)
        stats.append({'verified':r['verified'],'core_calls':r['core_calls'],
                      'shots':len(r['shot_ids']),'invites':sum(e['kind']=='INVITE' for e in r['events'])})
    return [{'start':i,'episodes':len(stats[i:i+100]),
             **{k:sum(r[k] for r in stats[i:i+100]) for k in ('verified','core_calls','shots','invites')}}
            for i in range(0,len(stats),100)]


def evaluate(pop,session,*,prefix='held',recall=True,shots=0,all_active=False,seed=0):
    rng=random.Random(seed);rows=[]
    for kind in (0,1):
        for mag in range(10,22):
            r=episode(pop,session,f'{prefix}-{kind}-{mag}',kind,mag,learn=False,recall=recall,
                      shots=shots,seed=rng.randrange(2**63),all_active=all_active)
            rows.append({'kind':kind,'magnitude':mag,'verified':r['verified'],
                'self_activated':r['self_activated'],'members':r['members'],'shots':r['shot_ids'],
                'invitations':[e for e in r['events'] if e['kind']=='INVITE'],
                'core_calls':r['core_calls'],'candidate':r['candidate']})
    return {'cases':len(rows),'passed':sum(x['verified'] for x in rows),
            'core_calls':sum(x['core_calls'] for x in rows),
            'invitations':sum(len(x['invitations']) for x in rows),'rows':rows}


def run_seed(path,seed):
    store=Cooperation(path)
    try:
        pop=population(store)
        learning=train(pop,'learned',seed)
        frozen=train(pop,'frozen',seed,learn=False)
        no_shots=train(pop,'no-shots',seed,shots=False)
    finally:store.close()
    # Restart: heads and memories must be recovered from SQLite.
    store=Cooperation(path)
    try:
        pop=population(store)
        result={'seed':seed,'training':{'learned':learning,'frozen':frozen,'no_shots':no_shots},
            'holdouts':{
                'learned':evaluate(pop,'learned',seed=seed),
                'learned_without_recall':evaluate(pop,'learned',prefix='no-recall',recall=False,seed=seed),
                'frozen_neutral':evaluate(pop,'frozen',seed=seed),
                'frozen_random_shots':evaluate(pop,'frozen',prefix='random-shots',shots=2,seed=seed),
                'learned_without_training_shots':evaluate(pop,'no-shots',seed=seed),
                'all_active':evaluate(pop,'frozen',prefix='all-active',all_active=True,seed=seed)}}
        return result
    finally:store.close()
