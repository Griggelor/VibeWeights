"""Experiential local social heads, using the existing Cooperation database.

Bounded additive numeric reference, not a general latent ABI. No competence
labels, routing targets or peer IDs in head inputs. The oracle assesses actual
group outputs and leave-one-out counterfactuals; only local observations train.
"""
import json
import math
import random
import re
import torch
from .abi import ContractError
from .coalition import Cooperation, Op
from .environment import identifier
from .qwen3_fragments import _hash

CONTRACT='social-episodes-v1'
SELF_THRESHOLD=.35
INVITE_THRESHOLD=.2
ACCEPT_THRESHOLD=-.1
CREDIT_POLICIES=('leave-one-out-v1','isolated-activation-v2')


def _vector(x,*,limit=128):
    if type(x) not in (tuple,list) or not 1<=len(x)<=limit or any(
        type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for v in x):
        raise ContractError('social numeric vector')
    return tuple(x)


class AffineSocialCore:
    """Immutable analytic reference core; its coefficients are never head inputs."""
    def __init__(self,id,matrix,bias):
        self.id=identifier(id)
        w=torch.tensor(matrix,dtype=torch.float32);b=torch.tensor(bias,dtype=torch.float32)
        if w.ndim!=2 or b.shape!=(w.shape[0],) or not 1<=w.shape[0]<=64 or not 1<=w.shape[1]<=128 or not torch.isfinite(w).all() or not torch.isfinite(b).all():
            raise ContractError('social core shape')
        self._w=w.clone();self._b=b.clone()
        self.output_width=w.shape[0]
        self.artifact_sha256=_hash({'op':'affine','w':w.tolist(),'b':b.tolist()})

    @torch.inference_mode()
    def contribute(self,payload):
        if _hash({'op':'affine','w':self._w.tolist(),'b':self._b.tolist()})!=self.artifact_sha256:
            raise ContractError('social core changed')
        if len(payload)!=self._w.shape[1]:raise ContractError('social affine input width')
        return tuple((self._w@torch.tensor(payload,dtype=torch.float32)+self._b).tolist())


class SocialFragmentCore:
    """Bridge the admitted A1 readout to social participation, without topic gate.

    Only previously admitted experimental fragments qualify; the old metadata
    head is not consulted. The new social head has no topic label as a feature.
    Binary prediction is a one-dimensional additive reference contribution.
    """
    def __init__(self,id,competence):
        from .fragment_adapters import FragmentCompetence
        if not isinstance(competence,FragmentCompetence):raise ContractError('admitted fragment required')
        self.id=identifier(id);self._expert=competence
        self.output_width=1
        self.artifact_sha256=_hash(competence.m)
        self._tensor_sha=_hash({k:v.tolist() for k,v in sorted(competence.t.items())})

    @torch.inference_mode()
    def contribute(self,payload):
        e=self._expert
        if (_hash(e.m)!=self.artifact_sha256 or
            _hash({k:v.tolist() for k,v in sorted(e.t.items())})!=self._tensor_sha):
            raise ContractError('social fragment adapter changed')
        if any(type(x) is not int for x in payload):raise ContractError('fragment token input')
        x=e.encoder.features(payload)
        return (float((e.t['w']@x+e.t['b']).item()>0),)


class SocialPopulation:
    """Local linear utility heads + learned per-peer invitation heads.

    All see the same unit-bounded numeric impulse. Selection by local scores,
    no central ranking. Additive group execution is a restricted test ABI.
    A shot is an explicit experimental inclusion, not a forced positive reward.
    """
    def __init__(self,cooperation,cores,*,width,credit_policy='leave-one-out-v1'):
        if not isinstance(cooperation,Cooperation) or type(width) is not int or not 1<=width<=64:
            raise ContractError('social store/width')
        if credit_policy not in CREDIT_POLICIES:raise ContractError('social credit policy')
        self.db=cooperation.db;self.width=width;self.credit_policy=credit_policy
        cores=tuple(cores)
        if not 1<=len(cores)<=32:raise ContractError('social population bounds')
        self.cores={}
        for c in cores:
            if (identifier(c.id) in self.cores or not isinstance(c.artifact_sha256,str) or
                not re.fullmatch('[0-9a-f]{64}',c.artifact_sha256) or type(c.output_width) is not int or
                not 1<=c.output_width<=64 or not callable(c.contribute)):
                raise ContractError('social core contract')
            self.cores[c.id]=c
        widths={c.output_width for c in cores}
        if len(widths)!=1:raise ContractError('social common output width')
        self.output_width=widths.pop()
        self.identities={k:c.artifact_sha256 for k,c in sorted(self.cores.items())}
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS social_episodes_v1(
          session TEXT NOT NULL, episode TEXT NOT NULL, spec_sha TEXT NOT NULL,
          receipt TEXT NOT NULL, receipt_sha TEXT NOT NULL, PRIMARY KEY(session,episode));
        CREATE TABLE IF NOT EXISTS social_population_v1(
          session TEXT PRIMARY KEY, contract TEXT NOT NULL, spec_sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS social_heads_v1(
          session TEXT NOT NULL, owner TEXT NOT NULL, peer TEXT NOT NULL,
          revision INTEGER NOT NULL, episode TEXT NOT NULL, previous TEXT NOT NULL,
          weights TEXT NOT NULL, sha TEXT NOT NULL,
          PRIMARY KEY(session,owner,peer,revision));
        ''')
        self.spec={'contract':CONTRACT,'width':width,'output_width':self.output_width,'cores':self.identities}
        # Preserve default v1 session identity and historical receipts.
        if credit_policy!='leave-one-out-v1':self.spec['credit_policy']=credit_policy

    def _bind(self,session):
        spec=json.dumps(self.spec,sort_keys=True,separators=(',',':'));sha=_hash(self.spec)
        row=self.db.execute('SELECT contract,spec_sha FROM social_population_v1 WHERE session=?',(session,)).fetchone()
        if row is None:self.db.execute('INSERT INTO social_population_v1 VALUES(?,?,?)',(session,spec,sha))
        elif row!=(spec,sha):raise ContractError('social session population changed')

    def _head(self,session,owner,peer):
        row=self.db.execute('SELECT revision,episode,previous,weights,sha FROM social_heads_v1 WHERE session=? AND owner=? AND peer=? ORDER BY revision DESC LIMIT 1',(session,owner,peer)).fetchone()
        if row is None:return torch.zeros(self.width,dtype=torch.float32),0,''
        rev,episode,previous,raw,sha=row
        v=json.loads(raw)
        if _hash([session,owner,peer,rev,episode,previous,v])!=sha or len(v)!=self.width:
            raise ContractError('social head integrity')
        if rev>1:
            prior=self.db.execute('SELECT sha FROM social_heads_v1 WHERE session=? AND owner=? AND peer=? AND revision=?',(session,owner,peer,rev-1)).fetchone()
            if prior!=(previous,):raise ContractError('social head ancestry')
        elif previous:raise ContractError('social head ancestry')
        w=torch.tensor(v,dtype=torch.float32)
        if not torch.isfinite(w).all():raise ContractError('social nonfinite head')
        return w,rev,sha

    def _update(self,session,episode,owner,peer,x,reward,lr):
        w,rev,prior=self._head(session,owner,peer)
        # One local squared-error SGD step on observed marginal utility.
        score=float(w@x)
        w=w+lr*(reward-score)*x
        if not torch.isfinite(w).all():raise ContractError('social nonfinite update')
        v=w.tolist();sha=_hash([session,owner,peer,rev+1,episode,prior,v])
        self.db.execute('INSERT INTO social_heads_v1 VALUES(?,?,?,?,?,?,?,?)',
            (session,owner,peer,rev+1,episode,prior,json.dumps(v,separators=(',',':')),sha))
        return {'owner':owner,'peer':peer,'revision':rev+1,'sha256':sha,'reward':reward,'before':score,'after':float(w@x)}

    def run(self,session,episode,impulse,payload,oracle,*,oracle_id,shots=0,seed=0,
            learn=True,recall=True,all_active=False,member_budget=32,invite_budget=128):
        """One atomic evidence/learning episode; trusted independent oracle.

        Oracle returns (exact_verified: bool, utility: float in [-1,1]).
        It receives only the additive group value. No oracle calls before
        participation is fixed. It cannot provide an ideal member assignment.
        """
        for v in (session,episode,oracle_id):identifier(v)
        z=_vector(impulse,limit=64);payload=_vector(payload)
        if len(z)!=self.width:raise ContractError('social impulse width')
        x=torch.tensor(z,dtype=torch.float32)
        if not 0<float(x.norm())<=1.000001:raise ContractError('social unit impulse')
        if (type(shots) is not int or not 0<=shots<=32 or type(seed) is not int or not 0<=seed<2**63 or
            type(member_budget) is not int or not 1<=member_budget<=32 or type(invite_budget) is not int or not 0<=invite_budget<=1024 or
            any(type(v) is not bool for v in (learn,recall,all_active)) or not callable(oracle)):
            raise ContractError('social episode budget')
        if {k:c.artifact_sha256 for k,c in sorted(self.cores.items())}!=self.identities:
            raise ContractError('social core identity changed')
        spec={'population':self.spec,'impulse':list(z),'payload':list(payload),'oracle':oracle_id,
              'shots':shots,'seed':seed,'learn':learn,'recall':recall,'all_active':all_active,
              'member_budget':member_budget,'invite_budget':invite_budget}
        spec_sha=_hash(spec)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._bind(session)
            prior=self.db.execute('SELECT spec_sha,receipt,receipt_sha FROM social_episodes_v1 WHERE session=? AND episode=?',(session,episode)).fetchone()
            if prior:
                receipt=json.loads(prior[1])
                if prior[0]!=spec_sha or _hash(receipt)!=prior[2]:raise ContractError('social episode identity/integrity')
                self.db.commit()
                return {**receipt,'historical':True}
            scores={k:float(self._head(session,k,'')[0]@x) for k in self.identities}
            members=set(k for k,v in scores.items() if all_active or v>SELF_THRESHOLD)
            initial=sorted(members);events=[];attempted=set()
            def emit(op,owner=None,peer=None):
                events.append({'kind':op.name,'owner':owner,'peer':peer})
            for k in initial:emit(Op.JOIN,k)
            if len(members)>member_budget:raise ContractError('social member budget')
            # Only owners already participating may invite from their own
            # learned peer heads. Recipients independently accept or abstain.
            if recall:
                while len(attempted)<invite_budget:
                    before=set(members)
                    for owner in sorted(before):
                        candidates=self.db.execute('SELECT DISTINCT peer FROM social_heads_v1 WHERE session=? AND owner=? AND peer<>? ORDER BY peer',(session,owner,'')).fetchall()
                        for (peer,) in candidates:
                            if peer not in self.cores or peer in members or (owner,peer) in attempted:continue
                            if float(self._head(session,owner,peer)[0]@x)<=INVITE_THRESHOLD:continue
                            if len(attempted)>=invite_budget:break
                            attempted.add((owner,peer));emit(Op.INVITE,owner,peer)
                            if scores[peer]>=ACCEPT_THRESHOLD:
                                if len(members)>=member_budget:raise ContractError('social member budget')
                                members.add(peer);emit(Op.JOIN,peer,owner)
                    if before==members:break
            # Random exploration, independent of competence scores; only
            # inactive candidates, no repeat, and bounded by remaining budget.
            available=sorted(set(self.cores)-members)
            shot_ids=random.Random(seed).sample(available,min(shots,len(available),member_budget-len(members)))
            for peer in shot_ids:
                emit(Op.SHOT,peer);members.add(peer)
            contributions={}
            for owner in sorted(members):
                v=_vector(self.cores[owner].contribute(payload),limit=64)
                if len(v)!=self.output_width:raise ContractError('social output width')
                contributions[owner]=list(v);emit(Op.PROBE if owner in shot_ids else Op.PROPOSE,owner)
            def value(exclude=None):
                return [math.fsum(v[i] for k,v in contributions.items() if k!=exclude) for i in range(self.output_width)]
            def assess(v):
                ok,u=oracle(tuple(v))
                if type(ok) is not bool or type(u) not in (float,int) or not math.isfinite(u) or not -1<=u<=1:
                    raise ContractError('social oracle contract')
                return {'verified':ok,'utility':float(u)}
            total=value();full=assess(total)
            without={k:assess(value(k)) for k in contributions}
            marginal={k:max(-1.,min(1.,full['utility']-v['utility'])) for k,v in without.items()}
            isolated=None;self_credit=marginal
            if self.credit_policy=='isolated-activation-v2':
                # Separate the question 'would my own contribution help an
                # empty workspace?' from 'did this partner help this group?'.
                # Reuse actual contributions; no additional core forwards.
                empty=assess([0.]*self.output_width)
                solo={k:assess(v) for k,v in contributions.items()}
                self_credit={k:max(-1.,min(1.,v['utility']-empty['utility'])) for k,v in solo.items()}
                isolated={'empty':empty,'solo':solo,'credit':self_credit}
            changes=[]
            if learn:
                for k,r in self_credit.items():
                    changes.append(self._update(session,episode,k,'',x,r,.15))
                for owner in sorted(members):
                    for peer in sorted(members-{owner}):
                        # Positive edge needs verified coalition and both
                        # members' actual positive marginal contributions.
                        positive=full['verified'] and marginal[owner]>0 and marginal[peer]>0
                        # Negative edge only for a harmful peer in a normal
                        # coalition, never mere unsuccessful shot co-occurrence.
                        negative=not shot_ids and marginal[peer]<0
                        if positive or negative:
                            changes.append(self._update(session,episode,owner,peer,x,1. if positive else -1.,.2))
            receipt={'contract':CONTRACT,'session':session,'episode':episode,'spec_sha256':spec_sha,
                     'oracle_id':oracle_id,'impulse':list(z),'payload':list(payload),'scores':scores,
                     'self_activated':initial,'shot_ids':shot_ids,'members':sorted(members),'events':events,
                     'contributions':contributions,'candidate':total,'verified':full['verified'],
                     'released':total if full['verified'] else None,'utility':full['utility'],
                     'without':without,'marginal':marginal,'updates':changes,
                     'core_calls':len(contributions),'oracle_calls':1+len(without)}
            if isolated is not None:
                receipt['isolated_activation']=isolated
                receipt['oracle_calls']+=1+len(contributions)
            self.db.execute('INSERT INTO social_episodes_v1 VALUES(?,?,?,?,?)',
                (session,episode,spec_sha,json.dumps(receipt,sort_keys=True,separators=(',',':')),_hash(receipt)))
            self.db.commit()
            return {**receipt,'historical':False}
        except BaseException:
            self.db.rollback()
            raise
