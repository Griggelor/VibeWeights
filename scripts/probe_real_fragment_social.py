"""Real frozen-fragment social trial with separately admitted readouts.

Two native-token input ports and two signed numeric output ports, intentionally
host-defined. No preassigned role, topic match or expert identity in local heads.
A third analytic distractor tests selective participation. No full teacher model.
"""
import argparse
import gc
import json
import random
from pathlib import Path
from time import perf_counter
from vibeweights.qwen3_fragments import MODEL_ID,REVISION,extract,FragmentStore,_hash,_write
from vibeweights.fragment_adapters import FrozenPrefix,fit_experiment,FragmentCompetence
from vibeweights.social_learning import SocialFragmentCore,SocialPopulation
from vibeweights.fragment_social_ports import pack_tokens,FragmentPortCore,OfflineMemo
from vibeweights.coalition import Cooperation
from probe_fragment_conditioning import TRAIN as SENT_TRAIN,HOLDOUT as SENT_ADMISSION

# Form readout: punctuation/sentence form, not a claim of broad syntax expertise.
FORM_TRAIN=[
 ('Where is the station?',1),('Can you open the window?',1),
 ('What does this button do?',1),('When will the bus arrive?',1),
 ('Who wrote the letter?',1),('Is the coffee ready?',1),
 ('How long is the movie?',1),('Did you see the bird?',1),
 ('The station is nearby.',0),('Please open the window.',0),
 ('This button starts the engine.',0),('The bus arrives at noon.',0),
 ('She wrote the letter.',0),('The coffee is ready.',0),
 ('The movie lasts two hours.',0),('I saw the bird.',0),
]
FORM_ADMISSION=[
 ('Which road leads to the lake?',1),('Could you pass the salt?',1),
 ('Why is the door locked?',1),('Have they finished the work?',1),
 ('How much does the ticket cost?',1),('Are the lights still on?',1),
 ('What is inside the box?',1),('Will it rain tomorrow?',1),
 ('The road leads to the lake.',0),('The salt is on the table.',0),
 ('Someone locked the door.',0),('They finished the work.',0),
 ('The ticket costs ten dollars.',0),('The lights are still on.',0),
 ('A book is inside the box.',0),('It will rain tomorrow.',0),
]
SENT_USAGE=[
 ('The orchestra delivered a magnificent performance.',1),
 ('I am delighted with the excellent results.',1),
 ('The new chair is comfortable and beautifully crafted.',1),
 ('Our host welcomed us with warmth and generosity.',1),
 ('The hike was refreshing and absolutely wonderful.',1),
 ('This purchase made me very happy.',1),
 ('The fresh bread smelled heavenly.',1),
 ('The presentation was engaging and informative.',1),
 ('The orchestra gave a dreadful performance.',0),
 ('I am deeply disappointed with the results.',0),
 ('The new chair is uncomfortable and poorly made.',0),
 ('Our host treated us with contempt.',0),
 ('The hike was exhausting and miserable.',0),
 ('I regret this terrible purchase.',0),
 ('The bread was burnt and tasted awful.',0),
 ('The presentation was tedious and confusing.',0),
]
FORM_USAGE=[
 ('Where did you put the keys?',1),('Does the train stop here?',1),
 ('Can I borrow your pencil?',1),('Who is at the gate?',1),
 ('When does the library close?',1),('What caused the noise?',1),
 ('Have you met the new neighbor?',1),('May we sit outside?',1),
 ('The keys are in the drawer.',0),('The train stops here.',0),
 ('I borrowed a pencil.',0),('A visitor is at the gate.',0),
 ('The library closes at six.',0),('The wind caused the noise.',0),
 ('We met the new neighbor.',0),('We may sit outside.',0),
]
ARMS=(('sent',16,'token-l2-mean-v1',SENT_TRAIN,SENT_ADMISSION),
      ('form',6,'last-l2-v1',FORM_TRAIN,FORM_ADMISSION))


class Distractor:
    id='e2'
    output_width=2
    artifact_sha256=_hash({'contract':'constant-reference-v1','value':[0.,.75]})
    def contribute(self,payload):return (0.,.75)


def task(goal,first,second):
    a,ay=first;b,by=second
    z=tuple(float(i==goal) for i in range(3))
    target=(float(2*ay-1) if goal in (0,2) else 0.,
            .25*(2*by-1) if goal in (1,2) else 0.)
    weights=(1.,4.) if goal in (0,1) else (1.,1.)
    def oracle(value):
        error=sum(w*abs(a-b) for w,a,b in zip(weights,value,target))
        return value==target,max(-1.,1.-error/2.)
    return z,pack_tokens(a,b),oracle,target


def run(pop,session,episode,goal,first,second,**kw):
    z,p,oracle,target=task(goal,first,second)
    r=pop.run(session,episode,z,p,oracle,oracle_id='two-port-utility-v1',**kw)
    return r,target


def train(pop,session,seed,trainsets,*,learn=True,shots=True,count=600):
    rng=random.Random(seed);stats=[]
    for i in range(count):
        goal=rng.randrange(3)
        a,b=[rng.choice(s) for s in trainsets]
        r,_=run(pop,session,f'train-{i}',goal,a,b,learn=learn,
            shots=i%3 if shots else 0,seed=rng.randrange(2**63))
        stats.append({'verified':r['verified'],'calls':r['core_calls'],
            'invitations':sum(e['kind']=='INVITE' for e in r['events']),
            'shots':len(r['shot_ids'])})
    return [{'start':i,'episodes':len(stats[i:i+100]),
             **{k:sum(r[k] for r in stats[i:i+100]) for k in ('verified','calls','invitations','shots')}}
             for i in range(0,len(stats),100)]


def evaluate(pop,session,usage,*,prefix='held',recall=True,shots=0,all_active=False,seed=0):
    rng=random.Random(seed);rows=[]
    for goal in range(3):
        for i,(a,b) in enumerate(zip(*usage)):
            r,target=run(pop,session,f'{prefix}-{goal}-{i}',goal,a,b,
                learn=False,recall=recall,shots=shots,all_active=all_active,seed=rng.randrange(2**63))
            rows.append({'goal':goal,'case':i,'target':target,'candidate':r['candidate'],
                         'verified':r['verified'],'self_activated':r['self_activated'],
                         'members':r['members'],'shots':r['shot_ids'],'events':r['events'],
                         'contributions':r['contributions'],'core_calls':r['core_calls']})
    def stats(rr):
        return {'cases':len(rr),'passed':sum(x['verified'] for x in rr),
                'core_calls':sum(x['core_calls'] for x in rr),
                'invitations':sum(sum(e['kind']=='INVITE' for e in x['events']) for x in rr)}
    return {**stats(rows),'by_goal':{str(g):stats([x for x in rows if x['goal']==g]) for g in range(3)},'rows':rows}


def run_seed(root,seed,cores,trainsets,usage):
    def open_pop():
        store=Cooperation(root/f'seed-{seed}.sqlite')
        memo=[OfflineMemo(c) for c in cores]
        return store,SocialPopulation(store,[*memo,Distractor()],width=3),memo
    def counts(memo):
        return {m.id:{'calls':m.calls,'actual_forwards':m.forwards} for m in memo}
    store,pop,memo=open_pop()
    try:
        training={'learned':train(pop,'learned',seed,trainsets),
                  'frozen':train(pop,'frozen',seed,trainsets,learn=False),
                  'no-shots':train(pop,'no-shots',seed,trainsets,shots=False)}
        train_cost=counts(memo)
    finally:store.close()
    # New memo and database connection after restart, no train-cache carryover.
    store,pop,memo=open_pop()
    try:
        held={'learned':evaluate(pop,'learned',usage,seed=seed),
              'without_recall':evaluate(pop,'learned',usage,prefix='no-recall',recall=False,seed=seed),
              'frozen_random_shots':evaluate(pop,'frozen',usage,prefix='random',shots=2,seed=seed),
              'without_training_shots':evaluate(pop,'no-shots',usage,seed=seed),
              'all_active':evaluate(pop,'frozen',usage,prefix='all-active',all_active=True,seed=seed)}
        return {'seed':seed,'training':training,'holdouts':held,
                'train_frozen_compute':train_cost,'evaluation_frozen_compute':counts(memo)}
    finally:store.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-pinned-qwen3',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    start=perf_counter()
    snapshot=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,
        allow_patterns=['config.json','model.safetensors','LICENSE','tokenizer.json',
                        'tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json']))
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
    def cases(rows):return [(tuple(tokenizer.encode(t,add_special_tokens=False)),y) for t,y in rows]
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    extract(snapshot,root/'fragments',model_id=MODEL_ID,revision=REVISION,origin='PRETRAINED',scales=[6,8,16])
    store=FragmentStore(root/'fragments',device=args.device,max_tokens=63)
    trainsets=[cases(SENT_TRAIN),cases(FORM_TRAIN)]
    usage=[cases(SENT_USAGE),cases(FORM_USAGE)]
    reports={}
    for i,(name,depth,pooling,train_text,admission_text) in enumerate(ARMS):
        train=cases(train_text);admission=cases(admission_text)
        if ({x for x,_ in usage[i]} & {x for x,_ in train+admission} or
            len({x for x,_ in usage[i]})!=len(usage[i])):
            raise RuntimeError('social usage/admission/train overlap')
        reports[name]={}
        for mode in ('pretrained','random','embedding-only'):
            enc=FrozenPrefix(store,depth,mode=mode,pooling=pooling)
            reports[name][mode]=fit_experiment(enc,train,admission,topic=name,
                training_oracle='readout-train-v1',holdout_oracle='readout-admission-v1',
                output=root/f'{name}-{mode}',conditioning='train-centered-radius-v1')
            del enc;gc.collect()
        _write(root/'readouts.json',reports)
    report={'source':store.manifest['source'],'device':str(store.device),
            'readouts':reports,'train_texts':[SENT_TRAIN,FORM_TRAIN],
            'admission_texts':[SENT_ADMISSION,FORM_ADMISSION],'usage_texts':[SENT_USAGE,FORM_USAGE],
            'train_tokens':trainsets,'usage_tokens':usage,
            'scope':'structured two-port task, two overlapping Qwen3 prefixes, no autonomous competency discovery'}
    if all(reports[name]['pretrained']['report']['admitted'] for name,*_ in ARMS):
        cores=[]
        for i,(name,depth,pooling,*_) in enumerate(ARMS):
            encoder=FrozenPrefix(store,depth,pooling=pooling)
            expert=FragmentCompetence(encoder,root/f'{name}-pretrained')
            cores.append(FragmentPortCore(SocialFragmentCore(f'e{i}',expert),port=i,scale=(1.,.25)[i]))
        report['social']=[run_seed(root,s,cores,trainsets,usage) for s in (7,19,41)]
        report['status']='COMPLETED'
    else:
        report['social']=[]
        report['status']='READOUT_ADMISSION_FAILED'
    report['elapsed_seconds']=perf_counter()-start
    _write(root/'report.json',report)
    summary={'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
        'readouts':{name:{mode:{'training':v['report']['training']['passed'],
                            'holdout':v['report']['holdout']['passed'],
                            'cases':v['report']['holdout']['cases'],
                            'admitted':v['report']['admitted']} for mode,v in variants.items()}
                    for name,variants in reports.items()},
        'social':[{ 'seed':v['seed'],'holdouts':{k:{j:r[j] for j in ('passed','cases','core_calls','invitations','by_goal')}
                     for k,r in v['holdouts'].items()},'train_frozen_compute':v['train_frozen_compute'],
                     'evaluation_frozen_compute':v['evaluation_frozen_compute']} for v in report['social']]}
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
