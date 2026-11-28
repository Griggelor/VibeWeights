"""Real Qwen3 candidates independently discover learning opportunities."""
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from vibeweights.coalition import Cooperation
from vibeweights.qwen3_fragments import MODEL_ID,REVISION,extract,FragmentStore,_write
from vibeweights.fragment_adapters import FrozenPrefix,FragmentCompetence,fit_experiment
from vibeweights.fragment_social_ports import FragmentPortCore,OfflineMemo
from vibeweights.social_learning import SocialFragmentCore,SocialPopulation
from vibeweights.local_learning import propose_usage,fit_requested_adapter
from probe_real_fragment_social import ARMS,Distractor,train,run,SENT_USAGE,FORM_USAGE
from probe_social_credit import NEW_SENT,NEW_FORM
from probe_usage_adapter import HOLDOUT as PREVIOUS_HOLDOUT

# Newly authored final acceptance set; never passed to the discovery function.
FINAL_SENT=[
 ('The hotel room was spotless and inviting.',1),
 ('The children adored the puppet show.',1),
 ('The repair exceeded my expectations.',1),
 ('The concert was an absolute joy.',1),
 ('The meal tasted wonderfully fresh.',1),
 ('The courier was courteous and prompt.',1),
 ('The novel kept me happily absorbed.',1),
 ('The team produced an outstanding result.',1),
 ('The hotel room was revolting and cramped.',0),
 ('The children loathed the puppet show.',0),
 ('The repair was a disastrous failure.',0),
 ('The concert was unbearable.',0),
 ('The meal tasted stale and foul.',0),
 ('The courier was rude and careless.',0),
 ('The novel bored me to tears.',0),
 ('The team produced a shameful result.',0),
]
FINAL_FORM=[
 ('Which shelf holds the dictionaries?',1),
 ('Did the oven stop heating?',1),
 ('What time does the ferry depart?',1),
 ('Could we take a different route?',1),
 ('Why are the curtains closed?',1),
 ('Has the parcel been delivered?',1),
 ('Where will they hold the exhibition?',1),
 ('Is anyone using this seat?',1),
 ('The dictionaries are on the top shelf.',0),
 ('The oven stopped heating.',0),
 ('The ferry departs at seven.',0),
 ('We took a different route.',0),
 ('The curtains are closed.',0),
 ('The parcel has been delivered.',0),
 ('They will hold the exhibition downtown.',0),
 ('Someone is using this seat.',0),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-pinned-qwen3',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
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
    store=FragmentStore(root/'fragments',device='cpu',max_tokens=63)
    cores=[];trainsets=[];parents={};final=[cases(FINAL_SENT),cases(FINAL_FORM)]
    usage=[cases(SENT_USAGE+NEW_SENT),cases(FORM_USAGE+NEW_FORM)]
    for i,(name,depth,pooling,train_text,admission_text) in enumerate(ARMS):
        t,a=cases(train_text),cases(admission_text);trainsets.append(t)
        prior=t+a+usage[i]+(cases(PREVIOUS_HOLDOUT) if i==0 else [])
        if {x for x,_ in final[i]}&{x for x,_ in prior}:raise RuntimeError('final overlap')
        enc=FrozenPrefix(store,depth,pooling=pooling)
        spec=fit_experiment(enc,t,a,topic=name,training_oracle='readout-train-v1',
            holdout_oracle='readout-admission-v1',output=root/name,conditioning='train-centered-radius-v1')
        parents[name]=spec
        if spec['report']['admitted']:
            cores.append(FragmentPortCore(SocialFragmentCore(f'e{i}',FragmentCompetence(enc,root/name)),
                port=i,scale=(1.,.25)[i]))
    report={'source':store.manifest['source'],'parents':parents,'final_texts':[FINAL_SENT,FINAL_FORM],
            'source_usage_texts':[SENT_USAGE+NEW_SENT,FORM_USAGE+NEW_FORM],'arms':{},
            'scope':'local evidence scan and requests; fixed two-port ABI, host-authorized candidate set and independent oracle'}
    if len(cores)!=2:report['status']='PARENT_ADMISSION_FAILED'
    else:
        db=Cooperation(root/'usage.sqlite')
        try:
            pop=SocialPopulation(db,[*[OfflineMemo(c) for c in cores],Distractor()],width=3,
                credit_policy='isolated-activation-v2')
            report['social_training']=train(pop,'source',7,trainsets)
            receipts=[]
            for i,(a,b) in enumerate(zip(*usage)):
                receipt,_=run(pop,'source',f'usage-{i}',2,a,b,learn=False,seed=i)
                receipts.append({'episode':receipt['episode'],'verified':receipt['verified']})
            report['usage_episodes']=receipts
        finally:db.close()
        db=Cooperation(root/'usage.sqlite')
        try:
            # Candidate set is fixed before discovery. Each explores both ports.
            for mode in ('pretrained','random','embedding-only'):
                enc=FrozenPrefix(store,8,mode=mode,pooling='token-l2-mean-v1')
                requests=propose_usage(db,enc,'source',oracle_id='two-port-utility-v1',
                                       scan_limit=1024,case_budget=64)
                arms=[]
                for req in requests:
                    item={'request':req,'candidate':None}
                    if req['requested']:
                        path=root/f'candidate-{mode}-port{req["port"]}'
                        spec=fit_requested_adapter(db,enc,req['id'],final[req['port']],
                            holdout_oracle='local-learning-final-v1',output=path)
                        row={'report':spec['report'],'core':spec['core'],
                             'adapter_sha256':spec['adapter_sha256']}
                        if spec['report']['admitted']:
                            e=FragmentCompetence(enc,path,evidence_store=db)
                            row['reloaded_passed']=sum(e.predict('usage-candidate',x)==y for x,y in final[req['port']])
                            del e
                        item['candidate']=row
                    arms.append(item)
                report['arms'][mode]=arms
                del enc;gc.collect()
            report['status']='COMPLETED'
        finally:db.close()
    report['elapsed_seconds']=perf_counter()-start
    _write(root/'report.json',report)
    summary={'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
        'usage_verified':sum(v['verified'] for v in report.get('usage_episodes',[])),
        'usage_cases':len(report.get('usage_episodes',[])),
        'arms':{mode:[{'port':v['request']['port'],'status':v['request']['status'],
            'evidence_classes':v['request']['coverage']['class_counts'],
            'calibration':v['request'].get('probe',{}).get('calibration',{}).get('passed'),
            'holdout':None if v['candidate'] is None else v['candidate']['report']['holdout']['passed'],
            'admitted':False if v['candidate'] is None else v['candidate']['report']['admitted'],
            'reloaded_passed':None if v['candidate'] is None else v['candidate'].get('reloaded_passed')}
            for v in rows] for mode,rows in report['arms'].items()}}
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
