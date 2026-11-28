"""Matched old vs isolated self-credit policies, new authored usage cases."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from vibeweights.qwen3_fragments import MODEL_ID,REVISION,extract,FragmentStore,_write
from vibeweights.fragment_adapters import FrozenPrefix,fit_experiment,FragmentCompetence
from vibeweights.fragment_social_ports import FragmentPortCore
from vibeweights.social_learning import SocialFragmentCore
from probe_real_fragment_social import ARMS,SENT_USAGE,FORM_USAGE,run_seed

# Never used in adapter admission or dev29 evaluation. Chosen before this run.
NEW_SENT=[
 ('The pianist gave an enchanting recital.',1),
 ('The parcel arrived in perfect condition.',1),
 ('The thoughtful gift made my day.',1),
 ('Our weekend getaway was wonderfully relaxing.',1),
 ('The technician resolved everything efficiently.',1),
 ('The garden was a gorgeous peaceful retreat.',1),
 ('The lecture was fascinating and clear.',1),
 ('We celebrated a truly splendid achievement.',1),
 ('The recital was grating and unpleasant.',0),
 ('The parcel arrived crushed and unusable.',0),
 ('The thoughtless remark ruined my day.',0),
 ('Our weekend getaway was a miserable ordeal.',0),
 ('The technician left everything broken.',0),
 ('The garden was a filthy neglected mess.',0),
 ('The lecture was confusing and painfully dull.',0),
 ('We suffered a humiliating defeat.',0),
]
NEW_FORM=[
 ('Which platform serves the airport?',1),
 ('Can you hear the music?',1),
 ('Why did the alarm ring?',1),
 ('Is the shop still open?',1),
 ('Where have you left your coat?',1),
 ('Would you like another cup?',1),
 ('When can we begin the meeting?',1),
 ('Who moved the furniture?',1),
 ('The platform serves the airport.',0),
 ('I can hear the music.',0),
 ('The alarm rang at dawn.',0),
 ('The shop is still open.',0),
 ('Your coat is on the chair.',0),
 ('I would like another cup.',0),
 ('We can begin the meeting now.',0),
 ('The movers rearranged the furniture.',0),
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
    usage=[cases(NEW_SENT),cases(NEW_FORM)]
    prior=[cases(SENT_USAGE),cases(FORM_USAGE)]
    trainsets=[];reports={};cores=[]
    for i,(name,depth,pooling,train_text,admission_text) in enumerate(ARMS):
        train=cases(train_text);admit=cases(admission_text);trainsets.append(train)
        if {x for x,_ in usage[i]} & {x for x,_ in train+admit+prior[i]} or len({x for x,_ in usage[i]})!=len(usage[i]):
            raise RuntimeError('fresh usage overlap')
        enc=FrozenPrefix(store,depth,pooling=pooling)
        spec=fit_experiment(enc,train,admit,topic=name,training_oracle='readout-train-v1',
            holdout_oracle='readout-admission-v1',output=root/name,conditioning='train-centered-radius-v1')
        reports[name]=spec
        if spec['report']['admitted']:
            expert=FragmentCompetence(enc,root/name)
            cores.append(FragmentPortCore(SocialFragmentCore(f'e{i}',expert),port=i,scale=(1.,.25)[i]))
    report={'source':store.manifest['source'],'readouts':reports,'new_usage_texts':[NEW_SENT,NEW_FORM],
            'usage_tokens':usage,'previous_usage_excluded':True,'arms':{},
            'scope':'matched credit-policy comparison on new authored structured tasks; not autonomous competence discovery'}
    if len(cores)==2:
        for policy in ('leave-one-out-v1','isolated-activation-v2'):
            target=root/policy;target.mkdir()
            report['arms'][policy]=[run_seed(target,s,cores,trainsets,usage,credit_policy=policy) for s in (7,19,41)]
        report['status']='COMPLETED'
    else:report['status']='READOUT_ADMISSION_FAILED'
    report['elapsed_seconds']=perf_counter()-start
    _write(root/'report.json',report)
    summary={'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
        'admissions':{k:v['report']['admitted'] for k,v in reports.items()},
        'arms':{policy:[{'seed':v['seed'],'holdouts':{k:{j:r[j] for j in ('passed','cases','core_calls','invitations','by_goal')}
             for k,r in v['holdouts'].items()}} for v in rows] for policy,rows in report['arms'].items()}}
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
