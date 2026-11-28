"""Train a new frozen prefix readout from verified joint usage, not parent weights."""
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
from vibeweights.usage_adapters import fit_usage_adapter,collect_usage
from probe_real_fragment_social import ARMS,Distractor,train,run
from probe_social_credit import NEW_SENT,NEW_FORM

# New candidate holdout, never used for the parent readouts or social training.
HOLDOUT=[
 ('The choir sang with extraordinary beauty.',1),
 ('Our friends prepared a delightful surprise.',1),
 ('The afternoon at the beach was blissful.',1),
 ('The replacement works flawlessly.',1),
 ('The staff handled our request brilliantly.',1),
 ('The soup was rich and delicious.',1),
 ('I treasure the memories of that trip.',1),
 ('The workshop exceeded all my hopes.',1),
 ('The choir sounded painfully out of tune.',0),
 ('The surprise was deeply upsetting.',0),
 ('The afternoon at the beach was wretched.',0),
 ('The replacement stopped working immediately.',0),
 ('The staff handled our request terribly.',0),
 ('The soup was watery and disgusting.',0),
 ('I shudder at the memories of that trip.',0),
 ('The workshop was a complete waste of time.',0),
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
    cores=[];trainsets=[];parents={};hold=cases(HOLDOUT)
    for i,(name,depth,pooling,train_text,admission_text) in enumerate(ARMS):
        t,a=cases(train_text),cases(admission_text);trainsets.append(t)
        if i==0 and {x for x,_ in hold}&{x for x,_ in t+a+cases(NEW_SENT)}:
            raise RuntimeError('candidate holdout overlap')
        enc=FrozenPrefix(store,depth,pooling=pooling)
        spec=fit_experiment(enc,t,a,topic=name,training_oracle='readout-train-v1',
            holdout_oracle='readout-admission-v1',output=root/name,conditioning='train-centered-radius-v1')
        parents[name]=spec
        if spec['report']['admitted']:
            cores.append(FragmentPortCore(SocialFragmentCore(f'e{i}',FragmentCompetence(enc,root/name)),
                port=i,scale=(1.,.25)[i]))
    report={'source':store.manifest['source'],'parents':parents,'candidate_holdout_texts':HOLDOUT,
            'usage_texts':[NEW_SENT,NEW_FORM],'candidates':{},
            'scope':'host-selected port and prefix, receipt-derived training, not autonomous discovery'}
    if len(cores)!=2:report['status']='PARENT_ADMISSION_FAILED'
    else:
        db=Cooperation(root/'usage.sqlite')
        try:
            pop=SocialPopulation(db,[*[OfflineMemo(c) for c in cores],Distractor()],width=3,
                credit_policy='isolated-activation-v2')
            report['social_training']=train(pop,'source',7,trainsets)
            ids=[];rejected=[]
            for i,(a,b) in enumerate(zip(cases(NEW_SENT),cases(NEW_FORM))):
                ep=f'usage-{i}'
                receipt,_=run(pop,'source',ep,2,a,b,learn=False,seed=i)
                if receipt['verified']:ids.append(ep)
                else:rejected.append(ep)
            report['verified_usage']=ids;report['rejected_usage']=rejected
        finally:db.close()
        db=Cooperation(root/'usage.sqlite')
        try:
            # Require all prespecified source receipts; no cherry-picking successes.
            if len(ids)!=16:report['status']='USAGE_EVIDENCE_INCOMPLETE'
            else:
                training,proof=collect_usage(db,'source',ids,port=0,oracle_id='two-port-utility-v1')
                report['evidence']=proof
                # Both the new 8-block prefix and matched controls use exactly the same receipt labels.
                for mode in ('pretrained','random','embedding-only'):
                    enc=FrozenPrefix(store,8,mode=mode,pooling='token-l2-mean-v1')
                    path=root/('candidate-'+mode)
                    spec=fit_usage_adapter(db,enc,'source',ids,hold,port=0,
                        oracle_id='two-port-utility-v1',holdout_oracle='fresh-usage-holdout-v1',output=path)
                    row={'report':spec['report'],'core':spec['core'],'adapter_sha256':spec['adapter_sha256']}
                    if spec['report']['admitted']:
                        e=FragmentCompetence(enc,path,evidence_store=db)
                        row['reloaded_passed']=sum(e.predict('usage-candidate',x)==y for x,y in hold)
                    report['candidates'][mode]=row
                    del enc;gc.collect()
                report['status']='COMPLETED'
        finally:db.close()
    report['elapsed_seconds']=perf_counter()-start
    _write(root/'report.json',report)
    summary={'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],
        'verified_usage':len(report.get('verified_usage',[])),
        'rejected_usage':len(report.get('rejected_usage',[])),
        'candidates':{k:{'training':v['report']['training']['passed'],'holdout':v['report']['holdout']['passed'],
                          'admitted':v['report']['admitted'],'reloaded_passed':v.get('reloaded_passed')}
                      for k,v in report['candidates'].items()}}
    print(json.dumps(summary,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
