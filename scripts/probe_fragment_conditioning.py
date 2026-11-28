"""A1 follow-up: prespecified conditioning and multi-scale controls.

New authored cases; previous holdout informed the hypothesis but is not reused.
No model selection or threshold tuning on this holdout.
"""
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from vibeweights.qwen3_fragments import MODEL_ID, REVISION, extract, FragmentStore, _write
from vibeweights.fragment_adapters import FrozenPrefix, fit_experiment, FragmentCompetence

TRAIN=[
 ("The gardener was delighted by the beautiful flowers.",1),
 ("The delivery driver was helpful and courteous.",1),
 ("I loved every moment of the exhibition.",1),
 ("The software saved me hours of tedious work.",1),
 ("The hotel room was spotless and comfortable.",1),
 ("The children laughed happily at the puppet show.",1),
 ("The new headphones sound wonderfully clear.",1),
 ("We were impressed by the team's excellent work.",1),
 ("The garden looked neglected and ugly.",0),
 ("The driver was hostile and unhelpful.",0),
 ("I could not stand the dreadful exhibition.",0),
 ("The software crashed and wasted my entire morning.",0),
 ("The hotel room was filthy and uncomfortable.",0),
 ("The children were frightened by the awful show.",0),
 ("The headphones produce a horrible distorted sound.",0),
 ("The team's careless work was a huge disappointment.",0),
]
HOLDOUT=[
 ("The mountain walk was a joyful adventure.",1),
 ("My appointment went smoothly and I felt respected.",1),
 ("This jacket is beautifully made and fits perfectly.",1),
 ("The museum guide made the visit fascinating.",1),
 ("The mechanic did an outstanding job.",1),
 ("The bakery's pastries were absolutely superb.",1),
 ("The festival was a cheerful and memorable occasion.",1),
 ("I am grateful for the thoughtful support.",1),
 ("The mountain walk turned into a miserable ordeal.",0),
 ("The appointment was chaotic and humiliating.",0),
 ("This jacket fell apart after one wearing.",0),
 ("The museum tour was painfully dull.",0),
 ("The mechanic made the problem even worse.",0),
 ("The pastries were stale and tasted terrible.",0),
 ("The festival was an unpleasant disaster.",0),
 ("I was upset by the dismissive treatment.",0),
]
# 2x2 at six blocks, plus a matched random control and a deeper scale pair.
ARMS=[
 ("p6-baseline","pretrained",6,"mean-l2-v1","none"),
 ("p6-pool","pretrained",6,"token-l2-mean-v1","none"),
 ("p6-condition","pretrained",6,"mean-l2-v1","train-centered-radius-v1"),
 ("p6-combined","pretrained",6,"token-l2-mean-v1","train-centered-radius-v1"),
 ("r6-combined","random",6,"token-l2-mean-v1","train-centered-radius-v1"),
 ("e-combined","embedding-only",6,"token-l2-mean-v1","train-centered-radius-v1"),
 ("p16-combined","pretrained",16,"token-l2-mean-v1","train-centered-radius-v1"),
 ("r16-combined","random",16,"token-l2-mean-v1","train-centered-radius-v1"),
]


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
    train=[(tokenizer.encode(t,add_special_tokens=False),y) for t,y in TRAIN]
    test=[(tokenizer.encode(t,add_special_tokens=False),y) for t,y in HOLDOUT]
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    extract(snapshot,root/'fragments',model_id=MODEL_ID,revision=REVISION,
            origin='PRETRAINED',scales=[6,8,16])
    store=FragmentStore(root/'fragments',device=args.device)
    models={}
    for name,mode,depth,pooling,conditioning in ARMS:
        encoder=FrozenPrefix(store,depth,mode=mode,pooling=pooling)
        spec=fit_experiment(encoder,train,test,topic='sentiment',
            training_oracle='authored-train-v2',holdout_oracle='authored-test-v2',
            output=root/name,steps=64,conditioning=conditioning)
        if spec['report']['admitted']:
            expert=FragmentCompetence(encoder,root/name)
            assert [expert.predict('sentiment',x,priority=True) for x,_ in test]==spec['report']['holdout']['predictions']
            assert expert.predict('unrelated',test[0][0]) is None
            del expert
        models[name]=spec
        del encoder
        gc.collect()
        _write(root/'partial-report.json',{'completed_arms':models})
    report={'status':'COMPLETED','source':store.manifest['source'],'device':str(store.device),
            'training_texts':TRAIN,'holdout_texts':HOLDOUT,'training_tokens':train,
            'holdout_tokens':test,'arms':ARMS,'models':models,
            'elapsed_seconds':perf_counter()-start,
            'scope':'prespecified authored follow-up, all arms reported, no independent benchmark or autonomous assimilation'}
    _write(root/'report.json',report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
