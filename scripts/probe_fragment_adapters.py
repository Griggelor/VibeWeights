"""A1 controlled frozen-fragment experiment, including negative controls."""
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from vibeweights.qwen3_fragments import MODEL_ID, REVISION, extract, FragmentStore, _write
from vibeweights.fragment_adapters import FrozenPrefix, fit_experiment, FragmentCompetence

# Fixed before running this experiment. Distinct sentences and token sequences.
# Labels are deliberately simple host-authored sentiment judgments, not a benchmark.
TRAIN=[
 ("I enjoyed the concert and left smiling.",1),
 ("The meal was delicious and the staff were kind.",1),
 ("The repair worked perfectly.",1),
 ("We had a wonderful holiday by the sea.",1),
 ("The book was inspiring and a pleasure to read.",1),
 ("Everything arrived safely and on time.",1),
 ("I hated the concert and left disappointed.",0),
 ("The meal was awful and the staff were rude.",0),
 ("The repair failed completely.",0),
 ("Our holiday was miserable and exhausting.",0),
 ("The book was boring and unpleasant.",0),
 ("Everything arrived broken and late.",0),
]
HOLDOUT=[
 ("What a delightful afternoon with friends.",1),
 ("This little device exceeded my expectations.",1),
 ("The service made me feel genuinely welcome.",1),
 ("I am thrilled with how the project turned out.",1),
 ("The cake tasted amazing.",1),
 ("Her performance was brilliant.",1),
 ("What a dreadful experience from start to finish.",0),
 ("This device is a frustrating waste of money.",0),
 ("The service left me angry and ignored.",0),
 ("I regret spending so much time on this project.",0),
 ("The soup tasted revolting.",0),
 ("His performance was a disaster.",0),
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
    def cases(rows):
        return [(tokenizer.encode(t,add_special_tokens=False),y) for t,y in rows]
    train,test=cases(TRAIN),cases(HOLDOUT)
    root=args.output.resolve()
    if root.exists():raise RuntimeError('output already exists')
    root.mkdir(parents=True)
    extract(snapshot,root/'fragments',model_id=MODEL_ID,revision=REVISION,
            origin='PRETRAINED',scales=[6,8,16])
    # No full language model is instantiated in the A1 experiment.
    store=FragmentStore(root/'fragments',device=args.device)
    models={}
    for mode in ('pretrained','random','embedding-only'):
        encoder=FrozenPrefix(store,6,mode=mode)
        models[mode]=fit_experiment(encoder,train,test,topic='sentiment',
            training_oracle='authored-train-v1',holdout_oracle='authored-test-v1',
            output=root/mode,steps=64)
        if models[mode]['report']['admitted']:
            expert=FragmentCompetence(encoder,root/mode)
            predictions=[expert.predict('sentiment',x,priority=True) for x,_ in test]
            if predictions!=models[mode]['report']['holdout']['predictions'] or expert.predict('unrelated',test[0][0],priority=True) is not None:
                raise RuntimeError('admitted replay or local abstention mismatch')
        del encoder
        if 'expert' in locals():del expert
        gc.collect()
    report={'status':'COMPLETED','source':store.manifest['source'],'device':str(store.device),
            'span':[0,6],'training_texts':TRAIN,'holdout_texts':HOLDOUT,
            'training_tokens':train,'holdout_tokens':test,'models':models,
            'elapsed_seconds':perf_counter()-start,
            'scope':'small authored sentiment probe; separate controls, no teacher, no autonomous assimilation'}
    _write(root/'report.json',report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
