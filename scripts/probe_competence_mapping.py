"""Prespecified small multi-domain Qwen3 localization screen, no routing."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from vibeweights.competence_mapping import scan
from vibeweights.qwen3_fragments import MODEL_ID,REVISION,_hf,_write
from vibeweights.factory import digest

# Each pair changes the underlying fact or task, and reverses the candidate
# answers where possible. Host-authored screening cases, not a benchmark.
CASES=[
 ('arithmetic','addition','2 + 3 =',' 5',' 6'),
 ('arithmetic','addition','2 + 4 =',' 6',' 5'),
 ('arithmetic','subtraction','9 - 4 =',' 5',' 3'),
 ('arithmetic','subtraction','9 - 6 =',' 3',' 5'),
 ('arithmetic','multiplication','3 * 2 =',' 6',' 9'),
 ('arithmetic','multiplication','3 * 3 =',' 9',' 6'),
 ('syntax','agreement','The dogs',' are',' is'),
 ('syntax','agreement','The dog',' is',' are'),
 ('syntax','article','She saw an',' apple',' dog'),
 ('syntax','article','She saw a',' dog',' apple'),
 ('syntax','tense','Yesterday she',' walked',' walks'),
 ('syntax','tense','Every day she',' walks',' walked'),
 ('world-knowledge','capital','The capital of France is',' Paris',' Rome'),
 ('world-knowledge','capital','The capital of Italy is',' Rome',' Paris'),
 ('world-knowledge','color','A clear daytime sky is usually',' blue',' green'),
 ('world-knowledge','color','Fresh grass is usually',' green',' blue'),
 ('world-knowledge','animal','A dog says',' woof',' meow'),
 ('world-knowledge','animal','A cat says',' meow',' woof'),
 ('sentiment','review','The food was delicious. The review was',' positive',' negative'),
 ('sentiment','review','The food was disgusting. The review was',' negative',' positive'),
 ('sentiment','experience','I loved every moment. My experience was',' good',' bad'),
 ('sentiment','experience','I hated every moment. My experience was',' bad',' good'),
 ('sentiment','emotion','She was thrilled by the news. She felt',' happy',' sad'),
 ('sentiment','emotion','She was devastated by the news. She felt',' sad',' happy'),
 ('logic','comparison','7 is greater than 3. This statement is',' true',' false'),
 ('logic','comparison','3 is greater than 7. This statement is',' false',' true'),
 ('logic','category','All cats are mammals. A cat is a',' mammal',' reptile'),
 ('logic','category','All snakes are reptiles. A snake is a',' reptile',' mammal'),
 ('logic','negation','If a light is off, it is not',' on',' off'),
 ('logic','negation','If a light is on, it is not',' off',' on'),
 ('code','python-bool','In Python, not True evaluates to',' False',' True'),
 ('code','python-bool','In Python, not False evaluates to',' True',' False'),
 ('code','python-length','In Python, len([1, 2]) returns',' 2',' 3'),
 ('code','python-length','In Python, len([1, 2, 3]) returns',' 3',' 2'),
 ('code','python-arithmetic','In Python, 10 // 3 returns',' 3',' 4'),
 ('code','python-arithmetic','In Python, 12 // 3 returns',' 4',' 3'),
]


def tokenize(tokenizer):
    cases=[];skipped=[]
    for i,(domain,family,prompt,a,b) in enumerate(CASES):
        row={'id':f'case-{i:02}','domain':domain,'family':family,'prompt':prompt,'answer':a,'distractor':b}
        prefix=tokenizer.encode(prompt,add_special_tokens=False)
        x=tokenizer.encode(prompt+a,add_special_tokens=False)
        y=tokenizer.encode(prompt+b,add_special_tokens=False)
        if x[:-1]!=prefix or y[:-1]!=prefix or not prefix or x[-1]==y[-1]:
            skipped.append({**row,'reason':'not distinct single-token continuations with unchanged prefix'})
            continue
        cases.append({k:row[k] for k in ('id','domain','family')}|{'tokens':prefix,'correct':x[-1],'foil':y[-1]})
    return cases,skipped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-pinned-qwen3',action='store_true',required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    start=perf_counter()
    snapshot=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,
        allow_patterns=['config.json','model.safetensors','LICENSE','tokenizer.json',
                        'tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json']))
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
    cases,skipped=tokenize(tokenizer)
    _,Model,*_=_hf()
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and not torch.cuda.is_available()):
        raise RuntimeError('explicit device unavailable')
    model=Model.from_pretrained(str(snapshot),local_files_only=True,use_safetensors=True,
                trust_remote_code=False,dtype=torch.float32,attn_implementation='eager')
    model=model.to(device).eval().requires_grad_(False)
    report=scan(model,cases,scales=[6,8,16],device=device)
    report.update(status='COMPLETED',model_id=MODEL_ID,revision=REVISION,
                  weights_sha256=digest(snapshot/'model.safetensors'),
                  config_sha256=digest(snapshot/'config.json'),
                  license_sha256=digest(snapshot/'LICENSE'),
                  torch=str(torch.__version__),device=str(device),dtype='float32',
                  planned_cases=CASES,skipped=skipped,elapsed_seconds=perf_counter()-start)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    _write(args.report,report)
    # Full per-case data in uploaded JSON, compact domain rankings in the logs.
    compact={}
    for domain,entry in report['domains'].items():
        eligible=entry['baseline_preferred']
        ranked=sorted([x for x in entry['regions'] if x['eligible']],
                      key=lambda x:(-x['lost_preference'],-x['mean_margin_drop'],x['start'],x['stop']))
        compact[domain]={'cases':entry['cases'],'baseline_preferred':eligible,'top_regions':ranked[:5]}
    print(json.dumps({'status':'COMPLETED','source_revision':REVISION,
        'scored_cases':len(cases),'skipped_cases':len(skipped),'regions':len(report['regions']),
        'elapsed_seconds':report['elapsed_seconds'],'domains':compact},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
