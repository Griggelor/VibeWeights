"""Run the social-head experiment on three fixed seeds, with persistent restart."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from vibeweights.social_experiment import run_seed
from vibeweights.qwen3_fragments import _write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    results=[run_seed(args.output/f'seed-{seed}.sqlite',seed) for seed in (7,19,41)]
    report={'status':'COMPLETED','contract':'social-episodes-v1','torch':str(torch.__version__),
            'device':'cpu','seeds':results,'elapsed_seconds':perf_counter()-start,
            'scope':'synthetic additive numeric tasks, no pretrained fragment cooperation or general semantic ABI'}
    _write(args.output/'report.json',report)
    summary={str(r['seed']):{k:{j:v[j] for j in ('cases','passed','core_calls','invitations')}
                            for k,v in r['holdouts'].items()} for r in results}
    print(json.dumps({'status':'COMPLETED','summary':summary,'elapsed_seconds':report['elapsed_seconds']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
