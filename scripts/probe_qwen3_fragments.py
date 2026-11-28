"""Opt-in local or explicitly downloaded, revision-pinned Qwen3 A0 experiment."""
import argparse
import json
import time
from pathlib import Path
from vibeweights.qwen3_fragments import MODEL_ID, REVISION, extract, FragmentStore, compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--snapshot',type=Path)
    source.add_argument('--download-pinned-qwen3',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--scales',type=int,nargs='+',default=[6,8,16])
    args=parser.parse_args()
    start=time.perf_counter()
    if args.download_pinned_qwen3:
        from huggingface_hub import snapshot_download
        snapshot=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,
                        allow_patterns=['config.json','model.safetensors','LICENSE']))
    else:
        snapshot=args.snapshot
        # The local snapshot must be the pinned source; local-only provenance
        # cannot establish repository identity without an external attestation.
    manifest=extract(snapshot,args.output,model_id=MODEL_ID,revision=REVISION,
                     origin='PRETRAINED',scales=args.scales)
    store=FragmentStore(args.output,device=args.device)
    report=compare(snapshot,store,[[17],[12,34,56,78],[11,22,33,44,55,66,77,88,99,111,123]])
    report['source_acquisition']='pinned_hub_download' if args.download_pinned_qwen3 else 'local_user_attested_snapshot'
    report['elapsed_seconds']=time.perf_counter()-start
    report['fragment_sizes']={s:[sum(b['parameters'] for b in manifest['blocks'][a:z]) for a,z in spans]
                              for s,spans in manifest['partitions'].items()}
    report['unique_artifact_bytes']=manifest['unique_artifact_bytes']
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
