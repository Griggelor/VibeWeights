"""Explicit-device population smoke test; no CPU fallback or performance claim."""
import argparse
import json
import platform
import tempfile
from pathlib import Path
from time import perf_counter
import torch
from vibeweights.factory import Catalog
from vibeweights.population import Population, SelectionEvidence, ACTIVE, ARCHIVED, LOW_ACTIVE_IMPORTANT
from vibeweights.environment import Signal


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def probe(device, root):
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):
        raise ValueError('This probe supports CPU and torch.cuda (including HIP) only')
    if device.type=='cuda':
        require(torch.cuda.is_available(),'requested GPU backend unavailable; no CPU fallback')
        device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    info={'platform':platform.platform(),'python':platform.python_version(),
          'torch':str(torch.__version__),'hip':torch.version.hip,'cuda':torch.version.cuda,
          'requested_device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
          'scope':'affine population lifecycle only; payload bytes, not physical VRAM'}
    c=Catalog(root)
    p=None
    checks=[]
    start=perf_counter()
    evidence=SelectionEvidence('probe-evidence',2,True,False,True,True,True,True)
    cases=[((0,),(1,)),((2,),(5,))]
    def resident():
        require(p.resident_ids()==('probe',) and p.tensor_bytes==8,'resident accounting')
        # Diagnostic inspection only; both tensors must actually reside on the requested device.
        e=p._resident['probe']
        require(e.w.device==device and e.b.device==device,'unexpected tensor device')
        info['actual_tensor_device']=str(e.w.device)
    try:
        p=Population(c,tensor_budget=8,device=device)
        m=c.stage_affine('probe','probe',[[2]],[1],source_topic='input',target_scope='global',target_topic='output')
        require(p.admit('probe',cases),'admission')
        resident()
        require(p.compute('probe',(7,))==(15.0,),'device inference mismatch')
        out=p.run([Signal('global','input',(7,))],('global','output'),
                  {('global','output'):lambda s,v:v==(15.0,)})
        require(out['status']=='OK' and out['value']==[15.0],'environment inference mismatch')
        checks.extend(['admission','resident_device','inference','environment'])
        p.select('probe',ARCHIVED,expected_revision=0,evidence=evidence)
        require(not p.resident_ids() and p.tensor_bytes==0,'archive accounting')
        require((Path(root)/(m['sha256']+'.safetensors')).is_file(),'archive artifact lost')
        checks.append('archive')
        p.close(); p=None; c.close(); c=None
        c=Catalog(root); p=Population(c,tensor_budget=8,device=device)
        require(not p.resident_ids(),'archived expert unexpectedly resident after restart')
        p.select('probe',ACTIVE,expected_revision=1,evidence=evidence)
        resident()
        require(p.compute('probe',(-3,))==(-5.0,),'restart inference mismatch')
        require(c.manifest('probe')['sha256']==m['sha256'],'artifact changed')
        checks.extend(['restart','reactivation','artifact_integrity'])
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            info['allocator_allocated_bytes']=torch.cuda.memory_allocated(device)
            info['allocator_reserved_bytes']=torch.cuda.memory_reserved(device)
        info.update(status='PASS',checks=checks,elapsed_seconds=perf_counter()-start)
        return info
    finally:
        if p is not None:p.close()
        if c is not None:c.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True,help='cpu or explicit cuda device; HIP also uses cuda')
    parser.add_argument('--require-windows',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        require(not args.require_windows or platform.system()=='Windows','Windows required')
        with tempfile.TemporaryDirectory(prefix='vibeweights-device-') as tmp:
            report=probe(args.device,Path(tmp))
        code=0
    except Exception as exc:
        report={'status':'FAIL','requested_device':args.device,'platform':platform.platform(),
                'torch':str(torch.__version__),'hip':torch.version.hip,'cuda':torch.version.cuda,
                'error':f'{type(exc).__name__}: {exc}'}
        code=1
    raw=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if args.output:args.output.write_text(raw,encoding='utf-8',newline='\n')
    print(raw,end='')
    return code


if __name__=='__main__':
    raise SystemExit(main())
