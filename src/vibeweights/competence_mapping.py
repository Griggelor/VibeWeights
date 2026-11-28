"""Offline competence localization, not a production routing policy.

Full-model forced-choice screening, residual update profiling and reversible
whole-block bypass. Effects are conditional on this intervention and task.
"""
from collections import defaultdict
from contextlib import contextmanager
import math
import torch
from .abi import ContractError
from .qwen3_fragments import _partition


def regions(n,scales):
    if type(n) is not int or not 1<=n<=64 or type(scales) not in (tuple,list) or len(scales)>8:
        raise ContractError('mapping region budget')
    result={(i,i+1):['block'] for i in range(n)}
    for scale in scales:
        for a,b in _partition(n,scale):
            tag=f'partition-{scale}'
            result.setdefault((a,b),[]).append(tag)
    return [{'start':a,'stop':b,'views':tags} for (a,b),tags in sorted(result.items())]


def _hidden(output):
    if isinstance(output,torch.Tensor):return output
    if isinstance(output,tuple) and output and isinstance(output[0],torch.Tensor):return output[0]
    raise ContractError('unsupported decoder output')


def _input(args,kwargs):
    x=kwargs.get('hidden_states',args[0] if args else None)
    if not isinstance(x,torch.Tensor) or x.ndim!=3:raise ContractError('decoder hidden input')
    return x


@contextmanager
def observe(layers,metrics):
    handles=[]
    def hook(i):
        def fn(module,args,kwargs,output):
            x=_input(args,kwargs).float()
            y=_hidden(output).float()
            if x.shape!=y.shape or not torch.isfinite(y).all():raise ContractError('decoder output boundary')
            delta=y-x
            rms=lambda z:float(z.square().mean().sqrt())
            inp=rms(x);change=rms(delta)
            cos=float(torch.nn.functional.cosine_similarity(x[0,-1],y[0,-1],dim=0,eps=1e-12))
            metrics[i]={'input_rms':inp,'update_rms':change,'relative_update_rms':change/max(inp,1e-12),
                        'last_token_cosine':cos}
        return fn
    try:
        for i,layer in enumerate(layers):handles.append(layer.register_forward_hook(hook(i),with_kwargs=True))
        yield
    finally:
        for h in handles:h.remove()


@contextmanager
def bypass(layers,start,stop):
    """Replace each selected decoder block output with its input; no weight edits."""
    if type(start) is not int or type(stop) is not int or not 0<=start<stop<=len(layers):
        raise ContractError('mapping bypass span')
    handles=[]
    def hook(module,args,kwargs,output):
        x=_input(args,kwargs)
        if _hidden(output).shape!=x.shape:raise ContractError('decoder bypass boundary')
        return x if isinstance(output,torch.Tensor) else (x,*output[1:])
    try:
        for layer in layers[start:stop]:handles.append(layer.register_forward_hook(hook,with_kwargs=True))
        yield
    finally:
        for h in handles:h.remove()


def _validate(cases,vocab,max_tokens):
    if type(cases) not in (tuple,list) or not 2<=len(cases)<=128:raise ContractError('mapping case budget')
    keys=set()
    for c in cases:
        if set(c)!= {'id','domain','family','tokens','correct','foil'}:raise ContractError('mapping case keys')
        if any(not isinstance(c[k],str) or not c[k] or len(c[k])>128 for k in ('id','domain','family')) or c['id'] in keys:
            raise ContractError('mapping case identity')
        keys.add(c['id'])
        t=c['tokens'];a=c['correct'];b=c['foil']
        if type(t) not in (list,tuple) or not 1<=len(t)<=max_tokens or any(type(i) is not int or not 0<=i<vocab for i in (*t,a,b)) or a==b:
            raise ContractError('mapping token bounds')


@torch.inference_mode()
def scan(model,cases,*,scales,device='cpu',max_tokens=128):
    """Report each case and span, including cases the intact model cannot solve."""
    if type(max_tokens) is not int or not 1<=max_tokens<=128:raise ContractError('mapping sequence budget')
    dev=torch.device(device);layers=list(model.model.layers);n=len(layers)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ContractError('mapping requires frozen evaluation model')
    _validate(cases,model.config.vocab_size,min(max_tokens,model.config.max_position_embeddings))
    spans=regions(n,scales)
    def score(c):
        out=model(input_ids=torch.tensor([c['tokens']],device=dev),use_cache=False).logits[0,-1].float()
        a=float(out[c['correct']]);b=float(out[c['foil']])
        if not math.isfinite(a) or not math.isfinite(b):raise ContractError('mapping nonfinite logits')
        return {'correct_logit':a,'foil_logit':b,'margin':a-b,'preferred':a>b,
                'top_token':int(out.argmax())}
    rows=[]
    for c in cases:
        metrics={}
        with observe(layers,metrics):baseline=score(c)
        if set(metrics)!=set(range(n)):raise ContractError('mapping incomplete layer observation')
        impacts=[]
        for span in spans:
            with bypass(layers,span['start'],span['stop']):ab=score(c)
            impacts.append({'start':span['start'],'stop':span['stop'],'margin':ab['margin'],
                            'preferred':ab['preferred'],'margin_drop':baseline['margin']-ab['margin']})
        # Verify hooks were removed and the unchanged original still computes the same score.
        if score(c)!=baseline:raise ContractError('mapping baseline changed after interventions')
        rows.append({'case':c,'baseline':baseline,'activations':[metrics[i] for i in range(n)],'bypass':impacts})
    return {'contract':'competence-mapping-v1','regions':spans,'cases':rows,'domains':aggregate(rows,spans),
            'scope':'forced-choice screening and conditional bypass effects; not independent fragment competence'}


def aggregate(rows,spans):
    """Separate baseline failures from causal damage to originally correct cases."""
    groups=defaultdict(list)
    for row in rows:groups[row['case']['domain']].append(row)
    result={}
    for domain,items in sorted(groups.items()):
        valid=[x for x in items if x['baseline']['preferred']]
        summaries=[]
        for i,span in enumerate(spans):
            values=[x['bypass'][i] for x in valid]
            summaries.append({**span,'eligible':len(valid),
                'lost_preference':sum(not x['preferred'] for x in values),
                'mean_margin_drop':sum(x['margin_drop'] for x in values)/len(values) if values else None,
                'mean_relative_update_rms':sum(sum(y['relative_update_rms'] for y in x['activations'][span['start']:span['stop']])/(span['stop']-span['start']) for x in items)/len(items)})
        result[domain]={'cases':len(items),'baseline_preferred':len(valid),
                       'families':sorted({x['case']['family'] for x in items}),'regions':summaries}
    return result
