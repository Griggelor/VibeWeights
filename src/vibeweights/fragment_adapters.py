"""A1 experimental frozen Qwen3 prefix + learned output adapter + local head.

Independent finite task gate. No global router, semantic ABI or catalog admission.
The native embedding is frozen; only the output adapter and local head learn.
"""
import json
import tempfile
import shutil
from pathlib import Path
from itertools import product
import torch
from safetensors.torch import save_file, load_file
from .abi import ContractError
from .environment import identifier
from .factory import digest
from .qwen3_fragments import FragmentStore, DecoderSpan, _hf, _hash, _write

CONTRACT='fragment-adapter-a1-v1'
HEAD_FEATURES=('topic_matches','input_available','recruitment','priority')
MODES=('pretrained','random','embedding-only')
POOLINGS=('mean-l2-v1','token-l2-mean-v1','last-l2-v1')
CONDITIONINGS=('none','train-centered-radius-v1')


class FrozenPrefix:
    """Experimental native-token input adapter and frozen fragment encoder."""
    def __init__(self,store,stop,*,mode='pretrained',pooling='mean-l2-v1'):
        if not isinstance(store,FragmentStore) or type(stop) is not int or not 1<=stop<=store.config.num_hidden_layers or mode not in MODES:
            raise ContractError('fragment prefix contract')
        if pooling not in POOLINGS:raise ContractError('fragment pooling contract')
        self.store=store;self.stop=stop;self.mode=mode;self.pooling=pooling
        self.embedding=store._load(store.manifest['embedding'])['weight'].to(store.device,dtype=torch.float32)
        if tuple(self.embedding.shape)!=(store.config.vocab_size,store.config.hidden_size):
            raise ContractError('fragment embedding shape')
        self.span=None
        if mode=='pretrained':self.span=store.load_span(0,stop)
        elif mode=='random':
            _,_,Layer,_,_=_hf()
            layers=[]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(29)
                for i in range(stop):
                    layer=Layer(store.config,i)
                    with torch.no_grad():
                        for m in layer.modules():
                            if isinstance(m,torch.nn.Linear):
                                m.weight.normal_(0,store.config.initializer_range)
                                if m.bias is not None:m.bias.zero_()
                    layers.append(layer.to(store.device,dtype=torch.float32).eval().requires_grad_(False))
            self.span=DecoderSpan(store.config,layers,0,stop,store.device,store.max_tokens)
        self.ref={'contract':'qwen3-native-prefix-v1','store_sha256':_hash(store.manifest),
                  'source_origin':store.manifest['source']['origin'],'start':0,'stop':stop,'mode':mode,
                  'embedding_sha256':store.manifest['embedding']['sha256'],
                  'block_sha256':[b['sha256'] for b in store.manifest['blocks'][:stop]] if mode=='pretrained' else [],
                  'random_seed':29 if mode=='random' else None}
        # Preserve default v1 identity; new representations are explicitly bound.
        if pooling!='mean-l2-v1':self.ref['pooling']=pooling
        self.core_bytes=0 if self.span is None else sum(p.numel()*p.element_size() for layer in self.span.layers for p in layer.parameters())
        self.embedding_bytes=self.embedding.numel()*self.embedding.element_size()

    def tokens(self,tokens):
        if type(tokens) not in (list,tuple) or not 1<=len(tokens)<=min(self.store.max_tokens,self.store.config.max_position_embeddings) or any(type(t) is not int or not 0<=t<self.store.config.vocab_size for t in tokens):
            raise ContractError('adapter token bounds')
        return tuple(tokens)

    @torch.inference_mode()
    def features(self,tokens):
        tokens=self.tokens(tokens)
        h=torch.nn.functional.embedding(torch.tensor([tokens],device=self.store.device),self.embedding)
        if self.span is not None:h=self.span.forward(h)
        return pool_latent(h,self.pooling).cpu()


def pool_latent(h,pooling):
    if (not isinstance(h,torch.Tensor) or h.ndim!=3 or h.shape[0]!=1 or not h.shape[1] or
        not h.shape[2] or h.dtype!=torch.float32 or not torch.isfinite(h).all() or pooling not in POOLINGS):
        raise ContractError('adapter latent pooling contract')
    if pooling=='token-l2-mean-v1':
        pooled=torch.nn.functional.normalize(h,dim=-1,eps=1e-12).mean(dim=1).squeeze(0)
    elif pooling=='last-l2-v1':pooled=h[0,-1]
    else:pooled=h.mean(dim=1).squeeze(0)
    if not torch.isfinite(pooled).all():raise ContractError('adapter nonfinite latent')
    return torch.nn.functional.normalize(pooled,dim=0,eps=1e-12)


def condition_training(x,conditioning):
    """Fit an isotropic affine normalization exclusively from training latents."""
    if (conditioning not in CONDITIONINGS or not isinstance(x,torch.Tensor) or x.ndim!=2 or
        not len(x) or not x.shape[1] or not torch.isfinite(x).all()):
        raise ContractError('adapter conditioning contract')
    x=x.to(dtype=torch.float64,device='cpu')
    center=x.mean(dim=0)
    radius=float((x-center).square().sum(dim=1).mean().sqrt())
    gain=min(1000.,1./max(radius,1e-6)) if conditioning!='none' else 1.
    origin=center if conditioning!='none' else torch.zeros_like(center)
    return (x-origin)*gain,origin,gain,{'contract':conditioning,'training_radius':radius,
        'training_mean_norm':float(center.norm()),'gain':gain,
        'training_latent_sha256':_hash(x.tolist())}


def fold_conditioning(w,b,center,gain):
    # No extra persistent adapter dimensions. Evaluate the exported FP32
    # readout itself, never the more favorable float64 fit.
    w=w.double()*gain;b=b.double()-w@center
    w,b=w.float(),b.float()
    if not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise ContractError('adapter nonfinite folded weights')
    return w,b


def _cases(encoder,cases):
    if type(cases) not in (tuple,list) or not 4<=len(cases)<=128:
        raise ContractError('adapter case count')
    result=[]
    for tokens,label in cases:
        if type(label) is not int or label not in (0,1):raise ContractError('adapter binary labels')
        result.append((encoder.tokens(tokens),label))
    if len({x for x,_ in result})!=len(result) or {y for _,y in result}!={0,1}:
        raise ContractError('adapter distinct balanced-class cases required')
    return result


def _fit(x,y,steps,*,regularization):
    if type(steps) is not int or not 1<=steps<=128:raise ContractError('adapter training budget')
    # Inputs are cached frozen latents, never teacher outputs or holdout labels.
    x=x.to(dtype=torch.float64,device='cpu');y=y.to(dtype=torch.float64,device='cpu')
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    b=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],lr=1,max_iter=1,max_eval=10,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(x@w+b,y)+regularization*w.square().sum()
        loss.backward();return loss
    initial=float(closure().detach())
    for _ in range(steps):opt.step(closure)
    final=float(closure().detach())
    if not torch.isfinite(w).all() or not torch.isfinite(b).all():raise ContractError('adapter nonfinite fit')
    return w.detach().float(),b.detach().float(),{'initial_loss':initial,'final_loss':final,'steps':steps,'optimizer':'LBFGS','l2':regularization}


def _evaluate(x,y,w,b):
    with torch.inference_mode():
        scores=x@w+b;pred=(scores>0).long()
    return {'cases':len(y),'passed':int((pred==y).sum()),'expected':y.tolist(),
            'predictions':pred.tolist(),'scores':scores.tolist()}


def _head_cases(priority):
    x=torch.tensor([[m,a,r,priority] for m,a,r in product((0,1),repeat=3)],dtype=torch.float32)
    return x,(x[:,0]*x[:,1]).long()


def fit_experiment(encoder,training,holdout,*,topic,training_oracle,holdout_oracle,output,steps=64,conditioning='none',usage_evidence=None,evidence_store=None):
    """Fit once, evaluate disjoint labels, atomically persist candidate and receipt.

    A failed control is a result, not a workflow error. Only a fully passed
    pretrained candidate is executable through FragmentCompetence.
    """
    for v in (topic,training_oracle,holdout_oracle):identifier(v)
    if training_oracle==holdout_oracle:raise ContractError('independent adapter oracle required')
    train=_cases(encoder,training);test=_cases(encoder,holdout)
    if {x for x,_ in train}&{x for x,_ in test}:raise ContractError('disjoint adapter inputs required')
    if usage_evidence is not None:
        from .usage_adapters import validate_usage
        source=validate_usage(evidence_store,usage_evidence)
        if source!=train or usage_evidence['oracle_id']!=training_oracle or usage_evidence['training_sha256']!=_hash(train):
            raise ContractError('usage evidence training mismatch')
    elif evidence_store is not None:raise ContractError('unexpected usage evidence store')
    output=Path(output).resolve()
    if output.exists():raise ContractError('adapter output already exists')
    x=torch.stack([encoder.features(v) for v,_ in train]);y=torch.tensor([v for _,v in train])
    normalized,center,gain,conditioning_stats=condition_training(x,conditioning)
    w,b,stats=_fit(normalized,y,steps,regularization=0.001)
    w,b=fold_conditioning(w,b,center,gain)
    stats['conditioning']=conditioning_stats
    # Heads have their own objective and metadata-only disjoint input split.
    hx,hy=_head_cases(0);hw,hb,head_stats=_fit(hx,hy,64,regularization=0.001)
    tx=torch.stack([encoder.features(v) for v,_ in test]);ty=torch.tensor([v for _,v in test])
    htx,hty=_head_cases(1)
    observed=_evaluate(x,y,w,b);held=_evaluate(tx,ty,w,b)
    head_train=_evaluate(hx,hy,hw,hb);head_test=_evaluate(htx,hty,hw,hb)
    passed=all(v['passed']==v['cases'] for v in (observed,held,head_train,head_test))
    report={'training':observed,'holdout':held,'head_training':head_train,'head_holdout':head_test,
            'passed':passed,'admitted':bool(passed and encoder.mode=='pretrained'),
            'training_oracle':training_oracle,'holdout_oracle':holdout_oracle,
            'training_sha256':_hash(train),'holdout_sha256':_hash(test),'adapter_fit':stats,'head_fit':head_stats}
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.adapter-',dir=output.parent))
    try:
        tensors={'w':w,'b':b,'head_w':hw,'head_b':hb}
        save_file(tensors,str(stage/'adapter.safetensors'),metadata={'contract':CONTRACT})
        spec={'contract':CONTRACT,'core':encoder.ref,'topic':topic,'features':list(HEAD_FEATURES),
              'pooling':encoder.pooling,'adapter_sha256':digest(stage/'adapter.safetensors'),
              'adapter_parameters':sum(t.numel() for t in tensors.values()),
              'core_fp32_bytes':encoder.core_bytes,'embedding_fp32_bytes':encoder.embedding_bytes,
              'report':report,'torch':str(torch.__version__)}
        if usage_evidence is not None:
            from .usage_adapters import validate_usage
            if validate_usage(evidence_store,usage_evidence)!=train:raise ContractError('usage evidence changed before publish')
            spec['usage_evidence']=usage_evidence
        _write(stage/'manifest.json',{'manifest':spec,'sha256':_hash(spec)})
        stage.rename(output)
        return spec
    finally:
        if stage.exists():shutil.rmtree(stage)


class FragmentCompetence:
    """Explicit experimental admitted frozen prefix; local head may abstain."""
    def __init__(self,encoder,root,*,evidence_store=None):
        self.encoder=encoder;root=Path(root)
        e=json.loads((root/'manifest.json').read_text(encoding='utf-8'));m=e['manifest']
        if (_hash(m)!=e['sha256'] or m['contract']!=CONTRACT or m['core']!=encoder.ref or
            m['features']!=list(HEAD_FEATURES) or m['pooling']!=encoder.pooling):
            raise ContractError('adapter manifest/core integrity')
        r=m['report']
        if encoder.mode!='pretrained' or r['admitted'] is not True or r['passed'] is not True or any(
            r[k]['passed']!=r[k]['cases'] or not r[k]['cases'] for k in ('training','holdout','head_training','head_holdout')):
            raise ContractError('fragment adapter not admitted')
        if 'usage_evidence' in m:
            from .usage_adapters import validate_usage
            proof=m['usage_evidence']
            train=validate_usage(evidence_store,proof)
            if _hash(train)!=r['training_sha256'] or proof['oracle_id']!=r['training_oracle']:
                raise ContractError('usage evidence adapter mismatch')
        elif evidence_store is not None:raise ContractError('unexpected usage evidence store')
        path=root/'adapter.safetensors'
        if path.stat().st_size>65536 or digest(path)!=m['adapter_sha256']:
            raise ContractError('adapter artifact integrity')
        t=load_file(str(path));width=encoder.store.config.hidden_size
        shapes={'w':(width,),'b':(1,),'head_w':(4,),'head_b':(1,)}
        if set(t)!=set(shapes) or any(tuple(t[k].shape)!=s or t[k].dtype!=torch.float32 or not torch.isfinite(t[k]).all() for k,s in shapes.items()):
            raise ContractError('adapter tensor contract')
        if sum(x.numel() for x in t.values())!=m['adapter_parameters']:raise ContractError('adapter parameter count')
        self.m=m;self.t=t

    @torch.inference_mode()
    def propose(self,topic,*,input_available,recruitment=False,priority=False):
        identifier(topic)
        if any(type(x) is not bool for x in (input_available,recruitment,priority)):
            raise ContractError('adapter head feature contract')
        x=torch.tensor([int(topic==self.m['topic']),int(input_available),int(recruitment),int(priority)],dtype=torch.float32)
        return bool((self.t['head_w']@x+self.t['head_b']).item()>0)

    @torch.inference_mode()
    def predict(self,topic,tokens,*,recruitment=False,priority=False):
        if not self.propose(topic,input_available=tokens is not None,recruitment=recruitment,priority=priority):
            return None
        if tokens is None:raise ContractError('adapter input missing')
        x=self.encoder.features(tokens)
        score=float((self.t['w']@x+self.t['b']).item())
        return int(score>0)
