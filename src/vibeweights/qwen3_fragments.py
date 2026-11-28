"""A0: immutable Qwen3 decoder-block extraction and independent replay.

Optional transformers==4.57.1. No remote code, pickle, quantization, padding,
sliding attention, KV cache or semantic adaptation in this narrow reference.
"""
import gc
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .abi import ContractError
from .factory import digest

MODEL_ID='Qwen/Qwen3-0.6B-Base'
REVISION='da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
CONTRACT='qwen3-fragments-a0-v1'
TRANSFORMERS_VERSION='4.57.1'


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def _hf():
    import transformers
    if transformers.__version__!=TRANSFORMERS_VERSION:
        raise ContractError('pinned Transformers version required')
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RotaryEmbedding,Qwen3RMSNorm
    return Qwen3Config,Qwen3ForCausalLM,Qwen3DecoderLayer,Qwen3RotaryEmbedding,Qwen3RMSNorm


def _config(path):
    C,*_=_hf()
    raw=json.loads(path.read_text(encoding='utf-8'))
    if raw.get('model_type')!='qwen3' or raw.get('quantization_config') or raw.get('rope_scaling') or raw.get('use_sliding_window') or not raw.get('tie_word_embeddings'):
        raise ContractError('unsupported Qwen3 configuration')
    c=C.from_dict(raw)
    if (not 1<=c.num_hidden_layers<=64 or not 1<=c.hidden_size<=4096 or
        not 1<=c.intermediate_size<=16384 or not 1<=c.vocab_size<=200000 or
        not 1<=c.num_key_value_heads<=c.num_attention_heads<=64 or
        c.num_attention_heads%c.num_key_value_heads or not 2<=c.head_dim<=256 or c.head_dim%2 or
        any(x!='full_attention' for x in c.layer_types) or c.hidden_act!='silu' or
        c.attention_bias or not 1<=c.max_position_embeddings<=131072):
        raise ContractError('Qwen3 A0 architecture bounds')
    c._attn_implementation='eager'
    return c


def _hex(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ContractError('fragment digest')
    return value


def _partition(n,scale):
    if type(scale) is not int or not 1<=scale<=n:raise ContractError('fragment scale')
    return [[start,min(start+scale,n)] for start in range(0,n,scale)]


def _state_shapes(c,i):
    _,_,Layer,_,_=_hf()
    with torch.device('meta'):
        return {k:tuple(v.shape) for k,v in Layer(c,i).state_dict().items()}


def _artifact(root,tensors):
    if not tensors or any(t.dtype not in (torch.float32,torch.bfloat16,torch.float16) or not torch.isfinite(t).all() for t in tensors.values()):
        raise ContractError('fragment tensor dtype or finite values')
    with tempfile.NamedTemporaryFile(dir=root,suffix='.tmp',delete=False) as handle:
        temp=Path(handle.name)
    try:
        save_file({k:v.contiguous() for k,v in tensors.items()},str(temp))
        sha=digest(temp);dst=root/(sha+'.safetensors')
        if dst.exists():
            if digest(dst)!=sha:raise ContractError('fragment content collision')
        else:temp.replace(dst)
        return {'sha256':sha,'file_bytes':dst.stat().st_size,
                'parameters':sum(t.numel() for t in tensors.values()),
                'payload_bytes':sum(t.numel()*t.element_size() for t in tensors.values()),
                'tensors':{k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in sorted(tensors.items())}}
    finally:temp.unlink(missing_ok=True)


def extract(snapshot,output,*,model_id,revision,origin,scales,byte_budget=2*1024**3):
    """Export exact source tensors once; multi-scale views share immutable block files."""
    snapshot=Path(snapshot).resolve();output=Path(output).resolve()
    if origin not in ('PRETRAINED','SYNTHETIC') or not isinstance(model_id,str) or not model_id or len(model_id)>256 or not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision):
        raise ContractError('fragment source identity')
    if type(byte_budget) is not int or byte_budget<1:raise ContractError('fragment source budget')
    if output.exists():raise ContractError('fragment output already exists')
    config=snapshot/'config.json';source=snapshot/'model.safetensors';license_file=snapshot/'LICENSE'
    if not all(p.is_file() for p in (config,source,license_file)) or source.stat().st_size>byte_budget or config.stat().st_size>1024**2 or not 1<=license_file.stat().st_size<=1024**2:
        raise ContractError('local single-file SafeTensors/config/license required within budget')
    c=_config(config)
    if type(scales) not in (tuple,list) or not 1<=len(scales)<=8 or len(set(scales))!=len(scales):raise ContractError('fragment scales')
    partitions={str(s):_partition(c.num_hidden_layers,s) for s in scales}
    source_sha=digest(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.fragment-',dir=output.parent))
    try:
        shutil.copyfile(config,stage/'config.json');shutil.copyfile(license_file,stage/'LICENSE')
        blocks=[]
        with safe_open(str(source),framework='pt',device='cpu') as src:
            keys=set(src.keys());consumed=set()
            def get(name,shape):
                if name not in keys:raise ContractError('missing source tensor: '+name)
                t=src.get_tensor(name)
                if tuple(t.shape)!=tuple(shape):raise ContractError('source tensor shape: '+name)
                consumed.add(name);return t
            embed=get('model.embed_tokens.weight',(c.vocab_size,c.hidden_size))
            if 'lm_head.weight' in keys and not torch.equal(get('lm_head.weight',embed.shape),embed):
                raise ContractError('tied output head differs from embedding')
            embedding=_artifact(stage,{'weight':embed});del embed
            norm=_artifact(stage,{'weight':get('model.norm.weight',(c.hidden_size,))})
            for i in range(c.num_hidden_layers):
                tensors={k:get(f'model.layers.{i}.{k}',shape) for k,shape in _state_shapes(c,i).items()}
                blocks.append({'index':i,**_artifact(stage,tensors)})
                del tensors
            if consumed!=keys:raise ContractError('unrecognized source tensors')
        if digest(source)!=source_sha or digest(config)!=digest(stage/'config.json') or digest(license_file)!=digest(stage/'LICENSE'):
            raise ContractError('source changed during extraction')
        artifacts=[embedding,norm,*blocks]
        unique={a['sha256']:a for a in artifacts}
        manifest={'contract':CONTRACT,'source':{'model_id':model_id,'revision':revision,'origin':origin,
                  'weights_sha256':source_sha,'weights_bytes':source.stat().st_size,
                  'config_sha256':digest(stage/'config.json'),'license_sha256':digest(stage/'LICENSE')},
                  'transformers':TRANSFORMERS_VERSION,'layers':c.num_hidden_layers,'hidden_size':c.hidden_size,
                  'tied_head':True,'embedding':embedding,'norm':norm,'blocks':blocks,'partitions':partitions,
                  'logical_parameters':sum(a['parameters'] for a in artifacts),
                  'unique_payload_bytes':sum(a['payload_bytes'] for a in unique.values()),
                  'unique_artifact_bytes':sum(a['file_bytes'] for a in unique.values())}
        _write(stage/'manifest.json',{'manifest':manifest,'sha256':_hash(manifest)})
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():shutil.rmtree(stage)


class FragmentStore:
    """Load individual layer spans from immutable artifacts, without the teacher model."""
    def __init__(self,root,*,device='cpu',max_tokens=128):
        self.root=Path(root).resolve();self.device=torch.device(device);self.max_tokens=max_tokens
        if self.device.type not in ('cpu','cuda') or type(max_tokens) is not int or not 1<=max_tokens<=128:
            raise ContractError('fragment execution bounds')
        if self.device.type=='cuda' and not torch.cuda.is_available():raise ContractError('requested GPU unavailable')
        envelope=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        m=envelope['manifest']
        if _hash(m)!=envelope['sha256'] or m['contract']!=CONTRACT or m['transformers']!=TRANSFORMERS_VERSION:
            raise ContractError('fragment manifest integrity')
        if (digest(self.root/'config.json')!=_hex(m['source']['config_sha256']) or
            digest(self.root/'LICENSE')!=_hex(m['source']['license_sha256'])):
            raise ContractError('fragment config/license integrity')
        c=_config(self.root/'config.json')
        if m['layers']!=c.num_hidden_layers or m['hidden_size']!=c.hidden_size or m['tied_head'] is not True:
            raise ContractError('fragment manifest architecture')
        if [b['index'] for b in m['blocks']]!=list(range(c.num_hidden_layers)):
            raise ContractError('fragment block coverage')
        if not m['partitions'] or any(p!=_partition(c.num_hidden_layers,int(s)) for s,p in m['partitions'].items()):
            raise ContractError('fragment partition coverage')
        self.manifest=m;self.config=c

    def _load(self,entry):
        path=self.root/(_hex(entry['sha256'])+'.safetensors')
        if not path.is_file() or path.stat().st_size!=entry['file_bytes'] or digest(path)!=entry['sha256']:
            raise ContractError('fragment artifact integrity')
        with safe_open(str(path),framework='pt',device='cpu') as f:
            out={k:f.get_tensor(k) for k in f.keys()}
        actual={k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in out.items()}
        if (actual!=entry['tensors'] or sum(t.numel() for t in out.values())!=entry['parameters'] or
            sum(t.numel()*t.element_size() for t in out.values())!=entry['payload_bytes'] or
            any(t.dtype not in (torch.float32,torch.bfloat16,torch.float16) or not torch.isfinite(t).all() for t in out.values())):
            raise ContractError('fragment artifact tensor contract')
        return out

    def load_span(self,start,stop):
        if type(start) is not int or type(stop) is not int or not 0<=start<stop<=self.config.num_hidden_layers:
            raise ContractError('fragment span')
        _,_,Layer,_,_=_hf()
        layers=[]
        for i in range(start,stop):
            with torch.device('meta'):
                layer=Layer(self.config,i)
            try:layer.load_state_dict(self._load(self.manifest['blocks'][i]),strict=True,assign=True)
            except RuntimeError as exc:raise ContractError('fragment layer state mismatch') from exc
            layers.append(layer.to(device=self.device,dtype=torch.float32).eval().requires_grad_(False))
        return DecoderSpan(self.config,layers,start,stop,self.device,self.max_tokens)

    @torch.inference_mode()
    def replay(self,scale,token_cases):
        """Fixed original-order control, not a cognitive routing policy."""
        if type(scale) is not int or str(scale) not in self.manifest['partitions']:raise ContractError('unknown fragment scale')
        if type(token_cases) not in (list,tuple) or not 1<=len(token_cases)<=16:raise ContractError('fragment case budget')
        e=self._load(self.manifest['embedding'])['weight'].to(self.device,dtype=torch.float32)
        if tuple(e.shape)!=(self.config.vocab_size,self.config.hidden_size):raise ContractError('embedding shape')
        hidden=[]
        for tokens in token_cases:
            if type(tokens) not in (list,tuple) or not 1<=len(tokens)<=min(self.max_tokens,self.config.max_position_embeddings) or any(type(t) is not int or not 0<=t<self.config.vocab_size for t in tokens):
                raise ContractError('fragment tokens')
            hidden.append(torch.nn.functional.embedding(torch.tensor([tokens],device=self.device),e))
        for start,stop in self.manifest['partitions'][str(scale)]:
            span=self.load_span(start,stop)
            hidden=[span.forward(h) for h in hidden]
            del span
        _,_,_,_,Norm=_hf()
        norm=Norm(self.config.hidden_size,eps=self.config.rms_norm_eps)
        norm.load_state_dict(self._load(self.manifest['norm']),strict=True)
        norm=norm.to(self.device,dtype=torch.float32).eval().requires_grad_(False)
        return [torch.nn.functional.linear(norm(h),e).cpu() for h in hidden]


class DecoderSpan:
    """One executable frozen block range; latent B×T×H boundary, no token head."""
    def __init__(self,config,layers,start,stop,device,max_tokens):
        self.config=config;self.layers=tuple(layers);self.start=start;self.stop=stop
        self.device=device;self.max_tokens=max_tokens
        _,_,_,Rotary,_=_hf()
        self.rotary=Rotary(config).to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self,hidden):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or not 1<=hidden.shape[0]<=4 or
            not 1<=hidden.shape[1]<=min(self.max_tokens,self.config.max_position_embeddings) or
            hidden.shape[2]!=self.config.hidden_size or hidden.dtype!=torch.float32 or hidden.device!=self.device or
            not torch.isfinite(hidden).all()):
            raise ContractError('fragment latent boundary')
        n=hidden.shape[1];pos=torch.arange(n,device=self.device);ids=pos.unsqueeze(0)
        mask=torch.zeros((1,1,n,n),device=self.device,dtype=hidden.dtype)
        mask.masked_fill_(torch.ones((n,n),device=self.device,dtype=torch.bool).triu(1),torch.finfo(hidden.dtype).min)
        rotary=self.rotary(hidden,ids)
        for layer in self.layers:
            hidden=layer(hidden,attention_mask=mask,position_ids=ids,use_cache=False,
                         cache_position=pos,position_embeddings=rotary)
        if not torch.isfinite(hidden).all():raise ContractError('fragment nonfinite output')
        return hidden


def compare(snapshot,store,token_cases):
    """Independent Hugging Face full model vs extracted fixed-order chains."""
    _,Model,*_=_hf()
    model=Model.from_pretrained(str(snapshot),local_files_only=True,use_safetensors=True,
                                trust_remote_code=False,dtype=torch.float32,attn_implementation='eager')
    model=model.to(store.device).eval().requires_grad_(False)
    cases=[list(x) for x in token_cases]
    # Two deterministic greedy extensions of the final prompt.
    with torch.inference_mode():
        for _ in range(2):
            last=cases[-1]
            logits=model(input_ids=torch.tensor([last],device=store.device),use_cache=False).logits
            cases.append(last+[int(logits[0,-1].argmax().item())])
        expected=[model(input_ids=torch.tensor([x],device=store.device),use_cache=False).logits.cpu() for x in cases]
    del model,logits;gc.collect()
    if store.device.type=='cuda':torch.cuda.empty_cache()
    result={}
    for scale in store.manifest['partitions']:
        actual=store.replay(int(scale),cases)
        rows=[]
        for x,a,b in zip(cases,actual,expected):
            delta=(a-b).abs()
            rows.append({'tokens':len(x),'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),
                         'logits_close':bool(torch.allclose(a,b,atol=1e-5,rtol=1e-5)),
                         'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))})
        result[scale]=rows
    passed=all(row['logits_close'] and row['argmax_equal'] for rows in result.values() for row in rows)
    return {'status':'PASS' if passed else 'FAIL','model':store.manifest['source'],
            'device':str(store.device),'compute_dtype':'float32','source_dtype_preserved':True,
            'torch':str(torch.__version__),'transformers':TRANSFORMERS_VERSION,'atol':1e-5,'rtol':1e-5,
            'token_cases':cases,'partitions':result,'manifest_sha256':_hash(store.manifest),
            'logical_parameters':store.manifest['logical_parameters'],
            'unique_payload_bytes':store.manifest['unique_payload_bytes'],
            'scope':'fixed original-order reproduction only; not assimilation or semantic competence'}
