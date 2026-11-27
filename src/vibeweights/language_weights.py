"""DIRECT lexical tensor transducers, surrounded by explicitly symbolic grammar.
Not a general encoder/decoder; no optimizer or pretrained model.
"""
from pathlib import Path
import hashlib,json,re
import torch
from safetensors.torch import save_file,load_file
from .abi import ContractError
from .language import ReferenceInput, Request
from .platform import TaskFrame


def generate():
    # 8 right-aligned decimal positions; each one-hot digit contributes digit * 10^position.
    return {'digits.w':torch.tensor([[d*10**p for p in range(7,-1,-1) for d in range(10)]],dtype=torch.float32),
            'digits.b':torch.zeros(1),
            'relation.w':torch.tensor([[-1.],[1.]],dtype=torch.float32),
            'relation.b':torch.tensor([1.,0.],dtype=torch.float32)}

def build(root):
    root=Path(root); directory=root/'artifacts/language'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/'lexical.safetensors'; weights=generate(); save_file(weights,str(path))
    m={'id':'language.lexical','version':'0.1.0','weight_origin':'DIRECT','parameters':sum(t.numel() for t in weights.values()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'limitations':'decimal lexical conversion and binary relation lexeme selection only; symbolic grammar and formatting'}
    (directory/'manifest.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')

class LexicalWeights(torch.nn.Module):
    def __init__(self,weights):
        super().__init__(); shapes={'digits.w':(1,80),'digits.b':(1,),'relation.w':(2,1),'relation.b':(2,)}
        if set(weights)!=set(shapes): raise ContractError('language weights keys')
        for k,shape in shapes.items():
            t=weights[k]
            if t.dtype!=torch.float32 or tuple(t.shape)!=shape or not torch.isfinite(t).all(): raise ContractError('language weights contract')
            self.register_parameter(k.replace('.','_'),torch.nn.Parameter(t.clone(),requires_grad=False))
    @classmethod
    def load(cls,root):
        d=Path(root)/'artifacts/language'; p=d/'lexical.safetensors'; m=json.loads((d/'manifest.json').read_text(encoding='utf-8'))
        if (m['id'],m['version'],m['weight_origin'],m['parameters'],m['sha256'])!=('language.lexical','0.1.0','DIRECT',85,hashlib.sha256(p.read_bytes()).hexdigest()): raise ContractError('language manifest')
        return cls(load_file(str(p)))
    @torch.inference_mode()
    def decimal(self,text):
        if not isinstance(text,str) or not 1<=len(text)<=8 or any(c not in '0123456789' for c in text): raise ContractError('decimal syntax')
        # Reject outside exact FP32 range before conversion using lexicographic lexical bound.
        if text.zfill(8)>'16777215': raise ContractError('decimal outside ABI')
        x=torch.zeros((8,10),dtype=torch.float32)
        for i,c in enumerate(text.zfill(8)): x[i,ord(c)-48]=1
        y=torch.nn.functional.linear(x.flatten(),self.digits_w,self.digits_b).item()
        if not torch.isfinite(torch.tensor(y)) or y!=int(y) or not 0<=y<=16777215: raise ContractError('decimal output')
        return int(y)
    @torch.inference_mode()
    def relation(self,ge):
        if type(ge) is not int or ge not in (0,1): raise ContractError('relation input')
        scores=torch.nn.functional.linear(torch.tensor([float(ge)]),self.relation_w,self.relation_b).tolist()
        if sorted(scores)!=[0.,1.]: raise ContractError('relation logits')
        return ('kleiner als','mindestens')[scores.index(1.)]

class WeightedInput(ReferenceInput):
    origin='DIRECT+SYMBOLIC'
    def __init__(self,weights): self.weights=weights
    def checked_decimal(self,text):
        value=self.weights.decimal(text)
        # Independent inverse lexical check; rejects wrong weights without replacing them.
        if str(value).zfill(len(text))!=text: raise ContractError('language input verification')
        return value
    def encode(self,text):
        if type(text) is not str or len(text)>256: raise ContractError('invalid language input')
        for form,indexed in self._forms:
            match=form.fullmatch(text.strip())
            if match:
                x,operand,threshold=map(self.checked_decimal,match.groups())
                return Request(x,threshold,**({'key':operand} if indexed else {'y':operand}))
        raise ContractError('unsupported language request; no guessed interpretation')
    def encode_task(self,text):
        if type(text) is not str or len(text)>256: raise ContractError('invalid language input')
        match=re.fullmatch(r'Ist ([0-9]{1,3}) mindestens ([0-9]{1,3})\?',text.strip(),re.IGNORECASE)
        if match:
            value,threshold=map(self.checked_decimal,match.groups())
            return TaskFrame('bounded.compare',(('PRODUCT',value),('THRESHOLD',threshold)))
        request=self.encode(text)
        return TaskFrame('bounded.multiply_compare',(('X',request.x),('THRESHOLD',request.threshold),('KEY',request.key) if request.key is not None else ('Y',request.y)))

class WeightedOutput:
    origin='DIRECT+SYMBOLIC'
    def __init__(self,weights): self.weights=weights
    def decode(self,view):
        if view.status!='OK': return f'Kein Ergebnis ({view.status}).'
        relation=self.weights.relation(view.ge)
        if {'kleiner als':0,'mindestens':1}.get(relation)!=view.ge: raise ContractError('language output verification')
        if getattr(view,'task_id','bounded.multiply_compare')=='bounded.compare': return f'{view.product} ist {relation} {view.threshold}.'
        return f'Das Produkt ist {view.product}. Es ist {relation} {view.threshold}.'
