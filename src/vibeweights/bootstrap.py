"""Explicit local text bridge for pre-trained Dense/MoE models.
No hidden-state integration, autonomous calls, tools or code execution.
Remote teachers are not enabled. Outputs remain unverified proposals.
"""
from dataclasses import dataclass
import json, urllib.request, urllib.error
from .environment import identifier
from .abi import ContractError

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None

@dataclass(frozen=True)
class BootstrapSpec:
    id: str
    model: str
    port: int=1234
    max_tokens: int=512
    timeout: int=60
    def __post_init__(self):
        identifier(self.id)
        if not isinstance(self.model,str) or not 1<=len(self.model)<=256 or type(self.port) is not int or not 1<=self.port<=65535 or type(self.max_tokens) is not int or not 1<=self.max_tokens<=4096 or type(self.timeout) is not int or not 1<=self.timeout<=300: raise ContractError('invalid bootstrap configuration')

class BootstrapExpert:
    def __init__(self,spec): self.spec=spec
    def propose_with_context(self,text,ledger,access,record_ids,*,max_chars=12000):
        """Explicit host-selected numeric records; no automatic search or external send.
        Text serialization is a bridge, not direct latent communication.
        """
        from .knowledge import integer
        if access.expert!=self.spec.id: raise ContractError('bootstrap access identity mismatch')
        if type(text) is not str or not text.strip(): raise ContractError('text input bounds')
        integer(max_chars,1,16000)
        if type(record_ids) is not tuple or len(record_ids)>32 or len(set(record_ids))!=len(record_ids): raise ContractError('context record IDs')
        records=[]
        for rid in record_ids:
            rec=ledger.current_verified(rid,access)
            records.append({'id':rec.id,'version':rec.version,'topic':rec.topic,'values':rec.values,'source':rec.source})
        payload=json.dumps({'request':text,'reference_data':records},ensure_ascii=False,allow_nan=False)
        if len(payload)>max_chars: raise ContractError('bootstrap context character budget')
        result=self.propose(payload)
        return dict(result,context_ids=list(record_ids))

    def propose(self,text):
        if not isinstance(text,str) or not 1<=len(text)<=16000: raise ContractError('text input bounds')
        s=self.spec; body={'model':s.model,'messages':[{'role':'user','content':text}],'max_tokens':s.max_tokens,'stream':False,'temperature':0}
        req=urllib.request.Request(f'http://127.0.0.1:{s.port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
            with opener.open(req,timeout=s.timeout) as resp: raw=resp.read(1024*1024+1)
            if len(raw)>1024*1024: raise ContractError('response too large')
            obj=json.loads(raw); content=obj['choices'][0]['message']['content']
            if not isinstance(content,str) or not content.strip(): raise ContractError('missing response text')
        except (OSError,ValueError,TypeError,KeyError,IndexError) as exc: raise ContractError('bootstrap transport or response failure') from exc
        return {'expert':s.id,'model':s.model,'origin':'PRETRAINED','transport':'LOCAL_TEXT','verified':False,'proposal':content}
