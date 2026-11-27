"""Local-only JSON service. No external model calls, no cross-origin browser API."""
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from .abi import ContractError

MAX_BODY=4096

def make_server(app,port=8765):
    if type(port) is not int or not 0<=port<=65535: raise ContractError('invalid port')
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup(); self.connection.settimeout(3)
        def log_message(self,*args): pass # avoid logging request data
        def reply(self,status,data):
            body=json.dumps(data,ensure_ascii=False).encode('utf-8'); self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
        def guard(self):
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed or self.headers.get('Origin') is not None:
                self.reply(403,{'error':'loopback clients only; browser origins denied'}); return False
            return True
        def do_GET(self):
            if not self.guard(): return
            if self.path=='/health': self.reply(200,{'status':'OK','backend':'cpu','version':'0.6.0'})
            else: self.reply(404,{'error':'unknown endpoint'})
        def do_POST(self):
            if not self.guard(): return
            if self.path not in ('/v1/ask','/v1/feedback','/v1/metrics','/v1/reflections'): self.reply(404,{'error':'unknown endpoint'}); return
            if self.headers.get_content_type()!='application/json' or self.headers.get('Transfer-Encoding') is not None:
                self.reply(415,{'error':'JSON content length required'}); return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 1<=n<=MAX_BODY: raise ContractError('request size')
                data=json.loads(self.rfile.read(n))
                if not isinstance(data,dict): raise ContractError('request fields')
                contracts={'/v1/ask':({'text'},{'text','session','budget','work_budget','recall','shots','shot_seed','parent_answer_id'},"ask"),'/v1/feedback':({'answer_id','kind'},{'answer_id','kind','note','session'},"feedback"),'/v1/metrics':(set(),{'session'},"metrics"),'/v1/reflections':(set(),{'session'},"reflections")}
                required,allowed,name=contracts[self.path]
                fn=getattr(app,name,None)
                if fn is None: self.reply(404,{"error":"endpoint unavailable in baseline"}); return
                if not required<=data.keys() or not data.keys()<=allowed: raise ContractError('request fields')
                response=fn(**data)
            except (ContractError,ValueError,TypeError,UnicodeError,TimeoutError) as e:
                self.reply(400,{'error':str(e)}); return
            self.reply(200,response)
    return HTTPServer(('127.0.0.1',port),Handler)
