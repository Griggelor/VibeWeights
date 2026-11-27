"""Local terminal chat and loopback API entry point."""
import argparse,json
from pathlib import Path
from .paths import project_root, data_root
from .application import Application
from .local_api import make_server
from .abi import ContractError

def main():
    p=argparse.ArgumentParser(description='VibeWeights local reference application')
    p.add_argument('--root',type=Path,default=None)
    p.add_argument('--memory',default=None); p.add_argument('--journal',default=None)
    p.add_argument('--cooperation',default=None); p.add_argument('--session',default='default'); p.add_argument('--seed',nargs=3,action='append',metavar=('KEY','VALUE','SOURCE'),default=[])
    p.add_argument('--shots',type=int,default=0); p.add_argument('--shot-seed',type=int,default=0)
    mode=p.add_mutually_exclusive_group(); mode.add_argument('--ask'); mode.add_argument('--serve',type=int,metavar='PORT')
    a=p.parse_args()
    a.root=a.root or project_root()
    try:
        with Application(a.root,memory_path=a.memory if a.memory is not None else data_root(a.root)/'knowledge.sqlite',journal_path=a.journal if a.journal is not None else data_root(a.root)/'journal.sqlite',cooperation_path=a.cooperation) as app:
            for k,v,s in a.seed: app.memory.write(int(k),int(v),s)
            if a.serve is not None:
                with make_server(app,a.serve) as server:
                    print(f'VibeWeights API http://127.0.0.1:{server.server_port}',flush=True)
                    try: server.serve_forever()
                    except KeyboardInterrupt: pass
            elif a.ask is not None:
                r=app.ask(a.ask,session=a.session,shots=a.shots,shot_seed=a.shot_seed); print(json.dumps(r,ensure_ascii=False)); return 0 if r['status']=='OK' else 1
            else:
                print('VibeWeights 0.5 – Referenzgrammatik. /exit, /history, /metrics, /feedback positive|negative|correction|preference [Notiz], /revise TEXT')
                last=None
                while True:
                    try: text=input('> ')
                    except (EOFError,KeyboardInterrupt): break
                    if text=='/exit': break
                    if text=='/history': print(json.dumps(app.journal.history(a.session),ensure_ascii=False,indent=2)); continue
                    try:
                        if text=='/metrics': print(json.dumps(app.metrics(session=a.session),ensure_ascii=False,indent=2)); continue
                        if text.startswith('/feedback '):
                            if last is None: raise ContractError('keine Antwort in dieser Sitzung')
                            parts=text.split(' ',2); print(app.feedback(last,parts[1],parts[2] if len(parts)>2 else '',session=a.session)); continue
                        parent=last if text.startswith('/revise ') else None
                        if text.startswith('/revise '):
                            if parent is None: raise ContractError('keine Antwort zur Revision')
                            text=text[8:]
                        response=app.ask(text,session=a.session,shots=a.shots,shot_seed=a.shot_seed,parent_answer_id=parent)
                        last=response['answer_id']; print(response['output'])
                    except ContractError as e: print(f'Ungültige Eingabe: {e}')
    except (ContractError,ValueError) as e: p.error(str(e))
    return 0
if __name__=='__main__': raise SystemExit(main())
