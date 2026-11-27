import argparse
import json
from pathlib import Path
from .paths import project_root, data_root



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=None)
    p.add_argument('--x', type=int)
    p.add_argument('--threshold', type=int)
    p.add_argument('--y', type=int)
    p.add_argument('--text')
    p.add_argument('--indexed', action='store_true', help='Use demo index (key 42; default value 6)')
    a = p.parse_args()
    a.root = a.root or project_root()
    if a.text is not None and any(v is not None for v in (a.x, a.threshold, a.y)):
        p.error('--text cannot be combined with numeric arguments')
    x, y, threshold = (7 if a.x is None else a.x, 6 if a.y is None else a.y, 30 if a.threshold is None else a.threshold)

    from .application import Application
    with Application(a.root,memory_path=data_root(a.root)/'knowledge.sqlite',journal_path=data_root(a.root)/'journal.sqlite') as app:
        if a.indexed:
            app.memory.write(1,2,'demo distractor'); app.memory.write(42,y,'demo requested source'); app.memory.write(99,31,'demo distractor')
        text=a.text if a.text is not None else (f'Multipliziere {x} mit dem Wert zu Schlüssel 42 und vergleiche mit {threshold}.' if a.indexed else f'Multipliziere {x} mit {y} und vergleiche mit {threshold}.')
        response=app.ask(text,budget=128)
        print(json.dumps(response,ensure_ascii=False,indent=2))
        return 0 if response['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
