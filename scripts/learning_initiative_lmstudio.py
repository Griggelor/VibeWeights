"""Opt-in real LM Studio exercise, calendar facts checked by Python calendar.
No automatic model download or promotion. Use a fresh Data root for this demo.
"""
import argparse
import calendar
import json
from pathlib import Path
from vibeweights.abi import ContractError
from vibeweights.bootstrap import BootstrapExpert, BootstrapSpec
from vibeweights.factory import Catalog
from vibeweights.knowledge import Access, KnowledgeLedger
from vibeweights.knowledge_learning import Domain, KnowledgeLearning
from vibeweights.learning_initiative import LearningInitiative, Opportunity
from vibeweights.population import Population, RESIDENT
from vibeweights.task_learning import TaskLearning


def approved():
    ops=[]; oracles={}
    for year in (2023,2024):
        oid=f'calendar-{year}'
        ops.append(Opportunity(oid,Domain(oid,12,4),
            f'Gregorian month lengths for {year}',
            f'Give the number of days in each Gregorian calendar month in {year}. '
            'Entity 0 is January, 1 February, through 11 December. '
            'Label 0 means 28 days, 1 means 29, 2 means 30, 3 means 31. '
            'Return all twelve facts using these IDs, not month names or raw day counts.',
            oid))
        oracles[oid]=lambda i,y=year: calendar.monthrange(y,i+1)[1]-28 if 0<=i<12 else None
    return tuple(ops),oracles


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('propose','stage','promote','inspect','observe','task-propose','task-stage'))
    parser.add_argument('--root',type=Path,required=True,help='Dedicated persistent demo directory')
    parser.add_argument('--cycle',default='calendar-1')
    parser.add_argument('--model',help='Exact loaded LM Studio model ID')
    parser.add_argument('--port',type=int,default=1234)
    parser.add_argument('--topic',choices=('calendar-2023','calendar-2024'))
    parser.add_argument('--entity',type=int,help='Month identity 0..11')
    parser.add_argument('--task',help='Unique task observation ID')
    args=parser.parse_args(argv)
    if args.action in ('propose','stage','task-propose','task-stage') and not args.model: parser.error('--model required for model calls')
    if args.action=='observe' and (args.topic is None or args.entity is None or args.task is None):
        parser.error('observe requires --topic, --entity and --task')
    args.root.mkdir(parents=True,exist_ok=True)
    c=Catalog(args.root/'experts'); p=k=None
    try:
        p=Population(c,tensor_budget=65536)
        k=KnowledgeLedger(args.root/'knowledge.sqlite')
        l=KnowledgeLearning(p,k,Access('host')); i=LearningInitiative(l); ops,oracles=approved()
        if args.action=='observe':
            output=TaskLearning(l).observe(args.task,next(x.domain for x in ops if x.id==args.topic),args.entity)
        elif args.action=='task-propose':
            output=TaskLearning(l).propose(args.cycle,ops,BootstrapExpert(BootstrapSpec('initiative-model',args.model,port=args.port)))
        elif args.action=='task-stage':
            output=TaskLearning(l).stage(args.cycle,ops,BootstrapExpert(BootstrapSpec('teacher-model',args.model,port=args.port)),oracles=oracles)
        elif args.action=='propose':
            output=i.propose(args.cycle,ops,BootstrapExpert(BootstrapSpec('initiative-model',args.model,port=args.port)))
        elif args.action=='stage':
            output=i.stage(args.cycle,ops,BootstrapExpert(BootstrapSpec('teacher-model',args.model,port=args.port)),oracles=oracles)
        else:
            r=i.receipt(args.cycle)
            if r['action']=='SKIP': output=r
            else:
                op=next(x for x in ops if x.id==r['target']); j=l.job(r['job_id'])
                if args.action=='promote':
                    # No model call in this phase; compare against stdlib oracle.
                    if j['state']=='STAGED':
                        if not l.promote(j['id'],oracles[op.oracle_id],oracle_id=op.oracle_id): raise ContractError('independent evaluation rejected')
                    elif j['state'] not in RESIDENT: raise ContractError('candidate not staged or resident')
                    j=l.job(j['id'])
                output={'receipt':r,'job':j}
                if j['state'] in RESIDENT:
                    output['answers_days']=[None if (v:=op.domain.decode(p.compute(j['expert_id'],op.domain.encode(n)))) is None else v+28 for n in range(12)]
        print(json.dumps(output,indent=2,ensure_ascii=False))
        return 0
    except (ContractError,OSError) as exc:
        parser.exit(2,f'Learning exercise refused: {exc}\n')
    finally:
        if p is not None:p.close()
        if k is not None:k.close()
        c.close()


if __name__=='__main__': raise SystemExit(main())
