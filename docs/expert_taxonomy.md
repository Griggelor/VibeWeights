# Kompetenzkatalog — Planung v0.1

Target budgets from handover, not claims of implemented experts. Families form a hierarchical starting point. Allocate individual slots only after competency scope, independent oracle and ABI are established; avoid artificially splitting trivial primitives. Current reference modules are listed in `catalog/manifests`.

| Group | Planned | Families / prerequisites | Current evidence |
|---|---:|---|---|
| Logic | 64 | predicates, propositional, relation, consistency; typed values | GE bounded comparison |
| Mathematics | 64 | numeric primitives, algebra, iteration, approximation; typed numbers | bounded multiplication chain |
| Programming | 64 | parsing, AST, dataflow, transformations, debugging; structured code ABI | none |
| Planning | 64 | goal decomposition, dependency, scheduling, alternatives; task/graph ABI | none |
| Abstraction | 64 | patterns, equivalence, invariants, analogy; relational ABI | none |
| Context/memory | 32 | selection, lookup, source management, compression; memory ABI | exact-key action head |
| Verification | 32 | proof/invariants, counterexamples, cross-checks; result contracts | no expert; test oracles external |
| Causality | 32 | dependencies, interventions, temporal models; graph/state ABI | none |
| Domain knowledge | 64 | typed domain facts, transformations, constraints; source-grounded memory | none |
| Metacognition/routing | 32 | resource choice, progress, uncertainty, halt; action ABI | fixed controller outside pool |
| **Total** | **512** | Long-term target | No 512-count claim |

Dependency path: numerical primitives → predicates → composition → typed relation/graph states → planning & verification → generalization tests → broader semantic interface. Memory cross-cuts all groups. Domain facts do not become neural skills just by putting them in an index. The current math init/finish adapters do not justify separate mature 100M target slots. Decide individual expertise granularity by non-redundancy and replacement tests. No 512-item nominal list before these criteria can be defined.
