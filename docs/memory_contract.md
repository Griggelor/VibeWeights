> Historische Referenz bis 0.2.0. Aktuelle Plattformverträge: [platform_v03.md](platform_v03.md).

# Memory contract v0.1

External ingestion `write(key,value,source)` appends a version row under a transaction. Numeric keys and values use ABI bounds; nonempty source label retained in DB. No delete/update-in-place API. `search(key)` uses indexed exact key, latest version; `read(record_id,key)` validates provenance binding and loads the pinned scalar. The DB may persist across process restarts. The active working memory capacity is one retrieved scalar. `EVICT` clears Y in the active state, preserves the record ID and original row. Re-executing a task with the same key can retrieve again. No access to arbitrary SQL via experts.

| Native action | Implementation status |
|---|---|
| SEARCH, READ, EVICT | Numeric head requests, runtime validates and executes |
| HALT | Numeric head requests after eviction (indexed mode); fixed controller in direct mode |
| READY | Reference internal action, grants compute phase |
| WRITE, UPDATE | External Python ingestion/version append only; no neural requests |
| LINK, COMPRESS, ROUTE, LOOP | Not implemented as native actions; recurrence exists via fixed controller |

One exact known key is supplied as structured input, so this is not semantic retrieval or autonomous task-level relevance selection. Source string remains in persistent DB, source ID in final state. No compression, contradiction handling beyond latest-version policy, snapshot across a whole multi-record query, tamper-proof storage, concurrent orchestration or general memory budgeting. SQLite busy timeout is 2s; overall wall-clock timeout not implemented. Future work: typed action arguments, separate intent and runtime execution, multi-record relevance and retrieval, source vs hypothesis vs verified facts, neural write/update/evict decisions, bounded active-context policy, semantic relevance against independent distractor tests.
