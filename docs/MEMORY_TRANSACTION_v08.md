# v0.8.0.dev15 – Quellengebundenes REMEMBER mit atomarem Ausführungsreceipt

## Ziel und Abgrenzung

Bisher verweigerte `Population.run` REMEMBER auf derselben KnowledgeLedger-Instanz, wenn `KnowledgeLearning` gebunden war. Ein öffentlicher `store` innerhalb der Source-Guard-Transaktion konnte außerdem durch einen fehlgeschlagenen verschachtelten BEGIN den äußeren Schreib-Lock zurückrollen. Ein verifizierter Lauf soll eine Erkenntnis dauerhaft speichern können, ohne die aktuelle Quellenprüfung auszuschalten. Der neue Vertrag vereinheitlicht diesen *konkreten* Memory-Schreibpfad mit dem bereits vorhandenen Source-/Learning-Guard. Er vereinheitlicht **nicht** alle Transaktionen über Katalog, Gewichtedateien und Knowledge-DB hinweg.

## Vertrag

- Opt-in: `population.run(..., knowledge=learning.knowledge, remember_experts=(...), memory_run_id='unique-run')` unter einem gebundenen KnowledgeLearning-Owner. Die bisherige Autorisierung und die explizite REMEMBER-Aktion bleiben erforderlich; eine Berechtigung allein ist keine Speicheranweisung. Das Environment verwendet weiterhin unabhängige Validatoren. Fehler im Lauf verhindern im neuen Pfad Freigabe und Speicherung; fehlgeschlagene Läufe können ein historisches Receipt ohne Erkenntnis besitzen. Nur verifizierte Zielwerte werden gespeichert.
- Der äußere Source-Guard hält `BEGIN IMMEDIATE` über Inferenz, Prüfung, Quellenbindung und Speicherung. `MemoryRunTransaction` verlangt den äußeren Guard und verweigert verschachtelte Aufrufe. Öffentliche `KnowledgeLedger.store` und `link` verweigern verschachtelte Schreiboperationen **vor** einer eigenen Transaktion; kein unbeabsichtigtes Rollback des Owners.
- Verifizierter Record, Head, Source-Index-Trigger und hashgebundenes Run-Receipt werden in *einer* Source-SQLite-Transaktion gespeichert. Fehler vor Commit rollen alle diese Änderungen zurück, Fehler werden nicht als erfolgreiche Speicherung maskiert. Die separate EventLedger-Instanz ist für diesen Pfad nicht zugelassen, um ungesicherte Dual-Writes zu vermeiden. Read-only- und Legacy-Pfade bleiben unverändert. SQLite-Katalogänderungen (z. B. bereits notwendige Sicherheitsquarantäne) sind weiterhin separat committed.
- Der Aufrufer kann eine stabile Run-ID vorgeben (max. 100 Zeichen, Identifier-ABI). Ohne ID wird eine UUID erzeugt. Eine bereits persistierte ID wird nicht erneut ausgeführt. `learning.memory_runs.receipt(id)` liefert ein historisches, integritätsgeprüftes JSON-Receipt, ausdrücklich keine frische Antwort. Nach einem Abbruch nach Commit kann der Host mit der vorgegebenen ID den Erfolg nachlesen; ein Abbruch vor Commit hinterlässt keine neue Erkenntnis/Receipt und erlaubt erneute Ausführung. Keine automatische Retry-Schleife.
- Konservative Provenienz: alle tatsächlich vorgeschlagenen Ausgaben, alle im Memory-Trace geladenen Record-IDs (einschließlich später evicteter Records), deklarierte Manifest-Eltern und transitive Quellenrefs der beteiligten Lernexperten. Die Kernhashes werden im Receipt persistiert. Der Speicherpfad fordert aktuelle verifizierte Records und gültige Expertenzustände; direkt beitragende Experten müssen resident sein. Parent- und Graph-Budgets: maximal 64 verschiedene Record-IDs und 512 Experten. Der Pfad verweigert Scope-Ausweitung auch bei privaten/Gruppensignalen im Lauf. Diese Überapproximation kann zulässige Speichervorgänge ablehnen; keine feingranulare kausale Attribution behauptet.
- Eine Revision der deklarierten Ursprungsquelle macht den gespeicherten abgeleiteten Record für `current_verified`/Live-Memory-READ ungültig und der Index sperrt betroffene Gewichtsartefakte. Das historische Receipt bleibt erhalten. Der Record ist nicht automatisch ein neuer Trainingsauftrag; neue Gewichte brauchen weiterhin expliziten Intake und separate unabhängige Freigabe.

```python
result = population.run(inputs, goal, validators,
                        knowledge=learning.knowledge,
                        remember_experts=('eligible-expert',),
                        memory_run_id='run-2026-001')
receipt = learning.memory_runs.receipt('run-2026-001')
# receipt is historical; current_verified checks live record validity
```

## Tests und Zwischenresultate

Sieben neue Testfälle: reale DIRECT-Gewichte und ein zweiter affiner Rechenkern mit explizitem test-only REMEMBER-Adapter, unabhängige Ergebnisprüfung, Speicherung und transitive Quellenbindung, Neustart und Duplicate-ID-Abweisung; Fehler nach dem Receipt-Insert, Rollback des Records und Indexereignisses; Scope-Schutz und fehlgeschlagener Validator; verschachtelte öffentliche Writes und konkurrierende SQLite-Verbindung; Ablehnung externer EventLedger und verschachtelter Guard; zwei harte Prozessabbrüche direkt vor Commit und direkt nach Commit. Der Testadapter fordert REMEMBER per Python-Logik an; kein neuer trainierter neuronaler Memory-Kopf.

Erster gezielter Lauf: 30/30 in 10,21 s (7 neue + 23 Memory-/Index-Regressionen). Rohlog: `evidence/v08-memory-transaction-intermediate-1.log`.

## Grenzen

Ein vertrauenswürdiger Einzelowner, keine Sandbox für Host-Callbacks und keine allgemeine Mehr-Owner-Katalogausführung. Keine formale Power-Loss-Dauerhaftigkeit oder atomare Katalog-/SafeTensors-Dateiänderung. Das Receipt belegt Kernhashes; eine spätere reine manuelle Expertenquarantäne ohne Quellenrevision entwertet nicht automatisch einen unabhängig verifizierten Memory-Record. Nicht deklarierte Quellen von frei gelieferten Eingangssignalen können nicht rekonstruiert werden. Keine automatische semantische Faktenzuordnung, keine automatische Übergabe beliebiger Run-Records in typisierte Learning-Jobs. `search` liefert weiterhin ggf. historische live-ungültige Ableitungs-IDs, die `current_verified` ablehnt. Keine realen LLM-, Fragment-, Windows-, GPU- oder Remote-CI-Tests. Weiterhin P0: vollständige Cross-DB-Recovery, gemeinsame Job-Identitäten, Journal-Kompaktierung, TRAINED-Gruppenkompetenz und rekursive Rekrutierung.

Gesamttest: **230/230** (223 bisherige + 7 neue) in 78,18 s. [Summary](../evidence/v08-memory-transaction-summary.json), [Rohlog](../evidence/v08-memory-transaction-tests.log), [Entwicklungsjournal](DEVELOPMENT_LOG.md).
