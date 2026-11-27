# v0.8.0.dev4 — Wiederanlauf, Learning Economics und Retention

CPU-Referenz, auf dev3 aufgebaut. Ergänzt den [Lernvertrag](KNOWLEDGE_FIRST_LEARNING.md), [dev3](RELATION_LEARNING_v08.md) und die [Population](POPULATION_v08.md). Keine Environment-Abnahme. Alle drei regulären Zustände ACTIVE/LOW_ACTIVE_IMPORTANT (Ziel-VRAM) und ARCHIVED (SSD) sowie additive Lern- und Konsolidierungsregeln bleiben unverändert.

## Wiederanlaufvertrag

In dev3 konnten verifizierte Lehrer-/Relationsfakten dauerhaft gespeichert sein, während der zugehörige Lernjob noch nicht angelegt war. Die ursprüngliche Funktion verweigerte danach eine erneute Ausführung wegen belegtem Topic. Dev4 legt `learning_intakes_v08` in der Wissens-DB an. Verifizierte Fakten und unveränderlicher Intake-Intent werden in derselben SQLite-Transaktion geschrieben. Der Intent enthält Job-/Experten-ID, exakte Quellenrevisionen und -hashes, Dataset-Hash, Domäne, Methode, Schrittbudget, Eltern und Ableitungsmetadaten. Eine ID kann nicht durch einen unpassenden direkten Job übernommen werden.

`KnowledgeLearning.intakes()` zeigt den Zustand der Intents (PENDING oder tatsächlicher Katalogzustand). `resume_intake(job_id)` prüft aktuelle globale VERIFIED-Quellen und exakten Intent/Job-Vertrag, rekonstruiert bei Bedarf den noch nicht vorhandenen Job und fährt einen INCOMPLETE-Build fort. Nach bereits erfolgter Staging/Admission prüft es Integrität und gibt das vorhandene Manifest zurück; es legt keinen zweiten Experten an. Kein erneuter Lehreraufruf. `resume_build(job_id)` ist auch für unterbrochene direkte Lernaufträge verfügbar. Der Build startet deterministisch neu, nicht am letzten SGD-Schritt; bestehende Build-Statistik muss übereinstimmen, PyTorch-Version muss identisch sein. Eltern müssen beim Neubau resident und gültig sein. Keine Freigabe ohne die unabhängige bestehende `promote`-Prüfung; nach einer bereits committed Admission stellt die Population den residenten Zustand aus dem Katalog wieder her. Eine korrigierte Quelle blockiert Recovery und die bestehenden Live-Guards sperren betroffene Varianten. Keine Löschung oder Ersetzung.

Minimaler Host-Aufruf nach Neustart (gleiche Datenwurzeln, ein vertrauenswürdiger Owner):

```python
for entry in learning.intakes():
    if entry['state'] in ('PENDING', 'INCOMPLETE'):
        learning.resume_intake(entry['job_id'])
# Anschließend explizite unabhängige Promotion mit dem ursprünglichen Orakel.
# Bei einem direkten INCOMPLETE-Job: learning.resume_build(job_id).
```

Das ist ein expliziter Recovery-Vertrag, kein autonomer Job-Scheduler. Kein automatischer Retry fehlerhafter oder widerrufener Jobs. Wiederholung eines bestätigten Build-Intents liefert einen Kandidaten, nicht dessen Freigabe. Bereits vor dev4 verwaiste Faktenbatches ohne Intent werden nicht automatisch rekonstruiert. Keine universelle Cross-DB-Atomarität, kein Schutz gegen manipulierten Host oder Dateisystem und kein getesteter Stromausfall. Prozessabbruchtests verwenden `os._exit` nach festgelegten Commit-Grenzen: Fakten+Intent, Job, Build-Statistik, Artefakt/Katalog, Evaluationsreceipt, Admission. Weitere Tests: Rollback vor Commit, veraltete Quellen, ID-Reservierung, direkter Build und Retention. Kein Löschen eventuell verwaister content-addressed Dateien.

## Learning Economics: reproduzierbarer Vergleich

`PYTHONPATH=$PWD/src python scripts/learning_economics_demo.py`

Synthetische Zwei-Hop-Relation aus dev3. Drei Pfade beantworten identische bekannte und unbekannte Abfragen: indexierte aktuelle Fakten (inklusive Prüfung deklarierter Eltern), zwei native Experten mit expliziter ID-Brücke und Live-Source-Guard, sowie ein zusätzlicher Summenexperte mit Live-Source-Guard. Bei allen Pfaden wird die Ausgabe unabhängig gegen dieselbe Fixture geprüft. Der Ledger-Pfad benutzt direkte SQL-Head-Suche und prüft nur die Abhängigkeiten des angefragten Records; die Live-Gewichtspfade auditieren aktuell alle Lernjobs. Das ist ein End-to-End-Mikrobenchmark dieser Implementierungen, kein fairer isolierter Kernelbenchmark, keine allgemeine Leistungsrangfolge. Zeitwerte sind hardware-/cacheabhängig; die gemischte Median-Latenz enthält auch unbekannte Fälle. Lernzeit enthält Faktenaufnahme, Gewichtsaufbau, Evaluation und Admission, kein echtes Lehrmodell. Summenexpertenerzeugung enthält Ableitung und erneutes Einlernen.

Messwerte: `evidence/v08-learning-economics-demo.json`. Im gemessenen CPU-Lauf, 100 Abfragen je Pfad, alle 300 korrekt: Ledger ca. 3,85 µs Median, Live-Zwei-Hop ca. 743 µs, Live-Summenexperte ca. 665 µs. Die beiden Eltern enthalten 43 Parameter, der Summenexperte 21. Alle bleiben erhalten: insgesamt 64 Parameter. Die Verdichtung reduziert damit die für diese konkrete Relation erforderliche Parameterzahl, nicht die Gesamtgröße der Population. Kein Nachweis, dass parametrisches Speichern gegenüber einem Index bei konkreten Fakten wirtschaftlicher ist. Der globale Audit-Overhead ist ein klarer Optimierungskandidat; Kostenmessung darf nicht durch Abschalten von Sicherheitsprüfungen beschönigt werden.

## Sequenzielle Wissensaufnahme

10 getrennte Domänen, je 5 Fakten, 5 unbekannte Abfragen inkl. Randfällen, abwechselnd DIRECT und TRAINED. Nach jedem Paket werden alle vorherigen Pakete vollständig erneut geprüft. Nach dem zehnten Paket und Neustart: 100/100 Fälle korrekt; ursprüngliche Artefakthashes unverändert. Logisches FP32-Tensorbudget wächst in diesem Versuch auf 1696 Bytes inkl. der drei Relationsexperten. Keine physische GPU-Residency-Messung, kein Beweis für störungsfreies Lernen innerhalb eines gemeinsam trainierten Netzes, keine realistische große Wissensdomäne. SSD-/DB-Dateigrößen stehen im Messbericht.

## Nächste Schritte

- Gemeinsamer Source-/Memory-/Population-Transaktionsvertrag inklusive REMEMBER, vollständige Recovery über DB- und Dateigrenzen, indexierte Abhängigkeiten statt globalem Audit bei jedem Vorwärtslauf.
- Echte LM-Studio-Lehrerabnahme mit unabhängig überprüfbaren Quellen, modellinitiierte Lernaufträge innerhalb technischer Grenzen und neue Methoden-/Transfer-Holdouts.
- Learning Economics bei größeren Domänen mit vergleichbaren Guard-Verträgen; nur nachgewiesenen Nutzen als Konsolidierungsevidenz verwenden. Selektion weiterhin getrennt; keine automatische Auslagerung wichtiger seltener Experten, keine evolutionäre Löschung.
