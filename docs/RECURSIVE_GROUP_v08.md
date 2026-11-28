# v0.8.0.dev17 – Wiederbeteiligung gelernter Experten

## Hypothese und Ziel

Eine unabhängig freigegebene Gruppenkompetenz ist nicht nur ein neues Gewichtsartefakt: Sie soll einen eigenen Wahrnehmungskopf bekommen, globale Impulse wahrnehmen, freiwillig kooperieren und zu weiteren Lernaufträgen beitragen. Das ist eine erste Rückkopplung des Lernkreislaufs, **keine autonome Rekrutierung** oder allgemeine Intelligenz.

## Technischer Vertrag

`LearningEvents.publish_group_knowledge(job_id)` erweitert den vorhandenen KNOWLEDGE-Eventpfad (kein zweiter Bus). Ausschließlich residente, separat freigegebene GroupLearning-Artefakte können neu angekündigt werden. Der Payload bindet Job-ID, Experten-ID, Kernhash, Spec-Hash, Freigabereceipt-Hash, Familie und Signal-Topics. Der bestehende Payload-Hash macht wiederholte Veröffentlichung idempotent. Die Domain enthält hier nur den Topic für die Metadatenwahrnehmung, keine behauptete assoziative Domain-Spezifikation. Der Live-Guard prüft Status und deklarierte Eltern; die Ankündigung prüft außerdem Job- und Reportbindung. Ein archiviertes Artefakt kann als Erkenntnis weiterhin angekündigt bleiben, aber ohne gesonderte Reaktivierung nicht arbeiten. Quarantäne oder abweichende Reportbindung sperren die Ankündigung. Kein automatisch freigegebener Kopf, kein automatisches Training durch ein Event.

Der bestehende `PerceptionHeads`-Pfad wird unverändert für neu gelernte Kerne verwendet: trainieren, unabhängige disjunkte Abnahme, Bindung an den Kernhash, danach lokale Vorschläge im Event-Tick. Gruppenausführung und DIRECT/TRAINED-Konsolidierung nutzen dieselbe Runtime wie dev12–16. Die API benötigt autorisierte Host-Aufrufe, vorgegebene binäre Features, Gruppenschlüssel, Task-Inputs, Orakel und Job-IDs. Keine automatische fachliche Entscheidung über benötigte Partner oder autonome rekursive Einladung durch eine laufende Gruppe.

## Experiment

1. Zwei Ursprungsexperten liefern 4 → 29 und 7 → 47. `student` lernt `6x+5` per SGD; separate Freigabe.
2. Neuer Kopf `student-head` mit 32 Trainings- und 32 disjunkten Abnahmeinputs (künstliche binäre Featurekombinationen; Beteiligung bei KNOWLEDGE). Ein weiterer residenter Experte `tail` berechnet `4y-1`, hat einen eigenen trainierten Kopf. Beide Köpfe schlagen unabhängig die Gruppe `next` vor, während die Ursprungsexperten weiter `research` vorschlagen.
3. Ein globaler KNOWLEDGE-Impuls aus `first-gen` wird veröffentlicht. Die zweite Gruppe bearbeitet zwei neue Eingaben: 2 → 17 → 67 und 5 → 35 → 139. Nur `student` und `tail` sind im Trace beteiligt.
4. `grandstudent` entsteht per SGD ausschließlich aus diesen zwei verifizierten Ein-/Ausgabepaaren. Nullinitialisierung, 256 Schritte, Verlust 5952,5 → 5,048709793414476e-29 vor FP32-Export; drei separate Freigabeinputs (-3, 0, 11) mit 3/3 korrekten Werten. Neue Funktion `24x+19`.
5. Auch `grandstudent` erhält einen eigenen Kopf und kann nach Neustart den neuen Wissensimpuls `second-gen` in Gruppe `third` beantworten. Beide neuen Köpfe: Verlust 0,6931471825 → 0,0204040818 (512 SGD-Schritte), je 32/32 disjunkte Abnahmefälle. 36/36 zusätzliche Recheninputs nach Neustart korrekt (Trainings- und Freigabeinputs ausgeschlossen). Alle zuvor vorhandenen Artefakthashes unverändert.

Die Demonstration ist eine zweite Generation mit freiwilliger Beteiligung aus trainierten Köpfen, kein selbst initiierter Lernauftrag. Die dritte Gruppe demonstriert Teilnahme, nicht eine dritte abgeschlossene Kompetenzübertragung. Die Modellklasse und unabhängigen Orakel bleiben künstlich vorgegeben. Kopfgewichte werden bisher im Katalog gespeichert, nicht als physisch VRAM-residente universelle Köpfe gemessen. Keine freie semantische Wahrnehmung, kein nichtlinearer Kompetenztransfer, kein Ressourcenvergleich mit budgetgleicher trainierter Baseline, keine reale Fragment-/LLM-/GPU-/Windows-Abnahme. Die allgemeine Cross-DB-Recovery bleibt offen.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_recursive_group_v08.py tests/test_learning_events_v08.py
python scripts/recursive_group_demo.py
python -m pytest -q
```

[Reproduzierbare Rohdaten](../evidence/v08-recursive-group-demo.json), [gezielter Testlauf](../evidence/v08-recursive-group-intermediate-1.log), [Gesamttest](../evidence/v08-recursive-group-summary.json), [Entwicklungsjournal](DEVELOPMENT_LOG.md).
