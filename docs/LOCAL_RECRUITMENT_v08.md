# v0.8.0.dev19 – Kandidat: lokale Rekrutierungsinitiierung

**Nicht abgenommen.** Basis: dev18-Kandidat `a352250fa350a2196af8e79a1ff313c6df7184ae`, inklusive Datenbankkorrektur `0711502f47df2caa66daa6c40486462dc99f7c15`. Separater Branch, damit dev18 unverändert geprüft werden kann.

## Hypothese und Vertrag

Ein Gruppenmitglied kann nach einem sauber stagnierenden Run selbst feststellen, dass sein deklarierter Eingang fehlt, und einen RECRUIT-Impuls vorschlagen. Der Host gibt nur ein opt-in Budget vor, keinen Requester, keinen benötigten Topic und keinen fachlichen Routingpfad. `_Member.recruitment_request(snapshot)` prüft ausschließlich den lokalen Eingangsvertrag im Gruppenscope. Diese erste Initiierung ist **deterministische lokale Metadatenlogik**, nicht durch ein neuronales Request-Modell gelernt. Die Entscheidung der Empfänger zur Teilnahme bleibt die separat trainierte v2-Perception.

`GroupWork.execute(..., recruitment_budget=2)` speichert ein unveränderliches Intent im Gruppen-Receipt. Default 0 erhält das bisherige Verhalten. Nur STAGNATION ohne Faults kann Requests erzeugen. Das Budget zählt Anfragen pro Run, maximal 64. Sind mehr lokale Anfragen vorhanden als erlaubt, wird keine ausgewählt oder publiziert (BUDGET). Bei Tiefe vier: DEPTH_LIMIT. NO_REQUEST bei keinem fehlenden Eingang, NOT_APPLICABLE bei sonstigem Run-Status. READY bezeichnet den autorisierten persistenten Auftrag, nicht notwendigerweise eine schon erfolgte Veröffentlichung.

Die Receipt-Transaktion wird vor dem Source-DB-Batch committed. Der automatische Dispatch validiert anschließend die Herkunft jeder Anfrage mit dem bestehenden dev18-Vertrag. `LearningEvents.publish_recruitments` validiert das gesamte Batch vor der ersten INSERT-Operation und schreibt alle Events in einer Source-DB-Transaktion. Keine globale Wertefreigabe, keine automatische Gewichtszulassung, keine Änderung bestehender Artefakte. Es gibt keine Cross-DB-Atomaritätsbehauptung. Bei unterbrochenem Dispatch bleibt das historische Receipt verfügbar; `GroupWork.dispatch_recruitment(run)` wiederholt die idempotente Publikation ohne erneute Berechnung. Ein bereits erfolgreich committedes Batch wird per Eventhash wiedererkannt. Stale ancestry sperrt erneute Live-Publikation. Der Host startet weiterhin nachfolgende Runs und liefert Validatoren, die Gruppenlabels sind weiterhin vorgegeben. Mehrere gleichzeitig fehlende Eingänge können redundante Anfragen entlang einer Kette erzeugen; keine globale Deduplizierung über unterschiedliche Runs.

## Versuchsplan

`tests/test_local_recruitment_v08.py`:

1. Ein Host startet eine Gruppe mit budget=2, aber ohne Requester-Angabe. Consumer fordert mid; ein weiterer stagnierender Lauf meldet mid/pre lokal; die pre-Anfrage wird über trainierte Köpfe beantwortet. Nach Neustart soll die dritte Kooperation 95 liefern. Kein Host-Aufruf der manuellen Publish-Methode; unveränderte Kernhashes.
2. Default bleibt unverändert. Zwei Anfragen bei Budget eins werden vollständig zurückgewiesen, kein fachliches Ranking. Bereits vorhandener Eingang erzeugt keine Anfrage trotz STAGNATION; ungültiger Budgettyp wird abgewiesen.
3. Fault Injection nach erstem Source-INSERT: kein Teilbatch, persistentes READY-Receipt. Wiederanlauf nach DB-Neustart, idempotenter Dispatch ohne Wiederholung des Gruppenruns.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_targeted_recruitment_v08.py tests/test_local_recruitment_v08.py
python -m pytest -q
```

## Entwicklungsjournal / Evidenz

- dev18 war auf dem korrigierten Commit 0711502 unter Linux/Python 3.13 (242 Tests, 53,92 s) und 3.12 (242 Tests, 63,55 s) einschließlich Weight-Rebuild und aller Smoke-Tests erfolgreich ([CI 46141227674](https://github.com/Griggelor/VibeWeights/actions/runs/46141227674)); Windows beim Branch-Aufbau noch offen.
- dev19: Batchvertrag `ee61d1b0494819b0f8baece992719ba6683b3172`, lokale Initiierung `fc6108332b95d27b4ad6ca07b9c166b06712a6b9`, drei neue Integrationstests `814c5e2b2ade494b2522cc11cdbfa0762637b3a7`. Kein behauptetes Testergebnis. Lokaler Git-Klon weiterhin DNS-blockiert.
- Nächste Schritte: CI-Abnahme beider Branches, dev18 finalisieren, dev19 sauber integrieren, neuronales lokales Request-Head-Experiment, Budget über ganze Kooperationsketten, automatische Wiederaufnahme mit unabhängig geprüfter Aufgabenbindung; nichtlinearer Kompetenztransfer.
