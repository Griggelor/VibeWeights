# v0.8.0.dev13 – Evidenzgebundene additive Konsolidierung einer Arbeitsgruppe

Historischer DIRECT-Stand dev13. [dev16](TRAINED_GROUP_v08.md) ergänzt einen separaten SGD-basierten TRAINED-Pfad aus Gruppenergebnissen; beide Methoden bleiben verfügbar.

Historischer Stand dev13. [dev14](DEPENDENCY_INDEX_v08.md) ergänzt die indexierte Quellenprüfung und transitive Sicherheitsquarantäne; keine trainierte Konsolidierung.

## Ziel und Grenzen

Die bisher getrennten Schritte aus dev10–12 (globaler Impuls, lokale trainierte Wahrnehmungsköpfe, freiwillige Arbeitsgruppe und unabhängig geprüfte Zwischen-/Zielsignale) werden mit der Expert Factory verbunden. `GroupLearning` akzeptiert zwei bis 64 erfolgreiche historische Ausführungsreceipts derselben aktuellen Gruppe mit verschiedenen Eingaben. Die ausgewählte Gruppe muss genau zwei affine FP32-Experten bilden, deren deklarierte Topics und Dimensionen einen eindeutigen gerichteten Ein-Eingang-/Ein-Ausgang-Pfad bilden. Kein zentraler fachlicher Router, keine automatische Auswahl eines Lernziels; der vertrauenswürdige Host beantragt die Konsolidierung. Keine freie Programminduktion oder semantische Aufgabeninterpretation.

Die Receipts müssen dieselben Event-/Gruppen-/Mitglieds-/Kopf-/Validatorbindungen haben; der Trace muss beide Eltern enthalten. Die gespeicherten Zwischenergebnisse und Zielwerte werden gegen aktuelle residente Kerngewichte erneut geprüft. Event, Kopfbindung und Mitgliedschaft bleiben bis zur Freigabe aktuell; alte, widersprüchliche, defekte oder unvollständige Ausführungen sind keine freigabefähigen Lernbelege. Der Job speichert Run-Hashes, Beobachtungen, geordnete Eltern, Kopf- und Kernhashes sowie ABI-Dimensionen; die vorhandenen vollständigen Ausführungsreceipts bleiben unverändert. Damit sind Kooperationserfahrungen nach Neustart wiederverwendbar. Noch kein neuronales Gedächtnis für allgemeine Kooperationsstrategien, kein automatisches Ranking der Gruppenerfahrung.

## Gewichte und Freigabe

Der Kandidat entsteht analytisch (DIRECT), nicht durch Stichprobenoptimierung: für A(x)=W_A x+b_A, B(y)=W_B y+b_B erhält C die Gewichte W_B W_A und den Bias W_B b_A+b_B. Die vorhandene `Catalog.compose_affine` wird mit einem neuen `group_job`-Manifestbezug genutzt. Die beobachteten Ein-/Ausgaben dienen als Evidenz und Replay, nicht als Quelle einer gelernten Regressionsmatrix. Der Katalog verlangt für solche markierten Kandidaten einen separaten bestanden/Case-Hash/Artefakt-Hash-Receipt; `Population.admit` ohne zugehörigen Report kann sie nicht freigeben. Der Report bindet zusätzlich die Orakel-ID und beobachtete/ungesehene Fälle; mindestens drei, höchstens 256 unabhängige, verschiedene, disjunkte Abnahmeeingaben. Ein fehlgeschlagener unveränderlicher Report bleibt erhalten, ein neuer Versuch benötigt einen neuen Job. Numerische Parität wird mit absoluter/relativer Toleranz 1e-6 geprüft. Für beliebige FP32-Eingaben ist aufgrund Rundung/Überlauf keine exakte Parität bewiesen.

Job vor Build persistiert. Identische explizite Wiederaufnahme rekonstruiert einen unvollständigen Build bzw. findet das bestehende Artefakt; eine abweichende Policy wird abgewiesen. Die Admission ist separat und additiv; keine Veränderung, Ersetzung, Selektion oder Archivierung der Eltern. Nach Freigabe benötigt die Berechnung nur die neuen Gewichte. Die deklarierte Eltern-Lineage bleibt eine konservative Sicherheitsabhängigkeit: Quarantäne eines Elternteils propagiert über den vorhandenen Source-Guard; Archivierung ist dagegen keine Ungültigkeit. Ein Event ist nur Auslöser der Kooperation, kein fachlicher Eingabewert der affinen Berechnung. Es muss zur Freigabe aktuell sein, eine spätere Revision des reinen Auslösers widerruft nicht automatisch die bereits unabhängig bestätigte Rechenkompetenz. Fachliche Abhängigkeiten der Eltern sind davon unberührt. Offline-Inferenz mit group_job bleibt wie bei knowledge_job nur explizit historisch zulässig, wenn kein Source-Owner gebunden ist.

```python
from vibeweights.group_learning import GroupLearning

learning_from_group = GroupLearning(work)  # vorhandene GroupWork-Instanz
candidate = learning_from_group.stage('fuse', 'sum', ['run-4','run-7'], family='combined')
assert candidate['state'] == 'STAGED'
# Separat von den beobachteten Eingaben und nicht aus Kandidatenausgaben erstellt:
assert learning_from_group.promote('fuse', [((-3,),(-13,)),((0,),(5,)),((11,),(71,))],
                                   oracle_id='independent-math')
```

## Reproduzierbares Experiment und Zwischenresultate

Zwei DIRECT-Kerne mit trainierten lokalen Wahrnehmungsköpfen kooperieren nach einem globalen KNOWLEDGE-Impuls. Im Gegensatz zur Ablationsfixture dev12 besitzt diese Demonstration bewusst keinen schon vorhandenen Shortcut-Experten. Beobachtung 4 → 9 → 29 und 7 → 15 → 47, neue Gewichte W=[[6]], b=[5]. Der Kandidat hat zwei statt vier Kernparameter; alle ursprünglichen Kernartefakte unverändert, neues Artefakt mit zuvor nicht vorhandenem Hash. 2/2 Replay und 3/3 disjunkte unabhängige Abnahmefälle; separater Subprozess-Test mit kopierter einzelner SafeTensors-Datei, 21/21 Berechnungen von -10 bis 10 (darunter auch bereits beobachtete Eingaben). Kein Elternmodell, Wissensledger oder Lehrer bei dieser Berechnung. Das ist keine Betriebssystem-Sandbox. Die Population wächst im Demo von 76 auf 84 Byte reiner FP32-Kerngewichte einschließlich des Impulsgebers; Kopfgewichte, Dateien, DB, Cache und temporäre Tensoren sind nicht in diesem Budget enthalten.

Gemessener Demo-Lauf auf lokaler CPU: Build 14,06 ms, separate Abnahme 18,83 ms; jeweils 100/100 korrekte Abfragen. Median zweier Elternberechnungen 494,79 µs, neuer Summenexperte 380,76 µs, ca. 23 % weniger. Beide Pfade unter identischem äußerem Source-Guard, alternierende Reihenfolge. Keine End-to-End-Gruppenlatenz (Kopfinferenz und Scheduler fehlen), kein GPU- oder statistischer Mehrfachlaufnachweis. Messung bei parallel laufender Regression; deshalb nur eine indikative Einzelmessung, keine allgemeine Effizienzbehauptung. Rohreceipts, Job, Report, Gewichts-Hash und Zahlen: [Demo JSON](../evidence/v08-group-learning-demo.json). Zwischenversuche und Entscheidungen: [Entwicklungsjournal](DEVELOPMENT_LOG.md).

```sh
python -m pip install -e ".[test]"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/group_learning_demo.py
python -m pytest -q tests/test_group_learning_v08.py
python -m pytest -q
```

## Grenzen und Folgearbeiten

Nur zwei affine Kerne, feste Topic- und Dimensionsverträge, Host-initiierte Konsolidierung und unabhängig definierte synthetische Evaluationspaare; keine trainierte Destillation, keine allgemeine neue Kompetenz aus beliebigen Gruppen, keine autonome Rekrutierung, keine semantische Auswahl oder automatische Gruppenkonsolidierung. Die neue Rolle ist ein Kernel-Experte ohne automatisch erzeugten Wahrnehmungskopf. Die vorhandenen Lernjob- und Gruppenjob-Tabellen haben noch keinen gemeinsamen globalen ID-Reservierungs- und Transaktionsvertrag; der vertrauenswürdige Host muss Job-/Experten-IDs koordinieren. Die persistente Job-/Artifact-Grenze ist per Ausnahmeinjektion und Neustart getestet, keine neue harte Crash-/Power-Loss-Abnahme. Kein Sicherheitsmodell für absichtlich manipulierenden Python-Host oder direkten SQLite-Zugriff. Niedrigstufige Katalog-APIs sind keine Isolation. Weiter offen: indexierter Abhängigkeitsgraph, gemeinsamer Source-/REMEMBER-Transaktionsvertrag, trainierte Methodenübertragung, autonomes Rekrutieren, universelles latentes Interface und reale Fragmentassimilation. Keine LLM-, Windows-, GPU- oder Remote-CI-Abnahme. Die Forschungsziele in [FRAGMENT_ASSIMILATION.md](FRAGMENT_ASSIMILATION.md) bleiben unverändert.

Gesamttest dev13: **217/217** bestanden, 81,25 s; [Testnachweis](../evidence/v08-group-learning-summary.json).
