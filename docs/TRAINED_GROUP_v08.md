# v0.8.0.dev16 – TRAINED-Kompetenztransfer aus Kooperation

Historischer TRAINED-Stand dev16. [dev17](RECURSIVE_GROUP_v08.md) zeigt zwei aufeinanderfolgende Kompetenzgenerationen mit separat trainierten Wahrnehmungsköpfen und globalen GroupLearning-Impulsen.

## Hypothese und Vertrag

Eine erfolgreich kooperierende Expertengruppe kann eine neue, eigenständig ausführbare Kompetenz erzeugen, ohne ihre Kerngewichte analytisch zu kopieren. Der erste Slice prüft dies für zwei kompatible affine Kerne (derselbe bereits verifizierte Arbeitsgruppenvertrag wie dev13). DIRECT-Komposition bleibt als unabhängiger Lernweg verfügbar. TRAINED erzeugt ausschließlich einen zusätzlichen Kandidaten; keine Änderung oder Ablösung der Eltern und keine implizite evolutionäre Selektion.

## Ablauf

- `GroupLearning.stage(..., method='TRAINED', steps=256)` benötigt 2–64 erfolgreiche, unterschiedliche, aktuelle Ausführungsreceipts derselben freiwilligen Gruppe. Event, Kopf-/Kernhashes, Mitgliedschaft, Ziel, Validatoren und Replay der tatsächlichen Elternberechnung werden geprüft. Die Trainingsdaten sind ausschließlich Eingabe-/Ausgabepaare der verifizierten Gruppe. Der Optimierer erhält weder Elterntensoren noch Holdout-Zielwerte. Die Replay-Prüfung vor dem Training nutzt dagegen bewusst die Eltern.
- `train_affine` ist ein begrenzter CPU-Optimierer: Float64, nullinitialisierte affine Gewichte, zentrierte und skalierte Eingaben, SGD auf quadratischem Fehler. Lernrate 0,5 geteilt durch mittlere quadratische Norm des augmentierten Designs. Export nach Rücktransformation in das ursprüngliche Koordinatensystem als eigenständige FP32-Matrix und Bias; keine analytische Ermittlung aus Elterngewichten. Volle affine Rangbedingung, höchstens 65.536 Parameter, Betragsgrenze 1e6 für Trainingswerte und 1–1.024 Schritte. Da höchstens 64 Beispiele erlaubt sind, kann die Eingangsbreite höchstens 63 betragen. Nahezu singuläre oder schlecht konditionierte Daten können trotz vollen Rangs zu einer abgelehnten Lösung führen.
- Job-Spezifikation enthält die Methode als `group-trained-v1`, Samples, Steps, Torch-Version, Trainingsorakel-ID, Zielscope, Run-/Kopf-/Kernidentitäten. Build-Statistik enthält Dataset-Hash, Initial-/Endverlust, Lernrate, Optimierer, Datentypen und Schritte. Persistierte Stats vor der Artefaktstufe ermöglichen Wiederanlauf mit identischer Policy und reproduzierter Statistik. Ein nicht trainierbarer Datensatz kann eine Job-ID reservieren; neue Policy/Datensatz brauchen neue IDs. Keine allgemeine Cross-DB-Transaktion oder formelle Bit-Reproduzierbarkeit über Hardware/Versionen.
- Factory akzeptiert TRAINED mit `group_job` nur bei passendem Jobvertrag, Eltern und Build-Receipt. Bestehende Knowledge-TRAINED- und Group-DIRECT-Verträge bleiben erhalten. Separates `promote` verlangt mindestens drei disjunkte Holdouts mit eigener Orakel-ID und prüft zusätzlich sämtliche Trainingsbeispiele am exportierten Kandidaten. Der Host verantwortet echte Unabhängigkeit; unterschiedliche IDs allein sind kein Beweis. Keine Freigabe bei Fehlern. Herkunftskanten bleiben im Abhängigkeitsindex, auch wenn Inferenz ohne Eltern möglich ist.

```python
student = groups.stage('train-job', 'student', ['run-4', 'run-7'],
                       family='combined', method='TRAINED', steps=256)
assert student['state'] == 'STAGED'
assert groups.promote('train-job', [((-3,), (-13,)), ((0,), (5,)), ((11,), (71,))],
                      oracle_id='independent-holdout')
```

## Experiment

`PYTHONPATH=src python scripts/trained_group_demo.py` reproduziert den synthetischen Test. Zwei freiwillig kooperierende DIRECT-Kerne mit trainierten Wahrnehmungsköpfen lösen 4 → 9 → 29 und 7 → 15 → 47. Aus lediglich (4,29) und (7,47) lernt ein neuer affiner Kern `6x+5`. 256 echte SGD-Schritte, initialer Verlust 762,5, final 7,888609052210118e-29 vor FP32-Export. Die Verluste nach 1/4/16/64 Schritten sind 428,90625 / 76,33609771728516 / 0,07659497461198289 / 7,763928838667247e-14. Die Kurve besteht aus separat neu initialisierten Läufen mit identischer Policy und unterschiedlichen Schrittbudgets, nicht Zwischencheckpoint-Dateien eines einzigen Laufs.

Zwei Replay- und drei unabhängige Freigabefälle bestanden; zusätzlich 96/96 neue ganzzahlige Eingaben aus [-50,50] (ohne Training/Freigabe), Nullgewichtskontrolle 0/96. Ein separater Subprozess erhält nur eine Kopie der Gewichtedatei, prüft 41/41 Punkte aus [-20,20] (einschließlich der Trainings- und Freigabeinputs). Vier Elternparameter, zwei neue Parameter, 76 → 84 Byte residente Kerngewichte; Kopf-/Katalog-/Aktivierungsspeicher nicht enthalten. Die Alt-Artefakthashes bleiben unverändert. Demo-Build 50,071 ms, Admission 8,695 ms, einzelne CPU-Messung während Regression, kein belastbarer Performancevergleich gegen DIRECT. [Rohdaten](../evidence/v08-trained-group-demo.json).

Weitere Tests: eigenständiges multivariates affines Training (2 Ein-/2 Ausgänge, vier Samples, drei neue Punkte); Rangdefizit; ein Schritt untertrainiert und abgelehnt; falscher oder überlappender Holdout; direkte Admission verweigert; Wiederanlauf nach injizierter Ausnahme nach Build-Stats vor Artefaktstufe; transitive Quarantäne. [Zwischenlog](../evidence/v08-trained-group-intermediate-1.log).

## Grenzen

Die affine Modellklasse ist vorgegeben; die getesteten Funktionen liegen exakt darin. Das Experiment beweist Generalisierung innerhalb dieser eng begrenzten Funktionsfamilie, keine unbekannte Aufgabenklasse, freie Regelentdeckung, nonlinearer Transfer oder Modellfragment-Assimilation. Die Nullgewichtskontrolle ist kein budgetgleicher trainierter Vergleich. Der multivariate Test betrifft nur den Trainer, nicht die gesamte Arbeitsgruppen-Runtime. Stage/Promotion, Tasks, Signalformate, Orakel und Lernbudget stammen vom Host. Keine autonome Lernzielwahl oder neue Kopfaufnahme. Das Training hält den Source-Guard; kein Training neben beliebiger gleichzeitiger Nutzung. Runtime-Budget zählt weiterhin keine echten Peak-Trainingsressourcen. Gemeinsame Job-ID-Reservierung, vollständige Cross-DB-/Datei-Recovery, semantische Wahrnehmung, rekursive Rekrutierung und reales Transformer-Experiment bleiben offen. Kein LLM-, Windows-, GPU-, Strix-Halo- oder Remote-CI-Nachweis.
