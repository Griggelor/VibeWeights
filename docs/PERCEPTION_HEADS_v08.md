# v0.8.0.dev11 – Trainierbare lokale Wahrnehmungsköpfe

Erster echter neuronaler Baustein für die dezentrale Beteiligung an globalen Lernimpulsen. `PerceptionHeads(LearningEvents(tasks))` bindet einen separat trainierten affinen Mehrfachklassifikationskopf an einen residenten unveränderlichen Expertenkern und dessen Artefakthash. Sechs binäre Metadatenmerkmale: KNOWLEDGE, RELEARN, gleicher Topic (bei Reparatur Ziel-Topic), betroffen, lokaler Reparaturauftrag, Folge-Reparaturauftrag. Der lokale Kopf erzeugt unabhängig einen Logit pro hostdefiniertem Gruppennamen. Positive Logits erzeugen lokale Vorschläge; bei mehreren Zielen begrenzt die lokale Rangfolge die Zahl der Vorschläge gemäß Eventbus-Budget. Kein globaler fachlicher Router, keine Änderung der Kerngewichte, keine implizite Lern-/Admission-Berechtigung.

`stage(head, expert, groups, cases, steps=160)` trainiert Null-initialisierte FP32-Gewichte und Bias mittels SGD (LR 0,5, BCEWithLogits, max. 512 Schritte, max. 8 Gruppen und 256 binäre Beispiele). SafeTensors-Bytes, SHA256, Kernhash, Spezifikation, Lerninput-Hashes und Verluste werden im Katalog auf SSD gespeichert. `admit(head, cases, oracle_id=...)` erfordert disjunkte Eingaben gegenüber dem Training und positive/negative Fälle je Gruppe; alle Entscheidungen müssen stimmen. Unveränderlicher Bericht und separate einmalige Bindung pro Kern, keine automatische Kopf-Ersetzung. Ablehnung lässt den Kandidaten erhalten. `propose` prüft Bindung, Artefakthash, Core-Residency, aktuelles Event und Quellen. `tick` verwendet die vorhandene technische Eventbus-Zustellung, nicht einen neuen Gruppenrouter. Archivierte und quarantänisierte Kerne werden nicht automatisch geladen. Kein zusätzliches reguläres RAM-Tier; derzeit CPU-Referenz und Laden der winzigen Kopfgewichte pro Entscheidung, nicht physische VRAM-Residency.

## Nachweis

Fünf Integrationstests: identisches Ereignis und unterschiedliche trainierte lokale Entscheidungen; ungesehene Featurekombinationen (32 Trainings- und 32 Holdout-Eingaben pro Kopf in den Fixtures); Cross-Topic-Entscheidung; Neustart mit identischen unveränderten Kernhashes; Kandidat ohne Freigabe und abgelehnter Kandidat; disjunkte Eingaben erzwingen; Artefaktkorruption, Archivierung und veraltete Reparaturimpulse; mehrere Reparaturziele. Die Testregeln sind künstliche, vom Host festgelegte Wahrnehmungspräferenzen, keine semantische Kompetenzmessung. Das Modell liefert Beteiligungsentscheidungen aus Gewichten, nicht aus einer fest programmierten Python-Join-Regel. Gruppenschlüssel und numerischer Featureencoder sind weiterhin vorgegeben, keine freie Gruppenentdeckung oder autonome Ausführung. Holdouts prüfen endliche binäre Kombinationen, keine unbekannte natürliche Sprache. Keine adversariale Host-Sandbox; keine GPU-/Windows-/Remote-CI-Abnahme oder tatsächliche Lehrmodell-Inferenz. Katalog-/Quellenaudit weiterhin scanbasiert; gemeinsamer Source-/REMEMBER-Transaktionsvertrag und autonome Gruppenarbeit bleiben P0.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_perception_v08.py
python -m pytest -q
```

Anschluss an Ziel 3: [Multi-Scale Fragment Assimilation](FRAGMENT_ASSIMILATION.md). Die neue Wahrnehmungsschnittstelle ist ein erster Baustein, kein Nachweis der Nutzbarkeit realer Transformer-Fragmente; latente Eingangs-/Ausgangsadapter und Reproduktion der Originalkette stehen aus.
