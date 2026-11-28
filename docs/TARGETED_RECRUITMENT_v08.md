# v0.8.0.dev18 – Kandidat: gezielte rekursive Rekrutierung

**Status: Implementierungskandidat, noch keine lokale Integrations- oder Gesamtabnahme.**
Basis: `668fd486c0990f3ebaba4caba438a74b791f68fa` (dev17).

## Hypothese und Vertrag

Ein Gruppenmitglied, dessen deklarierter numerischer Eingang in einem
fehlerfreien stagnierenden Gruppenlauf fehlt, kann einen gezielten neuen
globalen Impuls veranlassen. Andere Experten entscheiden anhand eigener
trainierter Wahrnehmungsköpfe, ob sie teilnehmen. Keine zentrale Expertenwahl,
keine automatische Aufnahme von Gewichten, keine Veränderung alter Artefakte.

`LearningEvents.publish_recruitment(run, requester)` ist zunächst ein expliziter
Host-Aufruf nach einem ausgeführten GroupWork-Lauf, nicht ein autonomer Aufruf
während des Forward-Passes. Der Bus leitet den Topic aus dem Core-Vertrag ab;
der Host darf ihn nicht frei vorgeben. Nur STAGNATION ohne Faults, belegte
ursprüngliche Gruppenmitgliedschaft, fehlender Eingangs-Topic und aktuelle
Kern-/Kopfbindungen sind zulässig. Keine Eingabewerte oder privaten Signale
werden in den Impuls übernommen; der Topic und technische IDs sind global
sichtbare Metadaten. Die explizite Publikation ist die Autorisierung dieser
Offenlegung. Alte Impulse bleiben historisch; die Live-Gültigkeit hängt an
der Elternkette und den ursprünglichen Teilnehmern. Wiederholte Publikation
derselben Anfrage ist idempotent. Maximale Rekrutierungstiefe: vier.
Es gibt noch kein globales Fan-out- oder Zeitbudget über verschiedene Läufe.

`PerceptionHeads.stage(..., feature_version=2)` erweitert die bisherigen sechs
binären Merkmale um recruit, provides_topic, is_requester. Lokale Heads werden
mit derselben SGD- und separaten Abnahme-Pipeline trainiert. Version 1 bleibt
der Default und lehnt RECRUIT ab. Es gibt keine automatische Kopf-Ersetzung;
neue v2-Köpfe benötigen eigene Freigabe. Provides_topic vergleicht deklarierte
Topics, nicht semantische Ähnlichkeit. Gruppenlabels sind weiterhin Teil des
lokalen Kopfvertrags. Die neue Gruppe arbeitet mit derselben GroupWork-Runtime
und unabhängigen Verifikatoren. Der Host startet einen neuen Run, keine
automatische Wiederaufnahme des alten privaten Zustands.

## Testplan

`tests/test_targeted_recruitment_v08.py` beschreibt zwei Rekrutierungsgenerationen:
consumer braucht mid; middle liefert mid, braucht aber pre; provider liefert pre.
Bei x=4 soll erst der dritte Lauf 95 liefern. Unbeteiligte und v1-Köpfe sollen
nicht beitreten. Neustart, unveränderte Kernhashes, Quarantäne der Elternkette,
Idempotenz, fehlende Mitgliedschaft, erfolgreiche Runs ohne Rekrutierungsrecht,
Tiefe 4 und beschädigte Receipts werden geprüft.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_targeted_recruitment_v08.py
python -m pytest -q
```

Der lokale Git-Klon war in der aktuellen Ausführungsumgebung wegen fehlender
DNS-Auflösung von github.com nicht möglich. Die Quelländerungen wurden über die
GitHub-Schnittstelle in einem separaten Branch vorgenommen. Ein isolierter
CPU-PyTorch-Versuch mit denselben neun binären Eingangsmerkmalen, 256 Trainings-
und 256 disjunkten Holdout-Fällen pro Regel, Nullinitialisierung und 512 SGD-
Schritten lieferte für beide vorgesehenen Regeln je 256/256 korrekte Holdouts.
Das prüft nur die Erlernbarkeit der künstlichen Regeln, nicht den Produktionspfad
oder die tatsächliche Speicherung der Köpfe. Keine behaupteten grünen
Integrationstests, kein neues Release und keine erfundenen Testlogs.

## Grenzen / nächste Schritte

Integrationstests und Gesamtsuite ausführen; erst nach bestätigter Evidenz
Status und Versionsnummer auf dev18 anheben. Danach tatsächliche lokale
Request-Generierung während laufender Gruppen, sichere Wiederaufnahme,
gemeinsame Rekrutierungsbudgets und nichtlineare Kompetenzübertragung.
Keine freie semantische Wahrnehmung, keine autonome Lernjob-Initiierung,
keine reale LLM-/Fragment-/GPU-/Windows-Abnahme.

## Abnahme auf Windows und Linux

[CI 46141672717](https://github.com/Griggelor/VibeWeights/actions/runs/46141672717) am korrigierten Codecommit `2a8d1753c555f5602233eea72097f914c434a15c`: vier grüne Jobs, jeweils 242/242 Tests einschließlich beider neuer Rekrutierungstests, Artefaktreproduktion und aller Smoke-Tests. [Maschinenlesbare Messdaten](../evidence/v08-targeted-recruitment-ci-final.json). Windows 11 / AMD Strix Halo ist das primäre Zielsystem; GitHub Windows CPU CI ersetzt keinen GPU-/UMA-Echtlauf. Host-initiierte Veröffentlichung bleibt Grenze von dev18.
