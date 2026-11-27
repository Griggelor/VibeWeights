# v0.5.0 – implementierter Environment-Ausbau

Aufbauend auf v0.4.0, kein Wechsel des Gewichtsbestands. Alle neuen Mechanismen sind SYMBOLIC-Infrastruktur. Das Zielbild steht in [VISION_ARCHITEKTUR_v05](VISION_ARCHITEKTUR_v05.md).

## Produktionspfad und Semantik

`chat` / `local_api` → `Application.ask` → gewichtsgestützte begrenzte Eingabe → `ArithmeticCoalitions.goal` → `CoalitionEngine.run_groups` → lokaler unabhängiger Kopfcheck vor erster Kompetenzberechnung → zustandsabhängige Beiträge → Verifikation → Ergebnisdecoder → `LearningLedger.record` und `SessionJournal` (eine SQLite-Transaktion für beide). Der historische zentrale Router wird ausschließlich von `BaselineApplication` verwendet. Aktuelle Workspace-Semantik: typisierte skalare FP32-Integer mit Validität, keine allgemeinen semantischen Vektoren. Zielräume entstehen aus expliziten Eingabesegmenten; keine autonome Teilzielerzeugung. Runden deterministisch und interleaved, nicht physisch parallel.

### Exploration

`shots` (0–128) ist ein gemeinsames Gesamtbudget über alle expliziten Zielräume. `shot_seed` (0 bis 2^63-1) initialisiert pro Gruppe den reproduzierbaren PRNG. Vor dem ersten Beitrag werden die Work-Heads aller Teilnehmer auf demselben Eingangszustand geprüft. Bei aktiviertem Shot-Budget wird vor der ersten Verarbeitung und nach Fortschrittsrunden jeweils höchstens ein bisher nicht beteiligter, aktuell nicht arbeitsbereiter Teilnehmer zufällig aus sortierten Kandidaten gezogen. Die Runtime hat keinen fachlichen Ranking- oder Top-K-Router. SHOT → JOIN; Erinnerung kann Einladungen auslösen. Bei gültigen Eingaben oder Masken und genügend Restbudget ist ein read-only PROBE (echter Core-Forward) erlaubt; kein `apply`, kein Memory-Schreib-/Leseeffekt, kein State-Commit, kein automatisches Credit. Ein Probe zählt als Expertenschritt und seine Parameterbesuche werden berechnet. Spätere reguläre Arbeit nur über den eigenen Work-Head. Frühere Gruppen können mehr Shot-Budget verbrauchen, Fairness ist offen. Ohne Shots bleiben reguläre Abläufe erhalten, aber die neue globale erste Wahrnehmung kostet gegebenenfalls zusätzliche Kopfprüfungen. Kein adaptiver Stagnations-, Cold- oder Diversity-Shot, keine dezentrale Shot-Anforderung in dieser Version.

### Ledger und Telemetrie

`LearningLedger` nutzt die SQLite-Verbindung des Journals. Jede Antwort, auch eine nicht verifizierte, erhält `answer_id`, optional `parent_answer_id`, Session, Anfrage und Ergebnis. Die UUIDs sind nicht Teil des deterministischen Event-Replays. `expert_observation_v05`: eine Zeile pro beobachteter Expertenidentität (Core- und Head-Hash), Antwort und Gruppe. Kontext ist die Task-ID. Erfasst JOIN, SHOT, PROBE, eingehende INVITE, OFFER, PROPOSE, Verifikationsstatus. Einladung ist Nachfrage, JOIN Teilnahme, PROPOSE tatsächlicher Zustandsbeitrag. `verified_group_contributions` ist nur eine Assoziation eines Beitrags mit einem verifizierten Gruppenergebnis, kein isolierter fachlicher Nachweis. Keine positive Bewertung bloßer Shot-Mitgliedschaft. Alle Feedbackereignisse bleiben erhalten, Kennzahlen verwenden nur die jeweils letzte Bewertung einer Antwort. `positive_answer_associations` / `negative_or_corrected_answer_associations` zählen nur Gruppen mit PROPOSE des jeweiligen Experten. Nicht auf kausale Fehlerzuschreibung schließen; Mehrgruppenfeedback kann mehrere Beiträge betreffen. Gesamtgelegenheiten, individuelle Kosten und kausale Grenznutzenmessung fehlen noch. Der historische Kooperationsspeicher v0.4 verwendet weiter gruppenbezogene verifizierte Erfolge/Misserfolge; neue Feedbackstatistiken beeinflussen Rekrutierung derzeit nicht automatisch.

`feedback_v05` hängt positive, negative, correction oder preference Ereignisse mit optionaler Notiz an eine vorhandene Antwort derselben Session. Je Feedback ein `reflection_v05`-Auftrag PENDING. Es existiert noch kein autonomer Reflexionsworker oder Recherchezugriff. Manuelle Revisionen sind neue Antworten mit Eltern-ID, nicht Überschreibungen. Feedback ersetzt keinen Faktenbeweis. `variant_v05` ist ein unveränderlicher Abstammungskatalog für STAGED-Kandidaten (ID, Familie, Eltern-ID, Digest, Herkunft). Kein automatisches Laden, kein Nachweis des Artefakts anhand des Metadatendigest, keine Freigabe und kein paralleles V1/V2-Produktivexperiment. Dies bleibt nächste Entwicklungsarbeit.

### API-Vertrag

`python -m vibeweights.chat --journal sessions.sqlite --serve 8765` (nur Loopback, lokale Nutzer nicht authentifiziert). JSON POST:

| Pfad | Pflichtfelder | Optional |
|---|---|---|
| `/v1/ask` | `text` | `session`, `budget`, `work_budget`, `recall`, `shots`, `shot_seed`, `parent_answer_id` |
| `/v1/feedback` | `answer_id`, `kind` | `session`, `note` |
| `/v1/metrics` | keine (`{}`) | `session` |
| `/v1/reflections` | keine (`{}`) | `session` |

GET `/health`. `Application.learning.stage_variant` ist interne Python-API, nicht HTTP. Terminal: `/feedback`, `/revise`, `/metrics`, `/history`, `/exit`; `/revise` arbeitet auf der letzten Antwort im laufenden Chat. Ohne Journaldatei nur prozesslokal. Keine Netzwerkfreigabe. Alte v0.3-Referenz-API bleibt kompatibel, neue Endpunkte dort nicht verfügbar.

## Testnachweis und Grenzen

Siehe `evidence/pytest.txt`, `pytest.xml`, `summary.json`. Erstlauf hatte einen API-Kompatibilitätsfehler mit BaselineApplication (eager lookup neuer Methoden), korrigiert durch verzögerte Endpunktauflösung; Fehlerreport erhalten. Neue Tests prüfen Shot-Replay, keine falsche Attribution, gemeinsames Budget, Session-Isolation, Persistenz, Revisionshistorie, Kandidaten-Abstammung, HTTP-Pfad. Keine klassischen Trainingsschritte. Noch keine Windows-/GPU-/Remote-CI-Tests. Registry und Goal-Zahl haben aktuell konfigurations-/implementierungsbedingte Sicherheitsgrenzen; keine architektonisch beabsichtigte kognitive Gruppengrenze. Vollständige freie Topologie, parallele Kandidaten, Evidenzledger, allgemeine Sprache, echtes Lernen und autonome Verbesserung sind offen.
