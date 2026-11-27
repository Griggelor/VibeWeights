# Implementierung v0.6.0 – erster Environment-Slice

## Scope

Additiver Umbau, kein vollständiger Ersatz der v0.5-Runtime. `python -m vibeweights.next demo 7` startet den neuen Pfad; bestehender Chat und HTTP-API verwenden weiterhin die arithmetische CoalitionEngine. Der neue Kern muss vor Ablösung der alten Runtime featureparitätisch integriert werden.

- `environment.py`: unveränderliche numerische Vektoren (maximal 4096 Komponenten), Scope/Topic-IDs; Runden mit gemeinsamer Snapshot-Sicht. Vorschläge getrennt nach Instanz. Nur genau ein unabhängig verifizierter Wert wird publiziert. Mehrere verschiedene akzeptierte Werte => CONFLICT; unvollständige Runde => BUDGET. Keine automatische Mehrheitsentscheidung. Globale Sicht auf alle publizierten Facts, keine privaten Zustände oder frei lernenden Gruppen. Verifikatoren sind vertrauenswürdige Host-Funktionen; im Demo ein arithmetisches Orakel. Akzeptierte Facts sind innerhalb eines Runs unveränderlich; noch keine allgemeine Revisionssemantik. Kein allgemeines semantisches Alignment. Optionale Probes haben eigene Aufrufquote und verwerfen Resultate; Fehler werden registriert. Keine Prozessisolation und keine harten Zeit-/Bytes-Grenzen für beliebige Python-Callbacks. Vertraute reine Adapter sind Voraussetzung. Der Scheduler führt Runden aus, nicht die fachliche Auswahl; Relevanz der affine Adapter ist aktuell symbolisch. Schrittgrenzen zählen Forward-Aufrufe, keine FLOPs.
- `factory.py`: persistenter SQLite-Katalog ohne festen Populationsgrenzwert; Familie und Instanz getrennt, mehrere Eltern erlaubt. Stage erzeugt SafeTensors analytisch aus vorgegebenen affinen Koeffizienten (kein Training), hashadressierte Artefakte. Nur ACTIVE standardmäßig ladbar, explizite hostgelieferte Abnahmefälle; keine automatische Promotion. Einzeldatelimit vor Laden, kein Resident-Cache oder globales Peak-RAM-Budget. Demo lädt drei kleine Module. Keine Unterstützung allgemeiner Graphen/GGUF/Q4/LLM-Artefakte in der Factory. Keine Sandbox für beliebigen Code. Atomare SQLite-Schreibvorgänge, aber noch kein vollständiges Crash-Recovery über Datei-/DB-Grenzen.
- `compose_affine`: direkte algebraische Fusion kompatibler affiner Module; neuer STAGED-Kandidat, beide Eltern bleiben unverändert. Kein einfaches Mitteln beliebiger Netzgewichte. FP32-Rundungsunterschiede können im allgemeinen Fall auftreten; daher stets unabhängige Prüfung.
- `bootstrap.py`: explizit aufgerufener Loopback-HTTP-Adapter an `/v1/chat/completions`, konfigurierbare Modell-ID und Port, Token- und Antwortgrößengrenze, Socket-Timeout (kein harter Gesamtdeadline), kein Proxy oder Redirect. Keine Tool-Ausführung. Ausgabe `verified=false`, PRETRAINED/LOCAL_TEXT; nur Transporttest mit lokalem Testserver. Keine latente oder autonome Integration in `Environment`, keine Modelllizenzprüfung oder Modellbereitstellung.
- CLI-Standardeinstellungen speichern Data/knowledge.sqlite und Data/journal.sqlite unter dem ermittelten Root; `VIBEWEIGHTS_ROOT` oder `--root` können ihn überschreiben. Eingebettete alte `Application`-Defaults bleiben flüchtig. Native Distribution noch offen. Decoderfehler im Legacy-Chat erhalten Kernverifikation und Ereignisse; Nutzerantwort bleibt INVALID. Präferenzfeedback überschreibt keine vorherige Bewertung mehr.

## Review-Matrix (R01–R16)

| Befund | Status in v0.6 |
|---|---|
| R01 | Teil: neue Factory parallele Instanzen, >512 Teilnehmer synthetisch, lazy load pro Artefakt. Legacy Registry unverändert; große reale Tensoren und Cache offen. |
| R02 | Teil: unabhängige Hypothesen/Permutation im neuen Slice geprüft; alter Chat noch first-writer-abhängig. |
| R03–04 | Teil: neue Probes separate Aufrufquote/Fehlerfang; Legacy Shots unverändert. Kein isolierter Prozess. |
| R05 | Offen: keine autonome Feedbackreflexion. |
| R06 | Behoben in getesteter Legacy-Metrik: Präferenzen verdrängen Rating nicht. |
| R07 | Teil: Decoderfehler erhalten Kernereignisse/Telemetrie. Vollständiges transaktionales Episoden-Recovery offen. |
| R08 | Teil: CLI standardmäßig persistent; eingebettete Referenzpfade weiterhin in-memory. |
| R09 | Teil: root-Auflösung unabhängig von cwd bei Checkout/editable install, kein natives Portable-Release. |
| R10 | Offen: Legacy Recall-Schreibpolicy. |
| R11 | Teil: freie numerische Vektor-Scope-Verträge und Weitergabe zwischen Scopes, keine autonome Gruppentopologie/privaten rekurrenten Zustände. |
| R12–13 | Offen: sparse Kooperation und natives Wissensledger. |
| R14 | Offen: Q4, GGUF, C++/GPU, Studio, Windows portable. |
| R15 | Teil: neue Factory mit tatsächlichem Artefakthash und expliziter Abnahme. Keine autonome Auswahl, kein kontrollierter Live-Wettbewerb im Chat. |
| R16 | Offen: sichere API, Isolation, Abbruch, Ressourcenlimits in Bytes/Zeit. Textadapter nur Loopback. |

## Testevidenz

87 Tests (72 Legacy + 15 neue Tests inklusive Parametrisierung), tatsächlicher lokaler CPU-Lauf, `evidence/v06-tests.log`, `evidence/v06-tests.xml`, `evidence/summary.json`. Historische v0.5-Review-Charakterisierungstests sind unter docs/review/v0.5.0 archiviert und gehören nicht als Solltests zur neuen Suite. Neu: Variante konkurriert unabhängig von Eingabereihenfolge, Konflikt, Verifierfehler, Probe-Isolation auf Ergebnisebene, Quoten, 601 synthetische Teilnehmer, persistente Factory/Hashprüfung, Affine-Verdichtung mit 201 zusätzlichen Eingaben, tatsächlicher HTTP-Transport mit Testserver, Telemetrie und root-Pfade. Kein Qwen-/Windows-/GPU-/GitHub-Lauf.
