# VibeWeights 0.4.0 – Decentralized Self-Organizing Recurrent Expert Network

Experimentelles KI-Forschungsprojekt. Ziel: ungefähr 512 unabhängig erzeugbare Experten à ca. 100M Parameter, direkte VibeWeights-Gewichtsprogrammierung, numerische Kommunikation, selbstorganisierte Arbeitsgruppen, rekurrente Verarbeitung, selbstverwaltetes indexiertes Gedächtnis und dedizierte Sprachschnittstellen. Das Ziel ist noch nicht erreicht. Diese Version ersetzt den zentralen Aufgabenplan im Standardpfad durch einen dezentralen Event-Referenzkern.

## Start (Windows / Linux, CPU-PyTorch)

```sh
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1` (alternativ `.venv\Scripts\python.exe`). Linux: `source .venv/bin/activate`.

```sh
python -m pip install -e ".[test]"
python -m pytest -q
python -m vibeweights.chat
```

Einzelanfragen und mehrere explizite Teilziele:

```sh
python -m vibeweights.chat --ask "Ist 42 mindestens 30?"
python -m vibeweights.chat --ask "Ist 7 mal 6 mindestens 30?; Ist 8 mal 5 mindestens 41?"
python -m vibeweights.chat --memory memory.sqlite --journal sessions.sqlite --seed 42 6 Quelle --ask "Multipliziere 7 mit dem Wert zu Schlüssel 42 und vergleiche mit 30."
```

`--seed` legt eine neue versionierte Quelle ab. `--journal sessions.sqlite` persistiert Sitzungsprotokoll und standardmäßig auch Kooperationsgedächtnis; alternativ `--cooperation cooperation.sqlite`. Ohne Dateipfad ist der Speicher nur im Prozess persistent. Dieselbe Session und Eingabeform für zwei unterschiedliche Aufgaben verwenden, um Erinnerungs-Einladungen im zweiten Lauf zu sehen. Keine fertigen Antworten im Kooperationsgedächtnis. `/history` zeigt Protokoll, `/exit` beendet. Die Sprache ist begrenzt: feste deutsche Grammatik, gewichtsbasierte numerische Teilfunktionen, keine freie Unterhaltung.

Lokale JSON-API (nur 127.0.0.1): `python -m vibeweights.chat --serve 8765`. `GET /health`; `POST /v1/ask`, JSON `{"text":"Ist 7 mal 6 mindestens 30?","session":"demo","budget":128,"work_budget":100000,"recall":true}`. `recall:false` deaktiviert Einladungen aus Erinnerungen, nicht die breite Discovery oder das Feedback. Keine Browser-Origin-Requests, keine Authentifizierung zwischen lokalen Nutzern; nicht im Netz exponieren. Python: `vibeweights.application.Application`; `BaselineApplication` nutzt ausdrücklich v0.3 zum Vergleich. Kompatibilitäts-CLI `python -m vibeweights --text "Ist 7 mal 6 mindestens 30?"` führt ebenfalls den neuen Pfad aus. JSON-Ausgabe enthält Ereignisse und numerische Experten-ID-Tabelle; bei mehreren Teilzielen auch `groups`.

## Reproduzieren

```sh
python scripts/build_weights.py
python scripts/build_language_weights.py
python scripts/build_coalition_weights.py
python -m pytest -q
python scripts/benchmark_coalitions.py
```

Keine Trainingsschritte. Fünf kleine Kompetenzmodule (43 Parameter), zehn Gate-Köpfe in fünf Artefakten sowie Sprach-Teilgewichte (85 Parameter). Exakte Zahlen und Testnachweis in `evidence/summary.json`. Symbolische Featureextraktion, Einladungslogik, Action-Adapter, unabhängige Verifikation und Grammatik sind ausdrücklich keine direkt programmierten neuronalen Kompetenzen.

## Dokumentation
- `docs/ARCHITEKTUR_v04.md`: beschlossenes Zielbild (historische Momentaufnahme vor Implementierung).
- `docs/platform_v04.md`: tatsächliche Implementierung, Grenzen, Ownership, Tests.
- `docs/FORSCHUNG_2026-11-26.md`: Forschungsstand aus vorheriger Recherche, nicht neu repliziert.
- `docs/status.md`: aktueller Stand und Reihenfolge der Weiterentwicklung.
- `docs/github_import.md`: Import und CI. Remote-CI, Windows und Strix Halo noch nicht verifiziert.
- `evidence/`: aktuelle lokale Test- und Benchmarkdaten; `baseline_v03/` historische Evidenz.

Die 512-Teilnehmer-Messung enthält inaktive synthetische Platzhalter, keine 512 funktionalen Experten. Es gibt keine autonome Teilzielbildung, keine echte Parallelisierung, keinen neuronalen Konsens und keine Selbstverbesserung in dieser Version. Mehrgruppenarbeit ist interleaved; Gruppenmitgliedschaften sind dynamisch, Zielräume werden noch von der Eingabe erzeugt.
