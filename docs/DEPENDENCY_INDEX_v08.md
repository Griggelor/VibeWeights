# v0.8.0.dev14 – Indexierter Abhängigkeitsgraph und inkrementelle Quellenprüfung

Historischer Stand dev14. [dev15](MEMORY_TRANSACTION_v08.md) ergänzt den atomaren Source-/REMEMBER-Schreibpfad mit Ausführungsreceipt; vollständige Cross-DB-/Datei-Recovery bleibt offen.

## Ziel und Vertrag

Globales Nachlernen, Konsolidierung und spätere Fragmentassimilation benötigen einen skalierbaren Rückwärtsindex. Bisher prüfte der Source-Guard bei jedem äußeren Aufruf sämtliche Lernjobs und durchlief alle Manifest-Elternbeziehungen. Dev14 ersetzt diesen Pfad im bestehenden `KnowledgeLearning` durch einen abgeleiteten SQLite-Index und transaktionale Änderungsjournale. Die fachliche Validierung `_check_sources` bleibt unverändert: aktuelle verifizierte globale Heads, deklarierte transitive Quellen, Record-Hashes und Datensatzhash müssen übereinstimmen. Keine Änderung der Gewichte, der drei regulären Residency-Klassen oder der unabhängigen Freigabe. Quarantäne bleibt Sicherheitsmaßnahme, keine evolutionäre Löschung.

## Technischer Aufbau

- `dep_sources_v08(record_id, expert)` enthält die vollständige beim Lernjob dokumentierte Quellensnapshot-Menge, einschließlich transitiver Record-Eltern. Der Primärschlüssel unterstützt die Suche nach Konsumenten einer korrigierten Revision. Ein Rückwärtsindex nach Experte erleichtert die Wartung.
- `dep_parents_v08(parent, child)` enthält deklarierte Manifest-Kanten (auch DIRECT-Kompositionen und Gruppen-Summenexperten). Die Traversierung besucht jeden erreichbaren Knoten höchstens einmal. Ein Child-Index erlaubt die Prüfung auf fehlende bzw. quarantänisierte Eltern. `record_conflict` verwendet dieselben Indizes für die initiale Betroffenheitsliste, ohne sämtliche Manifest-JSONs und Jobs zu lesen; Jobs ohne Artefakt sind keine operativen Rekonstruktionsziele.
- SQLite-Trigger aktualisieren die abgeleiteten Indexeinträge und protokollieren relevante Katalogänderungen in derselben Katalogtransaktion. Quellen-Trigger protokollieren alte/neue Head-IDs, Record-Änderungen und Konfliktfälle in derselben Source-Transaktion. Abgebrochene Source- oder Katalogtransaktionen hinterlassen keine bestätigten Journalereignisse. Die Journale besitzen sequenzielle IDs und zufällige Tokens zur Erkennung wiederverwendeter Anker nach Rollback.
- Bei der ersten Prüfung eines Owners und nach Restart wird der Index aus den vorhandenen Jobs und Manifesten aufgebaut und der gesamte Bestand validiert. `audit_sources(full=True)` erlaubt eine explizite vollständige Revalidierung (kein Index-Rebuild während laufender Nutzung). Bei nicht nachvollziehbarem Journalanker oder struktureller Job-/Manifest-/Konfliktänderung fällt der Guard auf vollständige Prüfung zurück. Die Checkpoints leben ausschließlich im Prozess. Keine dauerhafte Löschung alter Journalereignisse.
- Im normalen Guard werden seit dem letzten erfolgreichen Prüfpunkt betroffene Quellenkonsumenten, geänderte Katalogeinträge, neue Konfliktbeteiligte und deren Folgeexperten ermittelt. Erst diese Jobs werden per `_check_sources` erneut geprüft. Bei keinem Ereignis werden keine Job-Snapshots geladen. Der Source-Lock bleibt bestehen; der Checkpoint rückt erst nach erfolgreicher Sicherheitsquarantäne vor. Neue Nachkommen bereits quarantänisierter Experten werden ebenfalls gesperrt. Eine Katalogmanifest-/Lernjob-Identitätsabweichung wird fail-closed behandelt.
- `Population._check` vergleicht weiterhin die vollständige residenten ID-Menge, jetzt per SQL-State-Index statt durch Parsing sämtlicher Manifest-JSONs. Weitere Pfade (Task-Auswahl, Events, Rekonstruktion usw.) können weiterhin Katalogscans ausführen. Kein globaler Denkrouter.

## Reproduzierbares Experiment

```sh
python -m pip install -e ".[test]"
python scripts/dependency_index_demo.py
python -m pytest -q tests/test_dependency_index_v08.py
python -m pytest -q
```

100 einzeln freigegebene affine Lernexperten (je vier Entitäten, drei Labels, ein verifiziertes Faktum) und drei verkettete DIRECT-Folgeexperten. 100 alternierende Messungen pro Modus nach Warmup, gleicher äußerer Source-Lock, keine neuronale Inferenz im Messintervall. Median inkrementell ohne Änderung 53,3935 µs (0 Quellenprüfungen), explizit vollständig 10.780,7655 µs (100 Quellenprüfungen), Verhältnis ca. 202. Bei einer Quellenrevision: ein Lernjob geprüft, vier Experten quarantänisiert (Ursprung + drei transitive Nachfolger), 99 residente Experten verbleiben; 2.445,846 µs einschließlich Quarantäne. Alle ursprünglichen Artefakthashes unverändert; nachfolgender Vollaudit findet keine zusätzlichen zu sperrenden Experten. Siehe [Rohmessungen](../evidence/v08-dependency-demo.json). Einzelne CPU-Messreihe, kein Vergleich kompletter Releases, kein End-to-End-/GPU-Benchmark und kein Skalierungsnachweis über 100 Lernjobs hinaus.

## Grenzen

Vertrauenswürdiger einzelner Owner, keine OS-Sandbox, kein Schutz gegen willkürliche Manipulation der Index-/Journaltabellen, keine konsistente Mehr-Owner-Katalogausführung. Die Indexinstallation nutzt SQLite-JSON-Funktionen; auf der lokalen Python-/SQLite-Version geprüft, Windows-Kompatibilität noch nicht abgenommen. Startprüfung/Index-Rebuild bleibt linear in den katalogisierten Jobs und Kanten. Journale wachsen, keine Kompaktierung oder formelle Worst-Case-Budgets. Nur deklarierte Provenienz ist sichtbar; unbekannte semantische Zusammenhänge müssen später durch Wahrnehmung/Recherche entdeckt werden. Kein genereller Index für beliebige Memory-Reads, keine automatische Gruppenrekonstruktion oder Änderung von Gewichten. Source und Katalog bleiben getrennte DBs; die Trigger atomarisieren ihre jeweiligen lokalen Änderungen, nicht die gesamte Inferenz-/Memory-/Admission-Kette. Gemeinsamer REMEMBER-/Learning-Transaktionsvertrag und vollständige Cross-DB-Recovery bleiben P0. Reale LLM-, Fragment-, Windows-, GPU- und Remote-CI-Validierung nicht durchgeführt.
