# v0.8.0.dev8 — Konfliktstatus, persistenter Klärungsauftrag und additives Nachlernen

Historischer Stand dev8. Für globale Impulse, domänenübergreifende unabhängige Rekonstruktion und Retry globaler Kandidaten siehe [dev9](GLOBAL_RELEARNING_v08.md).

## Verbindlicher Vertrag

Ein tatsächlicher Wissenswiderspruch ist kein bloßes Telemetrieereignis. Ein betroffener Wissenseintrag darf nicht weiter als gesichert verwendet werden. Die Klärung wird persistent als Lernbedarf erfasst. Eine unabhängige Evidenz muss bestätigen oder korrigieren, anschließend wird ein neuer Kandidat erzeugt und separat freigegeben. Keine Mehrheitsentscheidung zwischen Experten als Wahrheit, kein Überschreiben oder evolutionäres Löschen von Gewichten. Die drei regulären Residency-Zustände bleiben ACTIVE (VRAM), LOW_ACTIVE_IMPORTANT (VRAM), ARCHIVED (SSD); QUARANTINED ist eine gesonderte Sicherheitssperre.

Der Wissensstatus ist unabhängig vom Expertenstatus. Die Ledger-Statuswerte OBSERVED, HYPOTHESIS, VERIFIED werden um UNCERTAIN (begründeter Zweifel), CONFLICTED (konkreter Widerspruch) und INVALIDATED (widerlegte Aussage) erweitert. Dev8 setzt CONFLICTED bei tatsächlich beobachteten unterschiedlichen nichtleeren Ausgaben in einer typisierten Domain automatisch. Andere neue Statuswerte sind im Ledger zulässig, noch keine autonomen Evidenzklassifikatoren. Der immutable frühere Record bleibt als Historie erhalten; ein neuer nichtverifizierter Head ersetzt seine Live-Gültigkeit, nicht seinen Inhalt. `current_verified` prüft aktuelle Heads und deklarierte transitive Eltern; Memory-READ und Bootstrap-Kontext benutzen diese Prüfung. `read` bleibt explizit historisch; die Standardsuche filtert aktuelle VERIFIED-Heads, kann aber IDs von abgeleiteten Records mit inzwischen ungültigen Eltern liefern, die erst beim vertrauenswürdigen Lesen abgewiesen werden.

## Ablauf

- `TaskLearning.observe` erkennt widersprechende Antworten aus zugelassenen residenten Gewichtsjobs. Es speichert die konkreten Ausgaben, betroffene Experten, Abhängigkeiten und die CONFLICTED-Revision mit einem persistenten Konfliktfall in *derselben* Source-SQLite-Transaktion. Folgeabfragen liefern CONFLICT/keine Antwort, auch wenn das ursprüngliche Zusammentreffen nicht mehr reproduzierbar ist. Kein wiederholter redundanter Fall für denselben offenen Fakt.
- Sicherheitsprüfung sperrt konservativ alle widersprechend beteiligten Experten und deklarierte Nachkommen, außerdem Kandidaten, deren verifizierte Quellen durch die neue Revision ungültig werden. Gewichte und Registry-Einträge bleiben erhalten. Nicht betroffene Experten bleiben verfügbar. Dies ist keine evolutionäre Selektion. Der Slice ist absichtlich fail-closed auf Expertenebene, nicht auf einzelne Gewichtsassoziationen begrenzt. Auch ein fachlich korrekter Teilnehmer kann dabei vorübergehend unverfügbar werden.
- `ConflictLearning.pending()` listet offene Klärungsfälle. `ExtensionLearning.review_conflict` dokumentiert unabhängige Evidenz zu den persistenten Ausgaben, nicht eine erzwungene Wiederinferenz der gesperrten Experten. Ohne hinreichende unabhängige Evidenz keine neue VERIFIED-Revision und keine Aufhebung der Sperre.
- `ConflictLearning.stage` akzeptiert explizit autorisierte Evidenz-/Orakel-ID, deterministische unabhängige Verifikationsfunktion, eindeutige Job-/Experten-ID und begrenztes Lernverfahren. Es erzeugt eine neue VERIFIED-Revision für das geklärte Faktum und einen dauerhaften Intake in derselben Source-Transaktion. Die Evidenz kann die bisherige Aussage bestätigen, korrigieren oder beide widersprüchlichen Antworten widerlegen. Die alten Gewichte werden nicht verändert. Der neue Kandidat wird DIRECT konstruiert oder per SGD TRAINED. Der Fakt wird im Memory nach unabhängiger Bestätigung wieder verfügbar; der typisierte Aufgabenpfad bleibt bis zur Kandidatenfreigabe gesperrt.
- Zusätzlich übernimmt der neue Kandidat unabhängig bestätigte, weiterhin aktuelle Fakten aus betroffenen Experten derselben Domain, um einen unnötigen Kompetenzverlust zu vermeiden. Fehlende/ungültige/unbestätigte Roots und domainfremde Abhängigkeiten werden als `pending_reconstruction` dokumentiert und sind weiterhin auffindbar über `reconstruction_pending()`. Alte Quarantäne-Experten werden nicht als gültige operative Eltern des neuen Experten eingesetzt. Die historische Herkunft steckt im Konfliktfall und in der Intake-Derivation.
- `ConflictLearning.promote` prüft mit unabhängig bereitgestellten Sollwerten und der vorhandenen exhaustive finite-domain Abnahme; unbekannte Entitäten müssen Enthaltung ergeben. Erst erfolgreiche Admission (bzw. deren nachweisbarer Wiederanlauf) schließt den Konflikt. Fehlgeschlagene Promotion/abweichende Evidenz hält ihn offen. Alte Artefakte bleiben in Quarantäne auf SSD. Kein automatischer Wechsel zu einer unbewiesenen Aussage.

## Nutzung

```python
from vibeweights.task_learning import TaskLearning
from vibeweights.conflict_learning import ConflictLearning

tasks = TaskLearning(learning)
obs = tasks.observe('contradiction', domain, entity)
if obs['outcome'] == 'CONFLICT':
    conflicts = ConflictLearning(tasks)
    case_id = obs['conflict_case']
    # oracle: vom Kandidaten unabhängige, hostautorisierte Wahrheitsquelle
    conflicts.stage(case_id, job_id='repair-job', expert_id='repair-expert',
                    oracle=oracle, oracle_id='independent-oracle',
                    evidence_id='evidence-2026', method='TRAINED')
    assert conflicts.promote(case_id, oracle, oracle_id='independent-oracle')
```

Persistenz: Unterbrechung nach Source-Commit, vor Taskjournal oder Gewichtsbuild lässt den Fall bzw. das Intent auffindbar. Nach erfolgter Admission, aber vor Case-Close ist die Aufgabe weiterhin gesperrt; erneute Promotion mit gleicher Evidenz schließt den Fall. Kein allgemeiner Cross-DB-Atomaritäts- oder Power-Loss-Nachweis. Live-Source-Guard bleibt scanbasiert. Frühere dev7-Konfliktberichte ohne gespeicherte Ausgaben müssen erneut beobachtet werden; keine erfundene Migration historischer Evidenz.

## Grenzen

Die automatisierte Reaktion gilt zunächst ausschließlich für beobachtete Konflikte im typisierten numerischen TaskLearning-Pfad, nicht für beliebige natürliche Sprache, Benutzerkorrekturen oder allgemeine Recherchesignale. Der persistente Klärungsauftrag wird automatisch erzeugt; Evidenzbeschaffung, Ausführung des Nachlernens und Promotion sind weiterhin explizite vertrauenswürdige Host-Aufrufe, kein autonomer Scheduler oder unkontrolliertes Selbstlernen. Die unabhängigen Orakel sind Test-Fixtures; keine reale externe Wissensprüfung, kein reales LLM. Nach einem endgültig fehlgeschlagenen Kandidaten bzw. nochmals geänderter Source muss ein neuer autorisierter Versuch mit Historie ermöglicht werden (noch offen; der Fall bleibt fail-closed). Keine automatische Rekonstruktion domainfremder Nachkommen. Eine spätere kleinteiligere faktspezifische Freigabe und indexierte Abhängigkeiten sollen breite Quarantäne und Scanaufwand vermeiden. Kein Windows-, GPU-, Strix-Halo- oder Remote-CI-Nachweis. Generischer Chatpfad weiterhin nicht migriert.

## Tests

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_conflict_learning_v08.py tests/test_extension_learning_v08.py tests/test_task_learning_v08.py
python -m pytest -q
```

Neue Tests: DIRECT/TRAINED-Korrektur einschließlich dritter neuer Antwort, Retention der bestätigten alten Kompetenz, Unsicherheitsstatus und historische Revisionen, kein Abschluss bei fehlender Evidenz, Ablehnung unzureichender Gewichte/falscher Verifikation, Wiederanlauf nach Fehler vor Build und nach Admission, transitive Quarantäne und separat ausgewiesene offene Rekonstruktion, Fehler nach Source-Commit vor Taskjournal. Konfliktreview-Test aus dev7 absichtlich an neuen Sperrvertrag angepasst; der Review selbst ändert keine weiteren Gewichte/Population. Alle Testkonflikte entstehen durch kontrolliert injizierte Ausgabefehler, nicht reale Quellenstreitigkeiten. Die Recovery-Tests dieser Runde injizieren Ausnahmen und öffnen DBs erneut, keine zusätzlichen harten Prozessabbruchtests.
