# v0.8.0.dev10 – Globaler Lern-Eventbus und freiwillige Arbeitsgruppen

## Ziel und Abgrenzung

Globale Lernimpulse sind nicht auf Fehlerkorrektur begrenzt. Auch neue unabhängig bestätigte Kompetenzen können weitere Exploration anstoßen. Ein globaler Impuls eröffnet eine populationsweite Beteiligung, erzwingt jedoch weder ein Training aller Experten noch eine zentral ausgewählte Expertengruppe. Der technische Dienst speichert, verteilt und begrenzt Impulse. Fachliche Interessen und Gruppenschlüssel werden von den einzelnen lokalen Vorschlagsadaptern bestimmt. Lernen und Konsolidieren bleiben additiv; nur unabhängige Selektion archiviert. ACTIVE und LOW_ACTIVE_IMPORTANT sind im VRAM vorgesehen, ARCHIVED auf SSD. Keine RAM-Zwischenstufe oder evolutionäre Löschung.

## Implementierter Referenzslice

`LearningEvents(tasks, membership_budget=64, memberships_per_expert=4)` nutzt die vorhandene Source-SQLite-Datenbank. `sync_conflicts(limit=32)` materialisiert aktuelle globale Impulse aus `GlobalRelearning` in einer persistenten Outbox. RELEARN enthält die aktuelle Konflikt-/Rekonstruktionssicht und zulässige Arbeitsziele (`local`, `repair/<predecessor>`). Wiederholte Synchronisierung desselben Zustands dupliziert nichts. Der Hash enthält die Reparaturversuchshistorie, damit eine später erneut ungültige Folgekompetenz einen neuen Impuls auslösen kann. Nicht mehr aktuelle Impulse bleiben historisch gespeichert, werden aber nicht mehr als Live-Arbeit angeboten. `publish_knowledge(job_id)` erzeugt explizit einen KNOWLEDGE-Impuls für eine zugelassene residente Kompetenz mit aktuellem Quellennachweis, Artefakthash und Explorationsziel. STAGED-Kandidaten dürfen nicht publiziert werden. Nach Archivierung bleibt die gültige Kenntnis discoverable; wird sie widerrufen oder quarantänisiert, ist der Impuls nicht länger aktiv.

`poll(expert, limit=16)` bietet jedem aktuell residenten Experten dieselben gültigen globalen Impulse an, die er noch nicht quittiert hat. Kein fachliches Ranking oder zentraler Router. `respond(event, expert, joins)` speichert eine unveränderliche Empfangs-/Interessensentscheidung; leere Liste bedeutet bewusstes Ablehnen. Ein Experte darf mehrere Gruppen mit selbst gewählten Gruppenschlüsseln vorschlagen; identische (Ziel, Schlüssel)-Paare bilden eine gemeinsame Gruppe. Auch eine Ein-Personen-Gruppe ist zulässig. Mehrfachmitgliedschaften sind möglich. Entscheidung und Mitgliedschaften committen atomar. Wiederholung derselben Entscheidung ist idempotent, Änderung wird abgelehnt. Ziel, Residency, Aktualität und Budget werden vor dem Commit geprüft. `groups(event)` zeigt nur die derzeit gültigen Gruppen mit residenten Mitgliedern. Archivierte und quarantänisierte Experten werden weder automatisch geladen noch als ausführbare Teilnehmer behandelt. Historische Quittungen bleiben erhalten. Mitglieder können in diesem Slice innerhalb eines Impulses nicht aktiv austreten oder ihre Entscheidung revidieren; ein neuer Impuls eröffnet neue Entscheidungen.

`tick(policies, delivery_budget=16)` synchronisiert Konflikte und ruft getrennte lokale Vorschlagsfunktionen höchstens so oft wie budgetiert auf. Jede Funktion erhält den Impuls und liefert eigene Gruppenvorschläge oder eine leere Liste. Technisches Round-Robin mit persistentem Cursor verhindert, dass ein fehlerhafter zuerst sortierter Teilnehmer bei kleinen Budgets alle anderen verdrängt. Kein DB-Lock während Callback-I/O; erneute Live-Prüfung beim Quittieren. Fehler werden als DEFERRED zurückgegeben, ohne eine falsche Quittung zu speichern. Crash zwischen Callback und Commit kann erneute Zustellung verursachen (at-least-once, keine exactly-once-Zusage für externe Nebenwirkungen). Adapter müssen nebenwirkungsfrei bzw. idempotent sein. Die persistente Quittung verhindert eine erneute reguläre Zustellung. Die Entscheidung eines Adapters ist keine Evidenz, kein Lernauftrag mit Freigaberecht und keine Autorisierung zur Rekonstruktion.

```python
from vibeweights.learning_events import LearningEvents
bus = LearningEvents(tasks, membership_budget=64)
event = bus.publish_knowledge('admitted-job')  # bestehender, freigegebener Job
# Beispiel für zwei lokale Vorschlagsadapter (Test-Fixtures, keine Modellgewichte)
policies = {
    'expert-a': lambda event: [('explore', 'relations')],
    'expert-b': lambda event: [('explore', 'relations'), ('explore', 'verification')],
}
# Beide Adapter müssen echte residente Experten-IDs verwenden. In einem
# gemischten Bus müssen sie je nach event['payload']['targets'] entscheiden.
bus.tick(policies, delivery_budget=8)
print(bus.groups(event))
```

## Tests und offene Grenzen

Sieben neue Integrationsfälle: globale Verfügbarkeit und überlappende Gruppen; persistente Quittungen über Neustart und idempotente Wiederholung; Budget- und Zielprüfung ohne Teilcommit; Ausschluss nicht freigegebener/veralteter Impulse; Archivierung ohne automatisches Laden; begrenzte Zustellung und Fairness nach Callback-Fehler/Neustart; überholte Konfliktansicht und erneuter Impuls nach späterer Invalidierung; erneute Quellengültigkeitsprüfung nach Callback. Einige Fälle decken mehrere dieser Eigenschaften ab.

Der Bus ist eine persistente Outbox mit Pull/Zustellschritten im Host, kein verteilter Netzwerk-Push. Die lokalen Vorschlagsadapter sind Python-Callbacks; neuronale Wahrnehmungsköpfe, direkte semantische Nachrichten und autonomes Entstehen/Auflösen von Gruppen sind noch nicht nachgewiesen. Kein eigenständiger Arbeitsgruppen-Executor und keine automatische Ausführung, Evidenzbeschaffung oder Freigabe von Lernjobs. Der Delivery-Budget zählt Callback-Aufrufe, der Membership-Budget gespeicherte Mitgliedschaften pro Ereignis; keine harten Zeit-/Token-/VRAM-Grenzen, keine persistente globale Budgetpolitik und keine Retentionsbegrenzung des Eventjournals. Die Synchronisierung, Source-Guards und Suche können die gesamte relevante Historie scannen; der transitive Abhängigkeitsindex und der gemeinsame Source-/REMEMBER-Transaktionsvertrag sind weiterhin P0. Vertrauenswürdiger einzelner Host-Owner, kein OS-Sandboxing oder Mehrprozess-Protokoll. Kein Echtmodell-/GPU-/Windows-/Remote-CI-Nachweis.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_learning_events_v08.py
python -m pytest -q
```
