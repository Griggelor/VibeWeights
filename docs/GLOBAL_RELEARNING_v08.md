# v0.8.0.dev9 – Globaler Nachlernimpuls und domänenübergreifende Rekonstruktion

Historischer Stand dev9. Persistente globale Zustellung und freiwillige Gruppenbildung über lokale Adapter wurden in [dev10](LEARNING_EVENTS_v08.md) ergänzt.

## Zielvertrag

Nachlernen kann lokal, auf Arbeitsgruppenebene oder global angestoßen werden. Ein globaler Impuls eröffnet eine populationsweite Überprüfung, keine unbedingte Änderung sämtlicher Gewichte. Lernbedarf, Widerspruch, neue Evidenz und später neue Erkenntnisse/Exploration können solche Impulse begründen. Die kognitive Auswahl und Gruppenbildung sollen dezentral erfolgen. Globale Benachrichtigung, Persistenz, Budgets und unabhängige Freigabe sind technische Dienste, kein zentraler kognitiver Router.

Die Architektur bleibt additiv: Lernen und Konsolidieren erzeugen neue Experten; unabhängige evolutionäre Selektion allein archiviert. ACTIVE und LOW_ACTIVE_IMPORTANT bleiben im VRAM, ARCHIVED auf SSD. Sicherheitsquarantäne ist getrennt von evolutionärer Selektion. Keine evolutionäre Löschung oder RAM-Zwischenstufe.

## Implementierter Slice

`GlobalRelearning(TaskLearning(...))` stellt die persistierten Konfliktfälle aus dev8 als globalen Impuls bereit (`impulses()`). Ein offener lokaler Fall bleibt sichtbar; nach lokaler Freigabe bleibt der Impuls so lange sichtbar, bis alle in `pending_reconstruction` aufgeführten betroffenen Experten unabhängig rekonstruiert sind. Die historische Liste von dev8 wird nicht überschrieben; die neue Ansicht berücksichtigt den tatsächlichen Abschluss und aktuelle Quellengültigkeit der Ersatzkandidaten. Die ursprüngliche Konflikt-ID und alle betroffenen Experten sind enthalten. Das ist eine pollbare globale Sicht, kein implementierter Broadcast-Eventbus, keine Zustellgarantie und keine autonome Gruppenbildung. Nicht betroffene Experten bleiben verfügbar.

`stage(case, predecessor, job_id=..., expert_id=..., oracle=..., oracle_id=..., evidence_id=..., method=..., steps=...)` akzeptiert ausschließlich einen quarantänisierten typisierten Lernexperten aus der offenen Rekonstruktionsliste eines lokal geklärten Falls. Für jede historische Root-Identität muss eine autorisierte unabhängige Evidenz einen aktuellen konkreten Wert liefern. Gültige bestehende Roots werden nur bei Übereinstimmung wiederverwendet. Eine verifizierte Root mit inzwischen ungültiger deklarierter Ableitung kann durch eine neu unabhängig bestätigte Revision ohne diese ungültige operative Ableitung ersetzt werden. Historische Root-IDs, Artefakthash, Fallhash, Orakel- und Evidenz-ID bleiben im unveränderlichen Reparaturreceipt erhalten. Das ist eigenständige Neuverifikation, keine erneute Ausführung des ursprünglichen Relationsprogramms. Offene Konflikte, nichtverifizierte Heads, fehlende Evidenz oder neue Widersprüche bei gültigen Heads blockieren die Rekonstruktion; kein blindes Übernehmen alter Werte oder Lehrerantworten. Keine neuen Fakten außerhalb der historischen Root-Identitäten in diesem Slice.

Neue Revisionen, Intake und Reparaturreceipt werden gemeinsam in der Source-SQLite-Transaktion persistiert. Der separate Kandidatenbuild nutzt die bereits vorhandene Wiederanlauflogik. Alte operative Eltern werden nicht als gültige Abhängigkeit des neu unabhängig belegten Kandidaten eingetragen. Das Lernverfahren bleibt DIRECT/TRAINED mit begrenzten Schritten. `promote` verlangt erneut das autorisierte unabhängige Orakel, prüft aktuelle Quellen und führt die vollständige finite-domain Abnahme inkl. Enthaltung außerhalb der Roots durch. Nur nach erfolgreicher Admission wird der Rekonstruktionsversuch als abgeschlossen markiert; bei fehlender Evidenz oder Budget bleibt der Auftrag offen. Ein bereits zugelassener Kandidat kann nach Unterbrechung idempotent abgeschlossen werden. Nach einem endgültig abgelehnten oder quarantänisierten globalen Reparaturkandidaten ist ein neuer Versuch mit neuen IDs möglich. Historische Versuche bleiben erhalten. Unvollständige Versuche müssen zuerst fortgesetzt werden; der allgemeine Retry-Vertrag für den ursprünglichen lokalen Konflikt ist weiterhin offen. Wird eine bereits abgeschlossene globale Rekonstruktion durch eine neue Quellenrevision ungültig, bleibt sie nicht als gültiger Abschluss zählen.

```python
from vibeweights.global_relearning import GlobalRelearning

global_learning = GlobalRelearning(tasks)
for impulse in global_learning.impulses():
    print(impulse)  # global sichtbar, keine automatische Freigabe
# Nach lokaler Konfliktklärung, mit separater unabhängiger Evidenz:
global_learning.stage('conflict', 'affected-child', job_id='repair-child',
                      expert_id='child-v2', oracle=independent_oracle,
                      oracle_id='domain-oracle', evidence_id='fresh-proof')
assert global_learning.promote('repair-child', independent_oracle,
                               oracle_id='domain-oracle')
```

## Nachweis und Grenzen

Kontrollierte Fixture über drei Domänen: Ausgangskonflikt, direkt betroffener Folgeexperte, transitiv betroffener weiterer Folgeexperte. Nach der lokalen Korrektur bleibt der globale Impuls offen. Beide Folgekompetenzen werden unabhängig neu bestätigt (hier ändern sich die Werte), additiv konstruiert bzw. trainiert und getrennt freigegeben. Erst dann verschwindet der Impuls. Unveränderte historische Gewichte, aktuelle Antworten aus neuen Gewichten, fehlende Evidenz, Neustart nach Source-Commit, fehlgeschlagene Promotion und Wiederholung mit neuer Identität werden geprüft. Keine allgemeine semantische oder autonome globale Forschung. Host wählt Arbeitsaufträge, Orakel und Evidenz; keine tatsächliche Modellkommunikation zwischen Gruppen. Nur deklarierte Abhängigkeiten, bestehende Source-Guard-Scans und affine FP32-Experten. Kein Nachweis für externe Quellen, echtes Lehrmodell, Windows, Strix Halo oder physische GPU-Residency; keine vollständige Cross-DB-Atomarität. Offene P0: indexierter Abhängigkeitsgraph, gemeinsame Source-/REMEMBER-Transaktion, autonome budgetierte Impulsbearbeitung, externe Konfliktsignale, lokaler Retry und Rekonstruktion durch erneute Ausführung deklarierter Relationsprogramme.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_global_relearning_v08.py
python -m pytest -q
```
