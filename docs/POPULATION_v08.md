# v0.8.0.dev1 — erster Population-/Residency-Slice

Status: ausführbarer CPU-Referenzbaustein, kein vollständiges v0.8-Release. Ergänzt den Knowledge-first-Lernvertrag. Der alte Chat/API-Pfad ist noch nicht migriert. Keine neue Modellintelligenz oder trainierte Gewichte in diesem Schritt.

## Drei reguläre Zustände

| Status | Ziel-Residency | Verhalten |
|---|---|---|
| ACTIVE | VRAM | regulär verfügbar |
| LOW_ACTIVE_IMPORTANT | VRAM | selten, aber wichtig; bleibt resident |
| ARCHIVED | SSD | keine residenten Gewichte; Katalog und Original erhalten |

Keine CPU-RAM-Cache-Stufe. Der CPU-Backendlauf emuliert die Residency; auf einem passenden PyTorch-CUDA/HIP-Backend kann `device='cuda'` übergeben werden. Das wurde hier nicht auf GPU getestet. Der `Catalog`-Root ist der permanente Artefaktspeicher; auf dem Zielgerät muss er auf SSD liegen. Kandidatenstatus STAGED ist kein regulärer freigegebener Expertenzustand.

## Implementierung

- `factory.py`: ACTIVE und LOW_ACTIVE_IMPORTANT regulär ladbar, ARCHIVED nur ausdrücklich experimentell. Dateiexistenz, Hash, Form und FP32-Werte werden beim Laden geprüft. Optionale Gerätetransfers; Eingabe auf Gewichtsgerät. Aufnahme verwendet State-CAS statt bedingungsloser Änderung.
- `population.py`: `Population` besitzt die residenten Tensorobjekte. Start rekonstruiert alle regulär residenten Experten aus dem SQLite-Katalog; Budgetüberschreitung oder defektes Artefakt bricht ab, ohne stille Auslagerung. `admit` fügt nur hinzu, `compose_affine` erzeugt ausschließlich einen neuen STAGED-Kandidaten. `run` verbindet die residenten Experten mit dem numerischen Environment; ARCHIVED ist aus diesem Run ausgeschlossen. Kein zusätzlicher Router.
- `select` ist separat von `admit` und `compose_affine`, erfordert einen expliziten Host-Evidenzreport und die erwartete Selektionsrevision. Archivierung erfordert ausreichende Beobachtungen, bestätigte Abdeckung, Ablation und geklärte Abhängigkeiten sowie fehlende Wichtigkeit. LOW_ACTIVE_IMPORTANT erfordert niedrige Aktivität und Bedeutung. Reaktivierung erfordert Nutzenbeleg, Integrität und freien Budgetplatz. Revision und Evidenzjournal werden gemeinsam mit dem Zustandswechsel in SQLite transaktional geschrieben. Veraltete Entscheidungen werden abgelehnt.
- Archivierung entfernt die Referenz auf das resident gehaltene Objekt; unveränderliche SafeTensors und Manifest bleiben auf SSD. Reaktivierung lädt das Original. Keine Löschoperation, kein LRU-Eviction, kein automatisches Downgrading wichtiger Experten.

## Verwendung (CPU-Referenz)

```python
from vibeweights.factory import Catalog
from vibeweights.population import Population, SelectionEvidence, LOW_ACTIVE_IMPORTANT, ARCHIVED, ACTIVE

c = Catalog('Data/experts')
p = Population(c, tensor_budget=1024, min_observations=10)
try:
    c.stage_affine('example@1', 'example', [[2]], [0], source_topic='input',
                   target_scope='global', target_topic='result')
    assert p.admit('example@1', [((2,), (4,)), ((-3,), (-6,))])
    # Illustrative host-attested report, not measurements from a real evaluator:
    report = SelectionEvidence('example-report', 10, True, True, False, False, False, True)
    p.select('example@1', LOW_ACTIVE_IMPORTANT, expected_revision=0, evidence=report)
    assert p.compute('example@1', (7,)) == (14,)
finally:
    p.close()
    c.close()
```

This example assumes a new catalog/ID. Use `revision(id)` and `history(id)` for existing experts. `resident_ids()` and `tensor_bytes` report objects held by the reference population. Physical GPU allocator behavior is not established by these values.

## Grenzen und nächste Arbeiten

- Nur vertrauenswürdige Host-Funktionen und affine FP32-Experten. Reports sind Host-Atteste, keine automatisch aus Langzeitmetriken berechneten Beweise. `min_observations=1` ist ein Referenzdefault, keine wissenschaftlich bestätigte Schwelle. Echte unabhängige Evidenzgenerierung, seltene Kompetenzen, Langzeitstatistik, Exploration im Archiv und transitive Abhängigkeitsprüfung sind offen. Keine selbstständige Archivierung.
- Ein Population-Owner pro laufendem Katalog, Aufrufe zwischen Runs. Keine Multi-Prozess-Residency-Koordination, keine Sicherheitsgrenze gegen direkten SQLite-/Catalog-Zugriff. Legacy-Factory-Aufrufe bleiben für Regressionen möglich. Wiederanlauf liest den persistenten Zustand; kein vollständiges Crash-/Power-loss-Fault-Injection-Programm.
- `tensor_budget` zählt FP32-Nutzdaten resident gehaltener Gewichte, nicht temporäre Tensoren, Caches, Allocator-Overhead, Aktivierungen oder andere Prozesse; `artifact_budget` begrenzt Einzeldateladen. Keine gemessene physische VRAM-Freigabe, kein SSD-Durchsatzbenchmark, keine Sicherung gegen SSD-Ausfall, keine quantisierten Experten. Temporäre CPU-Puffer sind keine zusätzliche Residency-Klasse. GPU/Windows/Strix-Halo-Abnahme ausstehend.
- Keine echte Wissensaufnahme in Gewichte in dieser Runde. Nächster P0: Learning Ledger + kleines DIRECT/TRAINED-Wissensexperiment, unabhängige Tests ohne Lehrer, Bootstrap und Faktenabruf; danach die evidenzbasierte Selektionsbewertung mit dem bestehenden Population-Vertrag verbinden.

## Nachweis

115 lokale CPU-Tests (104 Regressionen + 11 neue Testfälle). Siehe `evidence/v08-population-tests.log`, `evidence/v08-population-tests.xml`, `evidence/v08-population-summary.json`. Ursprüngliche v0.7-Testevidenz und das alte Importmanifest bleiben historische Baseline, nicht Prüfsumme des neuen Entwicklungsstandes. Kein GPU-, Windows- oder GitHub-CI-Ergebnis aus diesem Lauf.
