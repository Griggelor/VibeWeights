# v0.8.0.dev2 — erster parametrischer Wissenserwerb

Historischer Status dev2: ausführbarer CPU-Referenzslice. Für aktuelle automatische Quellenprüfung und relationale Lernkette siehe [dev3](RELATION_LEARNING_v08.md). 127 lokale Tests bestanden (115 Regressionen + 12 neue Fälle). Ergänzt den [Lernvertrag](KNOWLEDGE_FIRST_LEARNING.md) und die [Population](POPULATION_v08.md). Kein vollständiges v0.8-Release, kein allgemeines autonomes Lernen.

## Wissensdomäne und Verfahren

`Domain(topic, entities, labels)` begrenzt die erste Domäne auf 1–128 Entitäten und 2–32 Klassen. Jede Entität hat eine fest definierte One-Hot-Identität; keine Frage-Antwort-Tabelle im Encoder. Die Ausgabe ist ein Klassenvektor, Nullvektor oder uneindeutige Ausgabe wird durch `decode` als unbekannt (`None`) behandelt. Kein semantischer Sprachencoder, kein Reasoning, keine Generalisierung auf ungesehene Fakten. Außerhalb des Domains führt die Identitätskodierung zu Null. Die domänenspezifische Eingabe muss vom Host korrekt erzeugt werden; beliebige numerische Vektoren sind nicht als gültige Entitätsabfragen definiert.

- DIRECT: affine Assoziationsmatrix analytisch setzen, Bias null. Faktenspeicherung in Gewichten, nicht mathematische Intelligenz.
- TRAINED: dieselbe affine Architektur, Nullinitialisierung, echte PyTorch-SGD-Optimierung (Standard 80, maximal 256 Schritte, LR 0,25), quadratischer Fehler, Bias fest null. Kein nachträgliches Überschreiben der trainierten Gewichte mit Sollwerten. Keine Destillation eines echten Lehrmodells.
- Beide: unveränderliche SafeTensors und SHA-256, Herkunft DIRECT/TRAINED im Factory-Manifest, Learning-Job-ID, reguläre additive Population. Keine Änderung der drei freigegebenen Expertenzustände, keine Löschung.

## Learning Ledger und Freigabe

`KnowledgeLearning(population, knowledge, access)` verwendet drei neue Tabellen in derselben SQLite-DB wie der Expertenkatalog: `knowledge_jobs_v08` (unveränderliche Spezifikation, Dataset-Hash, Domäne, Methode, konkrete Quellenrevisionen und transitive Elternreferenzen mit Hash), `knowledge_builds_v08` (Schrittzahl, Fehler vor/nach Optimierung), `knowledge_reports_v08` (Orakel-ID, Fallhash, Artefakthash, Anzahl bekannte/unbekannte Fälle, bestandene Fälle, Entscheidung). Status wird aus dem Katalog gelesen; ein Job ohne Artefakt ist INCOMPLETE. Keine Wiederverwendung einer Job-/Experten-ID. Kein automatischer Wiederanlauf eines unterbrochenen Build; nach Budgetverweigerung bleibt STAGED erhalten und kann nach Neustart mit demselben Prüfbericht erneut geprüft werden.

`prepare` akzeptiert ausschließlich aktuelle VERIFIED-Records im globalen Scope, inklusive transitiver Eltern (max. 512). Schlüssel `<topic>:entity:<id>`, Topic passend zur Domäne, semantisch, genau ein ganzzahliger Klassenwert. Der Host verantwortet weiterhin die Wahrheit, Rechte und Eignung von VERIFIED; dieser Slice prüft keine externen Quellen. Keine privaten oder ungeklärten Records. `prepare` erzeugt nur STAGED.

`promote` sperrt die Wissens-DB für konkurrierende Schreiber während erneuter Quellenprüfung und Admission; Host-Orakel liefert Sollwerte für alle gültigen Entitäten und zwei außerhalb des Bereichs. Der Vorwärtslauf bekommt ausschließlich den Eingabevektor, nie Sollwerte. Alle Fälle müssen bestehen; keine Durchschnittsfreigabe. Bericht ist unveränderlich an Fall- und Artefakthash gebunden; generischer Catalog-/Population-Admit ohne passende Learning-Job-ID wird verweigert. Der Host bleibt vertrauenswürdig; kein Sicherheits-Sandboxing oder Schutz gegen direkt manipulierte SQLite-DB. Prüfbericht vor Admission gespeichert; Katalogstate ist Wahrheit über tatsächliche Aufnahme. Ein Budgetfehler erzeugt keinen residenten Kandidaten. Keine behauptete vollständige Cross-DB-Crash-Atomarität.

## Korrekturen und Quarantäne

`audit_sources()` prüft alle Jobs auf geänderte Quellen und abhängige Experten anhand deklarierter Katalog-Eltern. Betroffene STAGED-, ACTIVE-, LOW_ACTIVE_IMPORTANT- und ARCHIVED-Kandidaten werden über eine separate Sicherheitsoperation QUARANTINED; Gewichte, Herkunft und Historie bleiben erhalten, residente Referenz entfällt. Reguläres SELECT kann Quarantäne nicht reaktivieren. Korrigiertes Wissen erzeugt einen neuen Experten. Keine evolutionäre Löschung. QUARANTINED ist kein vierter regulärer Residency-Status. Aufruf nach Wissensrevisionen und vor Nutzung eines verbundenen Systems ist derzeit Host-Pflicht; automatische Invalidation vor jeder Inferenz/Archivreaktivierung und persistente Cross-DB-Recovery stehen aus. Offline-Artefakte können späteren Widerruf naturgemäß erst nach Synchronisierung erfahren. Nur deklarierte Abhängigkeiten erfasst, kein vollständiger Informationsflussnachweis.

## Reproduzieren

Vom Repo-Root nach `python -m pip install -e ".[test]"`:

```sh
python scripts/knowledge_learning_demo.py
python -m pytest -q tests/test_knowledge_learning_v08.py
python -m pytest -q
```

Minimaler Host-Aufruf (frischer Katalog):

```python
from vibeweights.factory import Catalog
from vibeweights.population import Population
from vibeweights.knowledge import KnowledgeLedger, Access
from vibeweights.knowledge_learning import Domain, KnowledgeLearning
c = Catalog('Data/experts')
p = Population(c, tensor_budget=65536)
k = KnowledgeLedger('Data/knowledge.sqlite')
a = Access('host')
d = Domain('demo', 4, 3)
# Kontrollierte Fixture; in echter Umgebung unabhängige Evidenz erforderlich.
truth = {0: 2}
r = k.store(a, scope='global', key='demo:entity:0', topic='demo',
            values=(2,), source='fixture', status='VERIFIED')
l = KnowledgeLearning(p, k, a)
l.prepare('job-1', 'expert-1', d, [r.id], method='TRAINED')
assert l.promote('job-1', truth.get, oracle_id='fixture-oracle')
assert d.decode(p.compute('expert-1', d.encode(0))) == 2
assert d.decode(p.compute('expert-1', d.encode(3))) is None
p.close(); c.close(); k.close()
```

## Nachweise und Grenzen

Demo: 6 künstlich erzeugte Fakten, 12 mögliche Entitäten, 5 Klassen, je 65 Parameter; beide Pfade 14/14 Fälle korrekt (6 bekannt, 8 unbekannt inkl. zwei außerhalb des Bereichs). Neustart ohne Wissens-DB erfolgreich; Tests zusätzlich in separatem Python-Prozess, der nur kopierte Gewichtedatei, Identitätsdimensionen und Inferenzcode verwendet, ohne Fakten- oder Lehrerzugriff. Kein OS-Sandbox-/Netzwerkverbotstest. Nulloutput-Ablation verliert alle bekannten Fakten. Drei sequentielle, getrennte Wissensdomänen behalten ihre früheren Zuordnungen; dies ist keine Retentionsprüfung eines gemeinsam trainierten Netzes. Keine gemessene semantische Generalisierung, kein unbekannter Fakt kann ohne zusätzliche Evidenz gewusst werden. Exhaustive Recall-/Abstention-Prüfung einer endlichen Domäne; bekannte Fakten sind bewusst dieselben wie im Lerndatensatz, nicht unabhängige Generalisierungs-Holdouts. Orakel ist unabhängig vom Kandidaten in der Fixture definiert, keine externe Forschungsreplikation. Die vollflächige Matrix ist kein Nachweis wirtschaftlicher Wissensspeicherung gegenüber Tabellen.

Weitere Tests: falscher VERIFIED-Inhalt, unzureichendes Training, nicht freigegebene/private Quellen, Quellrevision vor Freigabe, Korrektur mit transitive Abhängigkeiten, Budgetverweigerung und erneute Freigabe nach Neustart, keine generische Admission ohne Job-ID. Evidence: `evidence/v08-knowledge-learning-demo.json`, `v08-knowledge-learning-summary.json`, Testlog und JUnit. Kein Windows/Strix-Halo/GPU/Remote-CI-Nachweis; alter Chatpfad weiterhin nicht migriert.

Nächster P0: automatische Quellenrevision-/Quarantänebindung und Recovery, reale lokale Lehrmodell-Anbindung mit unabhängig überprüfbaren Lerninhalten, modellinitiierte Lernaufträge, bekannte Regeln und neue ableitbare Kombinationen als echte Holdouts; Ledger-only vs Gewichte inkl. Gesamtkosten und Langzeit-Retention. Additive Summenexperten und unabhängige Selektion bleiben getrennt.
