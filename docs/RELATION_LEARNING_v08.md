# v0.8.0.dev3 — Wissensverknüpfung, erneutes Lernen und Live-Quellenschutz

Status: ausführbarer CPU-Referenzslice; keine Environment-Abnahme. Baut auf [dev2](KNOWLEDGE_LEARNING_v08.md) und dem [Lernvertrag](KNOWLEDGE_FIRST_LEARNING.md) auf. Die drei regulären Zustände ACTIVE (VRAM), LOW_ACTIVE_IMPORTANT (VRAM), ARCHIVED (SSD), additive Aufnahme/Konsolidierung, unabhängige Selektion und keine evolutionäre Löschung bleiben unverändert. QUARANTINED ist ein separater Sicherheitszustand.

## 1. Neue relationale Lernkette

`derive_relation` in `vibeweights.relation_learning` verbindet zwei bereits freigegebene affine Wissensexperten. Der Host deklariert die Identität des Zwischenraums (`bridge_id`), beide Elternjobs und eine neue Zieldomäne. Die Anzahl der Klassen des ersten Experten muss mit der Entitätenzahl des zweiten übereinstimmen. Diese Prüfung beweist keine Semantik; die Identität der ID-Räume ist eine vertrauenswürdige Host-Vorgabe. Unabhängige vollständige Fixture-Prüfung prüft die Resultate, nicht die Entdeckung des Bridges.

Beispiel: Objekt → Material → Klasse. Der erste Experte liefert eine numerische Klassenrepräsentation, die eine reine Identitätskodierung in eine Eingabe für den zweiten Experten überführt. Der zweite liefert die Zielklasse. Die Brücke enthält keine Antworttabelle. Bei fehlendem Zwischenschritt oder unbekannter Antwort wird nicht geraten. Die neue Relation ist vorab nicht als Fakt im Ledger vorhanden; die Vorwärtsberechnung stammt aus den Elterngewichten. Quelleninhalte werden zur Revisions-/Hashprüfung gelesen, nicht zur Berechnung der Antworten verwendet. Der Host gibt das Lernziel vor; keine spontane Gruppenbildung oder selbstinitiierte Lernstrategie.

Das unabhängige Orakel prüft alle Eingaben einschließlich unbekannter Fälle; kein Sollwert wird in die Vorwärtsrechnung eingespeist. Erst nach vollständigem Erfolg werden alle neuen positiven Fakten als ein atomarer Ledger-Batch geschrieben. Jeder Fakt referenziert die beiden konkret verwendeten Quellenrevisionen. Das neue STAGED-Artefakt wird anschließend mit DIRECT oder TRAINED erzeugt; Eltern-Experten, Elternjobs, Bridge-ID, Orakel-ID und Prüfbereich werden gespeichert. Die bestehende getrennte `promote`-Abnahme prüft den Kandidaten erneut. Die Kandidatenaufnahme ist rein additiv. Bei einem Fehler zwischen gültigem Fakten-Batch und Kandidatenbau bleiben die verifizierten Fakten erhalten; automatischer Wiederanlauf/Recovery über die beiden DBs ist offen. Der neue Lernpfad verlangt einen frischen Ziel-Namespace und eindeutige Job-/Experten-IDs.

## 2. Live-Quellenschutz

`KnowledgeLearning(population, knowledge, access)` bindet den Population-Owner an die Quellendatenbank. `Population.compute`, `run`, `admit`, `select` prüfen automatisch die Quellenrevisionen und transitive deklarierte Elternabhängigkeiten vor Ausführung. Ein `BEGIN IMMEDIATE` hält konkurrierende Quellschreiber bis zum Ende der Operation ab. Verschachtelte kontrollierte Aufrufe teilen die Sperre; keine parallele Schreibtransaktion auf dieser Verbindung. Ein Lock-/DB-Fehler unterbindet die Operation. Korrigierte, widerrufene oder nicht mehr aktuelle Quellen führen zur Quarantäne aller betroffenen und deklariert abhängigen Experten; Artefakte bleiben erhalten. Dies gilt auch für archivierte Kandidaten vor Reaktivierung. Unabhängige Experten bleiben verfügbar.

Historischer Offline-Betrieb ist nun ausdrücklich opt-in: `Population(catalog, tensor_budget=..., allow_offline_knowledge=True)`. Bei vorhandenen Learning-Job-Artefakten ohne gebundenen Source-Owner verweigert die Population standardmäßig Inferenz, Admission und Selektion. Expliziter Offline-Modus erlaubt nur `compute`/`run` als historisches Experiment, keine Admission/Selektion. Direkte Low-Level-Catalog-/AffineExpert-Aufrufe bleiben Forschungsprimitive und sind keine Security-Grenze. Konstruktion der Population kann vor Bindung Gewichte laden, aber nicht produktiv ausführen. `resident_ids` ist eine Zustandsaufnahme, kein aktueller Quellenprüfbericht. Es gibt keinen Hintergrunddienst; Schutz erfolgt am Operationsrand. Offline-Gewichte können spätere Revisionen erst nach Synchronisierung erfahren.

Die Prüfung scannt derzeit alle Lernjobs und deklarierten Eltern; kein Skalierungsnachweis, keine inkrementelle Invalidierung. Ein quellengebundener `run` mit derselben Ledger-Instanz und `remember_experts` wird explizit verweigert, da der Quellenlock einen verschachtelten REMEMBER-Schreibvorgang nicht erlaubt. Ein integrierter Transaktionsvertrag für Inferenz plus Memory-Writes, vollständige Crash-Recovery und OS-/Rechteisolation bleiben P0. Vertrauenswürdiger einzelner Host; kein Schutz gegen manipulative Python-Adapter, direkte Datenbankzugriffe oder Orakel, das selbst die Sperre verletzt.

## 3. Lokaler Lehrer-Eingang

`stage_teacher_knowledge` verwendet die bereits vorhandene `BootstrapExpert`-Loopback-Textbrücke (OpenAI-kompatibel). Der Host übergibt eine ausdrücklich autorisierte Anfrage und eine unabhängig überprüfbare endliche Domäne. Nur striktes JSON `{ "facts": [{ "entity": 0, "label": 2 }] }` ist zulässig; doppelte Schlüssel/IDs, Bool-Werte, ungültige IDs, zusätzliche Felder, unvollständige oder falsche Fakten werden abgelehnt. Keine Ausführung von Modelltext, keine Übernahme eines VERIFIED-Flags vom Lehrer. Das unabhängige Orakel prüft alle Entitäten vor atomarer Speicherung. Anfrage-/Antwort-Hashes, Lehrer-/Modell-ID und Orakel-ID stehen im Lernjob. Der Host verantwortet Quellennachweis, Rechte und Unabhängigkeit des Orakels; die Hashes sind kein Beweis der Wahrheit oder Lizenz. Die nachgelagerte Promotion bleibt getrennt.

Tests verwenden einen Transport-Double der vorhandenen Textbrücke; frühere HTTP-Testserver-Evidenz bleibt separat. Kein tatsächliches Qwen/LM Studio oder reales externes Wissen in diesem Lauf. Kein automatischer Modellaufruf aus der Population. Der Adapter akzeptiert aktuell nur numerische Zuordnungen in einem frischen Namespace, nicht allgemeine natürliche Sprache.

## 4. Reproduzierbare Evidenz

Nach `python -m pip install -e ".[test]"` im Repo-Root:

```sh
python scripts/relation_learning_demo.py
python -m pytest -q tests/test_relation_learning_v08.py
python -m pytest -q
```

Demo: zwei Eltern (4 und 3 bekannte Zuordnungen), neue Zielrelation mit 6 Entitäten und 3 Klassen, 3 neue bekannte Zuordnungen und 5 unbekannte Testfälle einschließlich zwei außerhalb des Bereichs. Summen-/Relationskandidat: 21 Parameter, 80 SGD-Schritte, Verlust 1,5 → ca. 2,13e-14, 8/8 Fälle korrekt. Tests prüfen DIRECT und TRAINED, separate Prozess-Inferenz nur mit Kandidatengewichten, falsches Orakel, inkompatiblen Bridge, Batch-Rollback bei Schlüsselkollision, Quellenänderung vor Compute/Run/Select, Lock-Ausfall, explizite historische Offline-Policy und Lehrereingangsfehler. Bei Korrektur einer Elternquelle werden Elternexperte und abgeleiteter Experte quarantiniert, der zweite Elternexperte bleibt ACTIVE. Keine Gewichte gelöscht. Details: `evidence/v08-relations-demo.json`, `v08-relations-summary.json`, Testlog und JUnit.

## 5. Grenzen und nächste Schritte

Neue Kombinationen bereits bekannter Fakten sind nachgewiesen; keine Generalisierung auf unbekannte Relationen ohne explizite Kompositionsregel. Exhaustive Fixture-Prüfung ist nicht unabhängige externe Replikation oder semantischer Holdout. Noch keine freie Methodenbildung, neue kognitive Strukturen, Autonomie oder Leistungszuwächse auf realen Aufgaben. Keine physische GPU-Residency-Messung, Windows, echte Lehrmodelle oder Remote-CI. Historischer Chat/API-Pfad ist noch nicht migriert.

P0: gemeinsame Source-/Memory-/Population-Transaktionsgrenze mit Recovery, reale lokale Lehrmodell-Anbindung anhand einer unabhängig belegbaren Domäne, Kosten-/Retentionsvergleich gegen Ledger-only und Komposition ohne neue Gewichte, skalierbare Abhängigkeitsprüfung. Danach modellinitiierte Lernaufträge und neue deklarative Regeln. Additive Summenexperten und unabhängige Selektion weiterhin getrennt; keine automatische Ablösung oder Löschung.
