# ADR / technische Spezifikation: Platform 0.3.0

## Entscheidungen

- Bestehende fünf Experten bleiben binär unverändert (ABI 0.1 Gewichtsmanifeste). ABI 0.2 ist eine neue, explizite schema-gebundene Hülle. Der Loader akzeptiert Legacy-Manifeste nur im passenden arithmetischen Schema; neue Manifeste benötigen `schema_id` und `schema_sha256`. Keine Behauptung über einen frei emergenten latenten Raum.
- Getrennte Schichten: Textadapter -> TaskFrame -> TaskRouter -> Plan -> Engine -> Registry/Backend -> Ergebnisprojektion -> Textadapter. Keine Textnachrichten zwischen Experten. Namen in Code/Trace sind Metadaten, Tensorinhalte numerisch.
- Kein allgemeines autonomes Routing behauptet. Ein TaskRouter ordnet registrierte Task-IDs zu; der Plan enthält symbolische Zustandsprädikate. Die Engine selbst ist unabhängig von der Rechenaufgabe. Genau eine ausführbare Regel erforderlich; null oder mehrere ergeben ROUTING. Eine Reihenfolge der Regeln ist fachlich irrelevant. Neue Aufgaben können neue Pläne benötigen, nicht zwangsläufig Änderungen an Engine oder Gewichten.
- Verifikation als eigener verpflichtender Planvertrag. Der symbolische arithmetische Prüfer rechnet unabhängig nach, schreibt aber keinerlei Lösungsdaten. Falsche, formal gültige Gewichtsresultate werden verworfen. Die symbolische Verifikation ist kein neuronaler Kompetenznachweis und kann bei komplexen Problemen nicht immer existieren.
- Sprache: direkt konstruierte dezimale Stellenwertprojektion (80 Gewichte + Bias) und binäre Relationsprojektion (2 Gewichte + 2 Bias). Grammatik, Tokenisierung, Formatierung und inverse lexikalische Prüfung sind symbolisch. Die Eingabegrenze prüft das Resultat durch Rückkodierung, die Ausgabegrenze prüft die Bedeutung des ausgewählten Wortes. Keine Korrektur fehlerhafter Gewichte; Fehler werden abgewiesen. Nur vier feste Satzmuster, keine freie Unterhaltung.
- Einzelprozess/CPU. Die explizite CPU-Backend-Schnittstelle erlaubt spätere Implementierungen, aktuell kein GPU-Backend. Registry limitiert Anzahl und Parameterzahl; Engine limitiert Dispatches und nominelle Parameterbesuche. Parameterbesuche zählen die Größe jedes aufgerufenen Experten einschließlich Bias pro Dispatch, nicht FLOPs, nicht Laufzeit und nicht reale Speicherbandbreite. Sprach- und Verifikationsaufwand ist darin nicht enthalten.

## ABI 0.2 / feste Verträge

`Schema(id,fields,version)` ist unveränderlich. Jede `Field(name,min,max)` definiert einen exakten ganzzahligen Skalar innerhalb [-2^24,2^24]. `Frame` speichert ein Tupel von Werten oder None; fehlend unterscheidet sich von numerisch null. Geordnete Float32-Snapshots oder Gültigkeitsmasken gehen an die Gewichtsmodelle. Schema-Fingerprint bindet ID, ABI-Version, Reihenfolge, Namen und Wertebereiche. Generische Vektoren/Matrizen, probabilistische Werte, Entitäten und freies semantisches Grounding bleiben offen.

`TaskFrame(task_id, inputs)` trägt nur strukturierte numerische Daten. Plan definiert Eingabefelder und erlaubte exakte Varianten, Ergebnisfelder, unabhängige Portanforderungen, Routingregeln, Halting und Verifikator. Initial vorhandene Eingabewerte bleiben unverändert. Registry prüft Identität, Version, Herkunft, ABI, Hash, Parameteranzahl und Tensorformen. Nur `affine_relu` unterstützt; weitere Architekturen benötigen Backend-/Factory-Erweiterungen. Die Registry einer Schemafamilie lädt alle Manifeste aus einem gemeinsamen Root; andere Schemafamilien benötigen derzeit separate Roots/Registries. TaskRouter kann unterschiedliche registrierte Pläne/Registries adressieren, hat aber noch keinen globalen Expertenspeicher.

`Engine.run(plan, inputs, budget, work_budget) -> Execution`:
1. Verträge prüfen, initialen Frame erzeugen; keine Eingaben außerhalb der erlaubten Variante.
2. Registry-Snapshot (Referenzen, nicht tiefe Gewichtskopie); ausführbare Regel eindeutig wählen.
3. Budgets prüfen; Experten-Forward über Backend. Ausgabe als strikt integrale, endliche Float32-Daten in der vereinbarten Form.
4. Direktes Delta oder genau ein numerischer Action-Code; Action muss in der aktuellen Regel erlaubt sein. Core prüft und übernimmt nur erlaubte Felder. Memory-Handler sind vertrauenswürdiger Python-Code, keine fremd ausführbaren Manifeste.
5. Initiale Eingaben, optionaler Fortschritt, Wiederholungszustand und harte Grenzen prüfen. Wiederholte Zustände führen zu STAGNATION; kein allgemeiner zyklischer Algorithmus mit unverändertem Zustand möglich.
6. Halting erst mit vollständigen Ergebnisfeldern und positivem Verifikator; Fehlerstatus schließt erfolgreiche Ausgabe aus.

Status: OK, BUDGET, STAGNATION, INVALID, MEMORY, ROUTING, VERIFICATION. `Execution.frame` bleibt auch bei Fehlern zur Diagnose verfügbar; `FinalView` gibt bei Fehlern keine Nutzwerte an den Decoder. Die App liefert `verified=true` erst nach erfolgreicher Kernprüfung und Sprachgrenzen.

## Referenzpläne

`bounded.multiply_compare`: Eingabe X/Y/THRESHOLD oder X/KEY/THRESHOLD. Legacy-Arithmetikfelder, zusätzlich READY/HALTED. Indexpfad: numerischer Action-Head SEARCH, READ, READY, dann wiederholter arith.step; Vergleich, EVICT, HALT. Exakter Quellen-Record wird fixiert; Verifikator liest ihn erneut. Keine semantische Retrieval-Behauptung. `bounded.compare`: nutzt direkt den Legacy-Skalarport PRODUCT als linke Vergleichsgröße, benötigt nur `logic.ge`. Beide Pläne benutzen unveränderte Kompetenzgewichte. Ein drittes ABI mit neuem Testexperten ist ausschließlich Testfixture, kein neuer produktiver Experte.

## Speicher- und Betriebsgrenzen

SQLite append-only Versionen, exakter Schlüsselindex, stabiler Record-ID-Referenz, CAS-Update und Provenienz. Native SEARCH/READ/EVICT werden im Plan durch Gewichtsscores ausgelöst und validiert. WRITE/UPDATE werden bislang explizit durch Anwendung/Host aufgerufen. Generische ActionHandler sind kein transaktionaler Seiteneffektmanager: vor autonomen Schreibaktionen muss eine commit-/rollback-fähige Grenze implementiert werden. Mehrstufige Relevanzsuche, Konfliktlogik, LINK, COMPRESS, persistente Working-Memory-Strategie offen.

Sitzungsjournal hält Textanfragen und Antworten getrennt nach Session-ID. Kein Gesprächsgedächtnis, keine automatische Injektion von Historie. Gemeinsamer Memory-Store, keine Nutzerisolation. API ist nur für vertrauenswürdige lokale Clients, Single-Thread-Referenz mit begrenzter Requestgröße und Socket-Timeout. Kein produktionsgehärteter Server. Keine parallele Registry-Mutation, keine Echtzeit-/Wallclock-Garantie, kein Sandboxen von Python-Plugins.

## Plattform-Gate (noch nicht erreicht)

1. GitHub-Import und tatsächlich grüne Windows-/Linux-CI.
2. Native transaktionale Schreibaktionen, echte Working-Memory-Policy und Such-/Relevanzgenerierung.
3. Erweiterbare ABI-Tensorfelder und größere/verschiedenartige Backends, globaler Registry-/Ressourcenmanager mit Laden/Entladen, Trace- und Laufzeitbudgets.
4. Zunehmend gewichtsbasierter Router und sprachliche Semantik jenseits begrenzter Satzformen; Alternativen separat kennzeichnen. Keine vortrainierte Brücke ohne Freigabe.
5. Verifikation für mehrere Aufgabenklassen, sichere API/Chatoberfläche; Windows/Strix-Halo-Echtlauf. Erst danach systematischer Kompetenzkatalog-Ausbau.
