# Status v0.8.0.dev2

**Ziel 1 (Environment) in Arbeit; keine Environment-Abnahme.** 127 lokale CPU-Tests bestanden (115 Regressionen + 12 neue Testfälle). Siehe [Memory Architecture v0.7](MEMORY_ARCHITECTURE_v07.md), [Implementierung v0.6](platform_v06.md), [Zielarchitektur](ARCHITEKTUR_v06.md) und [Review v0.5](review/v0.5.0/REVIEW_UND_ROADMAP_v06.md). Der reguläre Chat/API-Pfad ist weiterhin die korrigierte arithmetische Referenz; der neue Environment-Pfad ist opt-in.

## Entwicklungsstand v0.8.0.dev2 (Wissensaufnahme)

Erster begrenzter Gewichtslern-Nachweis: DIRECT-Assoziationsmatrix und TRAINED-SGD-Experte, persistente Lernjobs, Quellenrevisionen und transitive deklarierte Eltern, hashgebundene Fallberichte, additiver Aufnahmeweg, explizite Quarantäne bei Quellenrevisionen, drei sequenzielle getrennte Wissensdomänen, Recall/Abstention nach Neustart ohne Wissens-DB und separate Gewichtedatei-Tests. Demo je 65 Parameter, 6/6 Fakten, 8/8 unbekannte Fälle, 14/14 nach Neustart. TRAINED: 80 echte Optimierungsschritte. Kein allgemeiner Generalisierungsnachweis, kein echtes Lehrmodell. Siehe [Implementierung](KNOWLEDGE_LEARNING_v08.md), [Demo](../evidence/v08-knowledge-learning-demo.json), [Testnachweis](../evidence/v08-knowledge-learning-summary.json). Quellenaudit noch explizite Host-Pflicht, keine automatische Cross-DB-Invalidation oder OS-Sandbox.

## Historischer Entwicklungsstand v0.8.0.dev1 (Population)

115 lokale CPU-Tests bestanden (104 Regressionen + 11 neue Testfälle). Siehe [Population/Residency](POPULATION_v08.md) und [aktuelle Testevidenz](../evidence/v08-population-summary.json). Implementiert: drei freigegebene Zustände, getrennte additive Aufnahme und Konsolidierung, explizite evidenzbasierte Selektionsschnittstelle, SQLite-Revisionsjournal, Budgetverweigerung ohne automatische Verdrängung, Archivierung/Reaktivierung und numerischer Environment-Anschluss. Nur affine FP32-Experten, ein vertrauenswürdiger Host-Owner, logisches Tensorbudget. Kein gemessener GPU-Residency-Nachweis, keine autonomen Selektionsberichte, keine echte Wissensaufnahme. Nächster P0: Learning Ledger + DIRECT/TRAINED-Wissensexperiment, Abnahme ohne Lehrer/Bootstrap/Faktenabruf, dann Langzeitevidenz für die Selektion.

## Fortschritt der historischen v0.7.0-Baseline

- Persistentes numerisches Wissensledger mit unveränderlichen Revisionen, atomarem Head-Wechsel, Zeit-/Quellen-/Topic-/Relationsindex, Sichtbarkeitsregeln und expliziter Embeddingraum-Suche.
- Eigene Arbeitskontexte für Experten, logische Gruppenbereiche, native SEARCH/READ/RELATED/EVICT/REMEMBER-Aktionen und Ressourcenlimits.
- Nachweis der Wiederverwendung einer neu gespeicherten Erkenntnis durch einen anderen direkt konstruierten Rechenexperten nach Neustart der DB mit 1.000 zusätzlichen Ablenkungsdatensätzen.
- Kontextübergabe an Bootstrap-Textadapter über explizit ausgewählte, verifizierte IDs. Kein echtes LLM oder autonomer Lehrbetrieb.

## Nächste Arbeitspakete — Knowledge-first-Priorisierung

Verbindliche Ergänzung: [Knowledge-first Learning und Validierungsvertrag](KNOWLEDGE_FIRST_LEARNING.md). Konkretes Wissen und stabile Methoden vor autonomer Veränderung von Intelligenzstrukturen. Die fünf Ziele bleiben bestehen; erster vertikaler Gewichtslern-Nachweis hat P0-Priorität innerhalb des Environment-Ausbaus.

1. P0: GitHub-Baseline ist importiert; tatsächliche Windows/Linux-CI-Ergebnisse prüfen.
2. P0: gemeinsamer Environment-/Ledger-/Factory-Vertrag, neuer regulärer Chat/API-Pfad, sichere Migration, Events und unabhängige Verifikation. Keine zweite dauerhaft konkurrierende Produktionsruntime.
3. P0: Lernledger und begrenzte Quellen-/Evaluationsprüfung implementiert. Nächster Schritt: automatische Quellenrevision-/Quarantänebindung vor Inferenz und Reaktivierung, vollständiger Recovery-Vertrag, getrennte Evaluationsrechte und echte Fehler-/Crash-Tests. Keine unkontrollierte Aufnahme falschen, widerrufenen oder privaten Wissens.
4. P0: realer lokaler Bootstrap- und Lehreradapter parallel; kontrollierte Lehrer-Fixture zuerst, echte Modellabnahme getrennt. Kein zentraler Planer.
5. P0: erster parametrischer Recall-Nachweis DIRECT/TRAINED, Neustart und Ablation umgesetzt. Ausbau: echte lokale Lehrerantworten mit unabhängig überprüfbaren Quellen, modellinitiierte Lernjobs, Regeln und neue ableitbare Kombinationen als Holdouts, Ledger-only-Vergleich samt Kosten, gemeinsame Langzeit-Retention. Kein Generalisierungsanspruch durch bloße Wiederholung gelernter Fakten.
6. P1: sequenzielle Wissenspakete, Replay, additive Konsolidierung (Summenexperten ohne Ablösung der Eltern), sichere Feedbackreflexion, modellinitiierte Lernjobs, Methodenübertragung und effizientere Lernverfahren. Separate evolutionäre Selektion nach langfristigen, kontextbezogenen Kennzahlen, Abhängigkeiten und Populationsablationen; genau drei reguläre Expertenzustände ACTIVE (VRAM), LOW_ACTIVE_IMPORTANT (VRAM) und ARCHIVED (SSD). Keine RAM-Zwischenstufe; wichtige seltene Kompetenzen bleiben resident. Dauerhafte SSD-Archivierung mit Suchkatalog, vollständigen Abhängigkeiten und bedarfsgerechter Reaktivierung; keine physische evolutionäre Löschung. SSD-Budget begrenzt neue Lernjobs, nicht den Erhalt alter Experten.
7. P1: freie dynamische Topologie, private Rekursion, semantisches ABI/Encoder, universelles Wissens-/Quellenformat, Index- und Konfliktverwaltung, integrierte Kooperation/Feedback/Episoden, tatsächliche Byte-/Zeitgrenzen. Gruppenrechte dynamisch aber überprüfbar; latent adapter experiments.
8. P2: Hardware/Studio parallel nach stabilen Verträgen: Q4-Tests, GGUF, C++-CPU-Parität, Vulkan/HIP auf Strix Halo, Lazy Cache und Peak-Budget, Windows Portable. Data und Models getrennt.
9. Environment-Abnahme und systematischer Ausbau der Basisintelligenz, Assimilation, Recherche und Neugier. Vergleich gegen eingefrorene Referenz und deaktivierte Bootstrap-Modelle; Verbesserung der Lernfähigkeit erst separat nachweisen.


Keine allgemeine Sprachintelligenz, keine autonome Expertensynthese, keine Qwen-Inferenz, kein GPU-/Windows-/Remote-CI-Nachweis aus diesem Lauf. Echte SGD-Optimierung in der künstlichen Wissensdomäne. Historische Nachweise unverändert archiviert; `evidence/summary.json` bleibt v0.7-Baseline.
