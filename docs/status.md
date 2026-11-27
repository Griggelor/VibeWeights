# Status v0.7.0

**Ziel 1 (Environment) in Arbeit; keine Environment-Abnahme.** 104 lokale CPU-Tests bestanden (87 Regressionen + 17 neue Testfälle). Siehe [Memory Architecture v0.7](MEMORY_ARCHITECTURE_v07.md), [Implementierung v0.6](platform_v06.md), [Zielarchitektur](ARCHITEKTUR_v06.md) und [Review v0.5](review/v0.5.0/REVIEW_UND_ROADMAP_v06.md). Der reguläre Chat/API-Pfad ist weiterhin die korrigierte arithmetische Referenz; der neue Environment-Pfad ist opt-in.

## Fortschritt

- Persistentes numerisches Wissensledger mit unveränderlichen Revisionen, atomarem Head-Wechsel, Zeit-/Quellen-/Topic-/Relationsindex, Sichtbarkeitsregeln und expliziter Embeddingraum-Suche.
- Eigene Arbeitskontexte für Experten, logische Gruppenbereiche, native SEARCH/READ/RELATED/EVICT/REMEMBER-Aktionen und Ressourcenlimits.
- Nachweis der Wiederverwendung einer neu gespeicherten Erkenntnis durch einen anderen direkt konstruierten Rechenexperten nach Neustart der DB mit 1.000 zusätzlichen Ablenkungsdatensätzen.
- Kontextübergabe an Bootstrap-Textadapter über explizit ausgewählte, verifizierte IDs. Kein echtes LLM oder autonomer Lehrbetrieb.

## Nächste Arbeitspakete — Knowledge-first-Priorisierung

Verbindliche Ergänzung: [Knowledge-first Learning und Validierungsvertrag](KNOWLEDGE_FIRST_LEARNING.md). Konkretes Wissen und stabile Methoden vor autonomer Veränderung von Intelligenzstrukturen. Die fünf Ziele bleiben bestehen; erster vertikaler Gewichtslern-Nachweis hat P0-Priorität innerhalb des Environment-Ausbaus.

1. P0: GitHub-Baseline ist importiert; tatsächliche Windows/Linux-CI-Ergebnisse prüfen.
2. P0: gemeinsamer Environment-/Ledger-/Factory-Vertrag, neuer regulärer Chat/API-Pfad, sichere Migration, Events und unabhängige Verifikation. Keine zweite dauerhaft konkurrierende Produktionsruntime.
3. P0: Lernledger, Quellenrevisionen und transitive Abhängigkeiten, unveränderliche Kandidaten, getrennte Evaluationsrechte, atomare Freigabe und Recovery. Falsches, widerrufenes und privates Wissen nicht automatisch einlernen.
4. P0: realer lokaler Bootstrap- und Lehreradapter parallel; kontrollierte Lehrer-Fixture zuerst, echte Modellabnahme getrennt. Kein zentraler Planer.
5. P0: erster vertikaler parametrischer Wissenserwerb mit kleinen DIRECT-Assoziationsgewichten und separatem TRAINED-Experiment; Neustart, Lehrer/Bootstrap/Faktenabruf aus, Kandidatenablation, Korrektur, Vergessen und Vergleich gegen Ledger-only. Kein Generalisierungsanspruch durch bloße Wiederholung gelernter Fakten.
6. P1: sequenzielle Wissenspakete, Replay, additive Konsolidierung (Summenexperten ohne Ablösung der Eltern), sichere Feedbackreflexion, modellinitiierte Lernjobs, Methodenübertragung und effizientere Lernverfahren. Separate evolutionäre Selektion nach langfristigen, kontextbezogenen Kennzahlen, Abhängigkeiten und Populationsablationen; reversibler Ruhestand und dauerhafte SSD-Archivierung mit Suchkatalog, vollständigen Abhängigkeiten und bedarfsgerechter Reaktivierung. Keine physische Löschung im normalen evolutionären Betrieb; Residency und Lebenszyklus getrennt. SSD-Budget begrenzt neue Lernjobs, nicht den Erhalt alter Experten.
7. P1: freie dynamische Topologie, private Rekursion, semantisches ABI/Encoder, universelles Wissens-/Quellenformat, Index- und Konfliktverwaltung, integrierte Kooperation/Feedback/Episoden, tatsächliche Byte-/Zeitgrenzen. Gruppenrechte dynamisch aber überprüfbar; latent adapter experiments.
8. P2: Hardware/Studio parallel nach stabilen Verträgen: Q4-Tests, GGUF, C++-CPU-Parität, Vulkan/HIP auf Strix Halo, Lazy Cache und Peak-Budget, Windows Portable. Data und Models getrennt.
9. Environment-Abnahme und systematischer Ausbau der Basisintelligenz, Assimilation, Recherche und Neugier. Vergleich gegen eingefrorene Referenz und deaktivierte Bootstrap-Modelle; Verbesserung der Lernfähigkeit erst separat nachweisen.


Keine allgemeine Sprachintelligenz, keine autonome Expertensynthese, keine Qwen-Inferenz, kein GPU-/Windows-/GitHub-Lauf in dieser Version. Keine klassischen Trainingsschritte. Historische Nachweise unverändert archiviert. Siehe `evidence/summary.json`.
