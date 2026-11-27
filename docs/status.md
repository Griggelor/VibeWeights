# Status v0.7.0

**Ziel 1 (Environment) in Arbeit; keine Environment-Abnahme.** 104 lokale CPU-Tests bestanden (87 Regressionen + 17 neue Testfälle). Siehe [Memory Architecture v0.7](MEMORY_ARCHITECTURE_v07.md), [Implementierung v0.6](platform_v06.md), [Zielarchitektur](ARCHITEKTUR_v06.md) und [Review v0.5](review/v0.5.0/REVIEW_UND_ROADMAP_v06.md). Der reguläre Chat/API-Pfad ist weiterhin die korrigierte arithmetische Referenz; der neue Environment-Pfad ist opt-in.

## Fortschritt

- Persistentes numerisches Wissensledger mit unveränderlichen Revisionen, atomarem Head-Wechsel, Zeit-/Quellen-/Topic-/Relationsindex, Sichtbarkeitsregeln und expliziter Embeddingraum-Suche.
- Eigene Arbeitskontexte für Experten, logische Gruppenbereiche, native SEARCH/READ/RELATED/EVICT/REMEMBER-Aktionen und Ressourcenlimits.
- Nachweis der Wiederverwendung einer neu gespeicherten Erkenntnis durch einen anderen direkt konstruierten Rechenexperten nach Neustart der DB mit 1.000 zusätzlichen Ablenkungsdatensätzen.
- Kontextübergabe an Bootstrap-Textadapter über explizit ausgewählte, verifizierte IDs. Kein echtes LLM oder autonomer Lehrbetrieb.

## Nächste Arbeitspakete in Reihenfolge

1. GitHub-Import der aktuellen Baseline, Windows/Linux-CI tatsächlich ausführen. Öffentliches Repository und Apache-2.0 beschlossen; vollständiger Import und CI noch offen.
2. Neuer Environment-Kern als regulärer Chat/API-Pfad mit Migration: gemeinsamer Kontext- und Sessionvertrag, dynamische Topologie, Eventbus, eigene rekurrente Zustände, unabhängige Verifikation, alte Review-Solltests. Keine zweite dauerhaft konkurrierende Produktionsruntime.
3. Memory: semantisches ABI/Encoder, universelles Wissens-/Quellenformat einschließlich Text/Dokumenten, Index- und Konfliktverwaltung, integrierte Kooperation/Feedback/Episoden, autonome Konsolidierung, transitive Provenienz, Recovery und tatsächliche Byte-/Zeitgrenzen. Gruppenrechte dynamisch aber überprüfbar.
4. Bootstrap: echte Qwen-MoE oder andere zulässige lokale Modelle, native Latentadapter erforschen; separate Text-/Latent-Evaluation und Bootstrap-Ablation. Kein zentraler Planer.
5. Expert Factory: weitere deklarative Operatorgraphen, Kandidatenvalidierung, Versionenvergleich, Feedbackreflexion, Lernjobs und Verdichtung; einen dauerhaften Kompetenzerwerb mit deaktiviertem Lehrer nachweisen.
6. Hardware/Studio parallel nach stabilen Verträgen: Q4-Tests, GGUF, C++-CPU-Parität, Vulkan/HIP auf Strix Halo, Lazy Cache und Peak-Budget, Windows Portable. Data und Models getrennt.
7. Environment-Abnahme, danach systematischer Ausbau der Basisintelligenz, Assimilation, Recherche und Neugier. Vergleich gegen eingefrorene Referenz und deaktivierte Bootstrap-Modelle.

Keine allgemeine Sprachintelligenz, keine autonome Expertensynthese, keine Qwen-Inferenz, kein GPU-/Windows-/GitHub-Lauf in dieser Version. Keine klassischen Trainingsschritte. Historische Nachweise unverändert archiviert. Siehe `evidence/summary.json`.
