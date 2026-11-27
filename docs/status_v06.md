# Status v0.6.0

Ziel 1 (Environment) in Arbeit. **Keine Environment-Abnahme.** v0.6 ist ein erster additiver Slice, die Standardanwendung ist noch die korrigierte v0.5-Referenz. Neue technische Beschreibung: [platform_v06.md](platform_v06.md). Aktualisierte Zielarchitektur: [ARCHITEKTUR_v06.md](ARCHITEKTUR_v06.md). Der [Review v0.5](review/v0.5.0/REVIEW_UND_ROADMAP_v06.md) bleibt die Baseline.

## Nächste Arbeitspakete

1. GitHub: private Baseline importieren, tatsächliche Windows/Linux-CI, Release-Gates. Lizenzentscheidung offen.
2. Neuer Environment-Kern: Eventbus mit Abonnements, echte private rekurrente Zustände, freie Scope-/Teilziel-/Gruppendynamik, Revisionen, unabhängige Verifikation. Neue Runtime in regulären Chat integrieren; alte Reviewfälle als Solltests schließen. Persistenz und Recovery, Prozessisolation und technische Budgetierung.
3. Bootstrap: lokal ein Qwen-MoE (oder anderes zulässiges Modell) real anbinden, heterogene Adapter, explizit getrennte Text-/Latent-Tests und baseline/ablation. Kein verdeckter zentraler Planer. LLM-Vorschläge zunächst unverified; native Freigabe und unabhängige Tests. 
4. Factory: deklarativer Operatorgraph, weitere analytische Konstruktoren, sicherer Compiler, realer Variantenvergleich und Rollback; kausale Leistungszurechnung, Feedbackreflexion, Knowledge Ledger, Lernjobs. Begrenzten Kompetenzerwerb mit Lehrer und anschließend Lehrer-offline nachweisen. Verdichtung über affine Funktionen hinaus prüfen.
5. Backend/Studio parallel nach stabilen Verträgen: Q4-Verhaltenstests, GGUF-/SafeTensors-Reader, C++-CPU-Parität, Vulkan/HIP auf Strix Halo, Resident-Cache und Peak-Byte-Budget, Windows-Portable-Release. Separates Data-Verzeichnis und Migration.
6. Environment-Abnahme; danach systematischer Ausbau der Basisintelligenz, Assimilation, Recherche und Neugier. Vergleich gegen eingefrorene Referenz und deaktivierte Bootstrap-Modelle.

Die Ziele 2–5 sind weiterhin Forschungsziele; echte selbst erworbene Kompetenzen wurden nicht nachgewiesen. Keine klassische Trainingspipeline ausgeführt, keine Qwen-Gewichte mitgeliefert.
