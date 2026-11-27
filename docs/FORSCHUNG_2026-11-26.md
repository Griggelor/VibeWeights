# VibeWeights — Forschungsstand und Architekturfolgen
26.11.2026 · gezielte Literaturrecherche; keine systematische Vollerhebung oder Replikation. Primär Abstracts, Projektseiten und ausgewählte HTML-Textstellen geprüft. Neuere arXiv-Arbeiten sind als Preprints zu behandeln. Keine Behauptung weltweiter Neuheit.

| Arbeit | Befund / Grenze | Konsequenz für VibeWeights |
|---|---|---|
| [RIMs, ICLR 2021](https://mlanthology.org/iclr/2021/goyal2021iclr-recurrent/) | Gelernte, separat rekurrente Mechanismen, sparse Aufmerksamkeit und Aktivierung; keine frei wachsenden Teams | Private Zustände und selektive Kommunikation |
| [Shared Global Workspace, ICLR 2022](https://arxiv.org/abs/2103.01197) | Spezialmodule koordinieren sich über gemeinsamen begrenzten Workspace | Lokale Gruppen-Workspaces plus globaler Ergebniskanal |
| [LIDA tutorial, 2016](https://www.sciencedirect.com/science/article/pii/S2212683X16300196), [Codelets und Koalitionen](https://pmc.ncbi.nlm.nih.gov/articles/PMC9019476/) | Kognitive Architektur mit Koalitionen, unterschiedlichen Gedächtnissen; hybride Mechanismen | Selbstorganisation als Inspiration, kein WSP-Nachweis |
| [Transactive Memory Review](https://onlinelibrary.wiley.com/doi/full/10.1111/tops.12359) | Gruppenwissen über die Kompetenz anderer Personen | Individuelles, evidenzbasiertes Kooperationsgedächtnis |
| [Contract Net, 1980](https://www.scilit.com/publications/51c2fb8fa05bbe0abed48d2b35d3e689) | Dezentrale Aufgabenverhandlung, kein neuronales System | Numerische Bedarfsmeldung/Angebot/Einladung, keine permanente Hierarchie |
| [Neural Production Systems, 2021](https://papers.nips.cc/paper/2021/hash/d785bf9067f8af9e078b93cf26de2b54-Abstract.html) | Regeln binden an Entitäten, selektive Aktualisierung | Typisierte Entitäten, Relationen, Kompetenzen |
| [Expert Choice, 2022](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/) | Experten wählen Token; weiterhin MoE-Zuweisung und Kapazitätsgrenzen | Selbstaktivierung als Analogie, keine Übernahme des festen Kapazitätsschemas |
| [DyLAN, 2023](https://arxiv.org/abs/2310.02170) | Teamoptimierung und dynamische LLM-Kollaboration; keine autonome neuronale Koalition | Vergleich mit statischen Teams, Beitrag messen |
| [MemGPT, 2023](https://arxiv.org/abs/2310.08560) | Virtuelles Kontextmanagement mit mehreren Speicherebenen | Working Memory, Index, explizite Memory-Actions |
| [Coconut, 2024](https://arxiv.org/abs/2412.06769) | Kontinuierliche Hidden States statt Sprach-Zwischentoken; trainiertes LLM | Denken und Ausgabe entkoppeln |
| [FlexOLMo, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/f2621bfa8b39cd3c5a361273aa7ec4db-Abstract-Conference.html) | Separat trainierte Experten kombinierbar; kein direktes Erzeugen aus dem Nichts | ABI/Kompatibilität und Austauschtests |
| [Huginn, 2025](https://arxiv.org/abs/2502.05171) | Rekurrente Tiefe in trainiertem Modell | Wiederverwendete Gewichte und Testzeit-Budget |
| [Mixture of Thoughts, 2025](https://arxiv.org/abs/2509.21164) | Frozen Experten, gelernter Router und Interaktionsschichten | Alignment ist zentral; Router nicht übernehmen |
| [Thought Communication, 2025](https://arxiv.org/abs/2510.20733) | Gemeinsame/private latente Repräsentationen unter Modellannahmen | Privaten und gemeinsamen Zustand trennen |
| [LatentMAS, ICML 2026](https://arxiv.org/abs/2511.20639) | Trainingsfreie Kollaboration bereits trainierter LLMs | Latente Kommunikation unabhängig testen; nicht mit trainingsfreien Gewichten verwechseln |
| [Sparse Looped Models, 2026](https://arxiv.org/abs/2605.09165) | Looped MoE und Early Exit | Sparse Aktivierung, unterschiedliche Tiefe messen |
| [Latent Communication Between Language Model Agents, 2026](https://arxiv.org/abs/2607.14103) | Alignment und komprimierte latente Kanäle | ABI und Bandbreite überprüfen |
| [Stabilizing Recurrent Dynamics, 2026](https://arxiv.org/abs/2605.26733) | Mehr Loops können Leistung nach optimaler Tiefe verschlechtern | Nicht nur Konvergenz, sondern Fortschritt und Qualität prüfen |
| [Dynamics of specialization, 2025](https://www.nature.com/articles/s41467-024-55188-9) | Strukturelle Modularität garantiert keine funktionale Spezialisierung | Spezifische Kompetenztests, Interventions- und Ablationstests |

## Weight-Space Programming (WSP)
- [Sparks of WSP](https://rikiyat.github.io/blog/weight-space-programming/): Ausgangspunkt für LLM-generierte Gewichtskonstruktion/-änderung; kein Beleg für allgemeine 100M-Experten. Die alternative Research-Library-Seite konnte nicht vollständig geladen werden.
- [Interpretable WSP, 13.11.2026](https://arxiv.org/abs/2611.18114): weiterführende Arbeit zu transparenten neuronalen Modellen. Als Kandidat für vertiefte Prüfung, nicht als validierter Baustein.
- [WSP Doesn't Need Transformers, 17.11.2026](https://arxiv.org/abs/2611.22055), [Projektseite](https://wsp-no-transformer.github.io/): Autoren berichten stark kompaktere, direkt geschriebene Modelle bei begrenzten Wissensaufgaben gegenüber einem ebenfalls direkt geschriebenen Transformer. Kein Vergleich mit gleichwertigen allgemein kompetenten LLMs. Assoziative Speicher und strukturierte Gewichte als Kandidaten; pro Kompetenz die Architektur frei wählen.
- [WeightCode, 19.11.2026](https://arxiv.org/abs/2611.25476): neue Fähigkeiten mittels ausführbarer Gewichtsprogramme in bereits trainierte Netze eingebracht. Autoren berichten u.a. hohe Erfolgsraten bei beschränkten Kompositions-/Navigationsaufgaben. Wichtig: vorhandene trainierte Repräsentationen; nicht DIRECT-from-scratch.
- [Forward Weight-Space Programming, 25.11.2026](https://arxiv.org/abs/2611.34612): deterministische Probing-/Editierpipeline für strukturierte Gewichtsänderungen; Bericht über Grenzen. Für spätere kontrollierte Experimente, keine Grundlage für eine Behauptung allgemeiner Selbstverbesserung.

## Priorisierte Schlussfolgerungen
P0: ABI-Semantik vor Expertenmenge; standardisierter äußerer Vertrag, heterogener innerer Kern. P0: kein zentraler kognitiver Router, aber explizite technische Nachrichten- und Ressourcenebene. P0: Gruppen-Workspaces, individuelle Erinnerung und offene Discovery. P0: Evidenz, Konflikte und sichere Resultatfreigabe vor Mehrheitsabstimmung. P1: gewichtsgesteuerte Aktivierungs- und Kommunikationsköpfe als erster Beweis. P1: qualitative und quantitative Ablation von Erinnerung, Teams, Loops, Sprachschnittstellen. P2: WSP-Experimente mit assoziativen Speichern, expliziten Schaltungen und rekurrenten Blöcken, keine Festlegung auf Transformer. P2: Skalierung und Bandbreite auf Zielhardware messen, keine Effizienzversprechen ohne Messung.

## Einordnung
Die Kombination der ausgewählten Elemente ist eine eigene Projektarchitektur. Es wurde kein bereits bestehendes System gefunden, das alle Vorgaben erfüllt. Das ist keine Aussage, dass es ein solches System definitiv nicht gibt. Keine Ergebnisse wurden unabhängig repliziert, kein fremder Code eingebaut. Die nächste Version muss die Mechanismen anhand des Abnahmeplans der Architekturentscheidung prüfen.
