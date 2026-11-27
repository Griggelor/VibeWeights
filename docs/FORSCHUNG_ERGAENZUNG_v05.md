# Forschungsergänzung v0.5 — Lernen, Exploration und Evolution

26.11.2026 · Ergänzung zur [Literaturübersicht](FORSCHUNG_2026-11-26.md). Gezielte Prüfung von bibliografischen Angaben und Abstracts/Projektseiten; keine systematische Review, keine Replikation, keine Patentrecherche. Die früheren, insbesondere sehr jungen WSP-Arbeiten sind als Forschungsreferenzen und nicht als verifizierte Leistungsversprechen zu lesen.

| Referenz | Bezug zu VibeWeights | Abgrenzung und konkrete Konsequenz |
|---|---|---|
| [Goyal et al., Coordination Among Neural Modules Through a Shared Global Workspace, 2021/22](https://arxiv.org/abs/2103.01197) | Bandbreitenbegrenzter geteilter Workspace für Spezialisten | Lokale und globale Arbeitsräume, messen statt unbegrenzt broadcasten; gelernte Module sind kein DIRECT-Nachweis |
| [Franklin et al., A LIDA cognitive model tutorial, 2016](https://www.sciencedirect.com/science/article/pii/S2212683X16300196), [Embodied Intelligence in LIDA, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9019476/) | Codelets, Koalitionen und globaler Broadcast | Kognitive Analogie, keine unabhängigen 100M-Gewichtsmodule; Gruppenbildung als Verhalten statt starre Struktur |
| [Hao et al., Coconut, 2024/25](https://arxiv.org/abs/2412.06769) | Verarbeitung kontinuierlicher latenter Zustände | Denkschritte nicht zwingend versprachlichen; trainierte LLM-Architektur, kein Beleg für untrainierte Expertensprache |
| [Li et al., A Contextual-Bandit Approach, 2010](https://arxiv.org/abs/1003.0146) | Kontextbezogene Auswahl und Lernen aus Feedback | Inspiration für lokale Rekrutierung und Exploration; kein zentraler Bandit-Router; Exposition und Auswahlwahrscheinlichkeit protokollieren |
| [Li et al., Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms, 2010](https://arxiv.org/abs/1003.5956) | Auswertung von geloggten Auswahlentscheidungen | Randomisierte Kontrollläufe, Selektionsbias nicht mit Kompetenz verwechseln |
| [Foerster et al., Counterfactual Multi-Agent Policy Gradients, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794) | Zurechnung von Gruppenresultaten zu einzelnen Beiträgen | Kontrafaktische Ablationen / gepaarte Varianten; kein Anspruch, COMA oder RL bereits implementiert zu haben |
| [Jaderberg et al., Population Based Training, 2017](https://arxiv.org/abs/1711.09846) | Population, Auswahl und Variation | Abstammung und isolierte Kandidaten; PBT ist ein Trainingsverfahren, nicht direkte Konstruktion oder unsere dezentrale Selektion |
| [Hinton, Vinyals & Dean, Distilling the Knowledge in a Neural Network, 2015](https://arxiv.org/abs/1503.02531) | Übertragung von Verhalten großer Modelle auf kleinere | Optionale explizite Destillation mit erlaubten Lehrern; nicht mit direkter Gewichtsprogrammierung gleichsetzen |

## Experimentelle Konsequenzen

1. **Exploration:** gepaarte Aufgaben mit gleicher Ausgangssituation und Seed, mit und ohne Shots. Qualitätsdifferenz und Rechenmehrkosten erfassen. Bei v0.5 sind die Shots noch Shadow-Probes, daher kann daraus keine nachgewiesene neue Kompetenz entstehen.
2. **Attribution:** Gruppenresultate und Nutzerfeedback sind Assoziationen. Erst Interventions-/Ablationsexperimente mit kontrollierter Exposition geben Hinweise auf kausalen Beitrag. Korrektheit, Zufriedenheit und Ressourcenverbrauch getrennt halten.
3. **Varianten:** V1/V2 auf vergleichbaren Aufgaben und unter gleichen Budgets testen. Kontextabhängige Koexistenz zulassen. Seltene Fähigkeiten nicht aufgrund geringer Nachfrage löschen. Holdouts, Mindeststichproben und Unsicherheitsintervalle für spätere Selektionsentscheidungen definieren.
4. **Assimilation:** Lehrerzugriff abschalten und unbekannte Aufgaben prüfen. Wissensspeicherung, symbolische Methoden, analytische Gewichte und trainierte Gewichte klar unterscheiden. Neue Fähigkeiten nicht aus bloß reproduzierten Frage-Antwort-Paaren ableiten.
5. **Bootstrap:** prüfen, ob das System nach Erwerb einer Kompetenz auch seinen Lernprozess verbessert. Keine Annahme unbegrenzten rekursiven Fortschritts.

Die Kombination der Architekturprinzipien ist eine eigenständige Projekthypothese. Eine belastbare weltweite Neuheitsaussage oder ein Nachweis allgemeiner Intelligenz ist nicht möglich.
