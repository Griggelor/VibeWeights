> Historische Referenz bis 0.2.0. Aktuelle Plattformverträge: [platform_v03.md](platform_v03.md).

# Plattform zuerst – Entwicklungsentscheidung 0.2.0

Ziel bleibt ~512 separat erzeugte Experten mit je ~100M Parametern, direkte Gewichtskonstruktion, numerischer Austausch, Rekursion und selbstverwalteter Index-Kontext. Zunächst werden die stabilen, expert-unabhängigen Schnittstellen gebaut; kein monolithisches LLM als versteckter Lösungsweg. Die fünf Referenzexperten bleiben Regressionstest und dienen als vertikaler Nachweis.

## Feste Module und Abnahme

| Modul | Vertrag / Abnahmekriterium | Stand 0.2.0 |
|---|---|---|
| Language input | Text -> typisierter TaskFrame; unsupported/ambiguous statt erfundener Semantik; austauschbare Encoder | Request/Protocol + begrenzter symbolischer Deutsch-Adapter; TaskFrame noch task-spezifisch |
| Language output | ausschließlich freigegebener Ergebniszustand -> Text; keine eigene Lösungsberechnung; Fehler nicht verschleiern | Protocol + Referenz-Verbalizer; Result enthält mehr als finalen Zustand, später reduzieren |
| Latent ABI | versionierte numerische Felder, Masken, Delta, namespaces, Typ-/Shape-/Owner-Prüfung | v0.1 auf Arithmetik begrenzt |
| Expert registry | manifestgesteuerte Entdeckung, Versions-/Hashprüfung, Capabilities, atomarer Austausch ohne Core-Änderung | feste SPECS/Loader; generalisieren |
| Controller | generischer Dispatcher, numerische Action-Requests, Berechtigungen, Budget, Verifikation, Fail-closed | fest verdrahteter Referenzpfad |
| Loop engine | intra/inter-expert, harte Grenzen, Fortschritt, Stagnation, definierte Fehler | begrenzter arithmetischer Loop |
| Memory | persistent, versioniert, indexiert; Search/Read/Write/Update/Evict/Compress, Quellenbindung, Budget | Exact-key Search/Read/Evict; externe Writes; rest offen |
| Verification | separater Verifikator, Fehlerstatus verhindert Ausgabe, nachvollziehbare Belege | noch kein allgemeiner Verifier; Ablation demonstriert Lücke |
| Backend/cache | CPU Referenz, austauschbare Geräte, Ressourcenlimits, später Strix Halo, Quantisierung und Cache | nur CPU |
| App/session | CLI, dann lokale API/Chatoberfläche; Session- und Persistenzgrenzen | CLI vorhanden, API/UI offen |
| CI | echte installierte Runtime, Gewichtsreproduktion, unabhängige Orakel, Regressionen auf Windows/Linux | Workflow vorbereitet; remote nicht ausgeführt |

## Reihenfolge
1. Privates, initialisiertes GitHub-Repository VibeWeights; Zugriff gewähren, vollständige Basis importieren, CI real ausführen und Fehler beheben. Kein öffentlicher Upload ohne Entscheidung. Kleine Tensorartefakte in Git, große später über getrennten Artefaktspeicher. Workflow-Berechtigungen gesondert beachten.
2. TaskFrame/ABI und Registry/Manifest abstrahieren; Routing, Loop und Action-Vertrag gemeinsam stabilisieren. Referenzexperten müssen weiterhin funktionieren und neue Testexperten ohne Core-Patch integrierbar sein.
3. Speicherverwaltung, Verifikation, Backend/Ressourcenmanagement und Sitzungen integrieren. Fehlerpfade und Limits im öffentlichen Entry Point prüfen. Sprachadapter gegen neue TaskFrames erweitern.
4. Sprachmodule separat erforschen: DIRECT für kontrollierte semantische Aufgaben, ggf. SYNTHESIS; breites Sprachverstehen und generative Ausgabe ohne Training sind offen. PRETRAINED-Brücke nur nach expliziter Entscheidung, getrennt vom reinen Nachweis. Keine externe LLM-Lösungsberechnung.
5. Plattform-Gate: reproduzierbarer E2E-Durchlauf Text -> ABI -> dynamischer Expertenpfad -> Memory/Loops -> Verifikation -> Ausgabe; neuer Experte per Manifest, Austausch/Fehlerinjektion, Windows/Linux CI grün; Strix Halo als separater Test. Kein Gate behauptet, allgemeine Sprachintelligenz nachgewiesen zu haben.
6. Danach Kompetenzkatalog schrittweise ausbauen, zunächst 8/32/128/512 als Kontrollpunkte, nur funktional begründete Parameterzahlen.

## Herkunft
Sprachadapter v0.2.0: SYMBOLIC, keine zusätzlichen Gewichte. Kernel weiterhin 43 DIRECT-Parameter. Der Input parst drei festgelegte Satzformen; Output gibt nur Ergebnisfelder wieder. Unbekannte Formulierungen werden explizit abgewiesen. Keine Mikrofon-/Audio-Schnittstelle; Sprache bedeutet hier Text. Der Nutzer hat noch keine Freigabe für vortrainierte Sprachkomponenten erteilt.
