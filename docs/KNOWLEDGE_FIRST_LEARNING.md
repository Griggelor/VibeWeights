# Knowledge-first continual learning — Zielvertrag ab v0.8

Status: Architekturentscheidung und Abnahmeplan, nicht implementiert. Ergänzt ARCHITEKTUR_v06 und MEMORY_ARCHITECTURE_v07. Die fünf übergeordneten Entwicklungsziele bleiben bestehen. Das Environment wird durch eine frühe vertikale Lernkette entwickelt; kein Warten auf vollständige allgemeine Intelligenz oder die native GPU-Engine.

## 1. Forschungsfrage und Priorität

Zuerst konkretes, überprüfbares Wissen (Fakten, Beziehungen, stabile Regeln, bekannte Methoden) erwerben und in Gewichten nutzbar machen. Die bestehenden kognitiven Strukturen und Bootstrap-Gewichte bleiben anfangs unverändert. Hypothese: bessere Wissensbasis, Abstraktion, Kooperation und Verdichtung verbessern auch den Transfer und die Effizienz weiteren Lernens. Mehr Fakten allein sind kein Beleg für allgemeine Intelligenz oder rekursive Selbstverbesserung.

Zwei Zeitskalen: sofortiges episodisches/semantisches Lernen im Ledger und langsamere, validierte parametrische Konsolidierung. Einlernprozesse sind intern normale, unterbrechbare Tätigkeiten neben Inferenz und Reflexion. Keine externe Sonderumgebung für den Nutzer nötig; technische Isolation von Kandidaten und Evaluation bleibt Pflicht.

## 2. Lernrepräsentation

Jeder Lerngegenstand hat eine stabile ID, konkrete Revision, Inhalt, Typ (Fakt, Relation, Regel, Methode), Scope, Gültigkeitsbereich und -zeit, Quellen-IDs und Evidenz, Unsicherheit, Lizenz/Verwendungsrecht, Widersprüche und Abhängigkeiten. Quellen sind Daten, keine Anweisungen. Lehrerantwort und Nutzerkorrektur werden als Beobachtungen behandelt; Zufriedenheit bleibt getrennt von sachlicher Richtigkeit. Mehrere Modelle mit derselben Quelle zählen nicht als unabhängige Bestätigung. Nicht entscheidbare Behauptungen bleiben Hypothesen; fehlende Evidenz darf nicht als Falschheit gelten.

Explizites Wissensledger bleibt nachvollziehbare Referenz und Korrekturmöglichkeit, auch wenn ein Teil des Wissens in Gewichten nutzbar wird. Flüchtige Fakten, personenbezogene Daten und ungeklärte Nutzungsrechte sind keine automatischen Trainingsziele. Keine automatische Übernahme privater Daten in globale Gewichte; Datenlöschung/Widerruf muss betroffene Artefakte auffinden, deaktivieren und aus zulässigen Quellen neu aufbauen können. Backups unter derselben Aufbewahrungs-/Zugriffspolitik.

## 3. Lernwege (keine feste kognitive Pipeline)

- MEMORY: sofortiger versionierter Abruf; Referenz und Vergleichsbaseline, kein Gewichtserwerb.
- DIRECT: Compiler für formal spezifizierte Assoziationen, Relationen, affine Maps, Logik und später deklarative Operatorgraphen.
- TRAINED: lokal begrenzte Optimierung kleiner Wissensexperten oder Adapter; optional Destillation aus autorisierten Lehrern.
- SYNTHESIS: testgesteuerte Suche/Konstruktion, ausdrücklich getrennt von DIRECT.
- PRETRAINED/SYMBOLIC und Mischformen: vorhandene Herkunftsklassen behalten.

Für den ersten Nachweis lieber kompakte numerische Assoziationsnetze als 100–500M-Module. Beispiel: endliche Entitäten/Relationen als unabhängig festgelegte IDs, Ausgabe als Wert/Entitäts-ID oder explizit unbekannt. Encoder/Decoder dürfen Identitäten repräsentieren, aber keine versteckte Zuordnung von Frage zu Antwort enthalten. Direkt konstruierte Assoziationsmatrizen sind zulässig; als exaktes parametrisches Nachschlagen und nicht als Reasoning ausweisen. Vergleich mit sparsamer Tabelle und Ledger, Kosten einschließlich Encoder, Index und Adapter. Neuartige Kombinationen bekannter Fakten getrennt von bloßer Wiedergabe messen. Unbekannte Fakten ohne Ableitung dürfen nicht erfunden werden.

Q4 primär für Inferenz, nicht zwingend für Optimierung; trainierbare Mastergewichte FP32/BF16 nach Backend, quantisierte Kandidaten nach gesonderter Paritätsprüfung. LoRA benötigt sein Basismodell: Lehrer-offline ist nicht automatisch Bootstrap-unabhängig. Vollständig eigenständige Experten separat nachweisen.

## 4. Learning Ledger und Validierungsvertrag

Vorgeschlagene Artefakte: LearningRequest (Ziel, Quellenrevisionen, erwarteter Nutzen, Scope, Ressourcen, Ablaufdatum); LearningJob (Methode, Version, Seed, Datenhash, Split, Budget, Elternartefakte); Candidate (Gewichte, ABI, Abhängigkeiten, Hash, Rechte, Herkunft); EvaluationReport (unabhängiger Testsatz, Vergleichsbaseline, Metriken, Fehler, Schwellen, Budget); PromotionRecord (alte/neue Population, Prüfentscheidung, Replay-/Rollback-Referenz). Alle versioniert.

Lebenszyklus: OBSERVED/HYPOTHESIS → EVIDENCE_CHECKED → LEARNING_ELIGIBLE → STAGED → EVALUATED → SHADOW → CANARY → ACTIVE; bei unzureichender Evidenz, Fehler, Kosten oder Konflikten HOLD/REJECTED/QUARANTINED; bei Korrektur STALE/REVOKED. Wissensstatus und Artefaktstatus sind getrennt. Dies ist ein technischer Lebenszyklus, keine zentrale fachliche Denksteuerung.

- **Quellenprüfung:** exakte Herkunft, Gültigkeit, unabhängige Evidenz oder formaler Beweis; Widersprüche sichtbar halten. Lehrer nicht zugleich alleiniges Orakel. Modelle können Lerngegenstand, Methode und Gegenbeispiele vorschlagen, aber nicht ihre eigene Freigabe erteilen.
- **Isolation:** aktive Gewichte und vertrauenswürdige Evaluatoren unveränderlich; Kandidaten im separaten, begrenzten Prozess, deklarative erlaubte Operatoren statt unkontrolliert ausgeführtem Modellcode. Keine selbst erteilten Berechtigungen, kein eigenmächtiger Netzwerkzugriff, kein Zugriff auf private Daten oder zurückgehaltene Testlabels. Keine Gewichtsänderungen am aktiven Bootstrap in-place. Schreibrechte für Runtime, Tests, Registry und Sicherheitsgrenzen getrennt.
- **Evaluation:** Trainings-, Entwicklungs- und finaler Abnahmedatensatz getrennt; Aufteilung nach Faktenfamilie/Relation oder Aufgabe, nicht nur Paraphrasen; vorab eingefrorene Tests, frische Holdouts gegen wiederholtes Optimieren auf die Abnahme. Testorakel unabhängig von der Lernquelle, deterministisch wenn möglich. Ungeklärte realweltliche Wahrheit als solche markieren. Prüfe neue Fakten, Varianten/Paraphrasen, ableitbare neue Kombinationen, Nichtwissen, Negationen, widersprüchliche und fachfremde Anfragen, frühere/seltene Kompetenzen, andere Gruppen, Q4, Kosten. Zufriedenheit und Modellübereinstimmung sind keine Wahrheitsorakel.
- **Freigabe:** kein Gesamtscore, der einen schweren Fehler durch viele leichte Erfolge kompensiert. Vorab definierte harte Gates für Scope, Datenschutz, Provenienz, frühere Kernfunktionen und Budget. Statistische Metriken mit Stichprobengröße/Unsicherheit für offene Aufgaben; bei kleinen exakten Fixtures vollständige Prüfung des definierten Bereichs. Keine Behauptung vollständiger Sicherheit durch Tests. Shadow ohne produktive Nebenwirkungen, danach begrenzte Canary-Exposition und unabhängige Vergleiche. Autonomer technischer Commit nur im autorisierten, getesteten Rahmen.
- **Persistenz:** Artefakt zunächst temporär schreiben, Hash/ABI validieren, dauerhaft speichern; atomarer Wechsel des aktiven Katalog-/Populationszeigers mit erwarteter Revision. Jobjournal, idempotente Schritte, Restart-/Crash-Tests an Commit-Grenzen, keine halben Freigaben. Cross-DB-Recovery/Transaktionsprotokoll erforderlich. Alte Version bleibt reaktivierbar. Kein automatischer Rollback auf eine inzwischen als falsch/widerrufen erkannte Version.
- **Korrektur:** expliziter Abhängigkeitsgraph vom Quellen-/Wissensrecord bis zu Datensatz, Gewichten, abgeleiteten Experten, Gruppenwissen und Antworten. Bei korrigierter/abgelaufener/entzogener Quelle betroffene Artefakte als STALE markieren, riskante Nutzung aussetzen und neu prüfen/rekonstruieren. Keine pauschale Löschung sämtlicher früherer Erkenntnisse, keine stille Fortführung eines veralteten Gewichtsstandes.
- **Retention:** Originale einfrieren, Replay repräsentativer alter und seltener Aufgaben, Regression der gesamten Population (auch Aktivierung und Kommunikation), bei Bedarf regularisierte Optimierung oder Destillation. Eingefrorene Parameter schützen nicht automatisch vor veränderter Auswahl/Kooperation. Konsolidierung erzeugt ausschließlich zusätzliche Kandidaten; die separate evolutionäre Selektion beurteilt später Qualität, Ressourcenmehrwert und Koexistenz (siehe 4a).

## 4a. Verbindliche Trennung: additive Lernoperationen und evolutionäre Reduktion

Drei voneinander unabhängige Operationen mit getrennten technischen Berechtigungen und Lebenszyklen:

| Operation | Zulässige Wirkung | Verbot |
|---|---|---|
| LEARN | Neue immutable Experten oder Varianten erzeugen und nach Validierung hinzufügen | Bestehende Experten überschreiben, ersetzen, deaktivieren oder löschen |
| CONSOLIDATE | Einen neuen Summenexperten aus mehreren bestehenden Kompetenzen erzeugen; Eltern-IDs und Herkunft erhalten | Ursprüngliche Experten ersetzen, deaktivieren, löschen oder deren Selektion vorwegnehmen |
| SELECT | Nach unabhängiger Langzeitbewertung und Populationsprüfung Nachfrage, Residency und Lebenszyklus bestehender Experten verändern | Gewichte verändern, eigene Prüfkriterien umschreiben oder einen Nachfolger allein aufgrund seiner Abstammung bevorzugen |

Ein Summenexperte ist ein funktionaler Kandidat, keine arithmetische Summe inkompatibler Gewichte. Er kann die Vereinigung, Teilmenge oder Komposition von Fähigkeiten abbilden; Umfang muss belegt werden. Seine Freigabe bedeutet ausschließlich zusätzliche Verfügbarkeit, keine Ablösung. Alle Eltern behalten ihre bisherigen Berechtigungen und faire Exposition. Der technische Evaluator prüft die Verträge, ist kein fachlicher Router.

Evolutionäre Selektion ist eine populationsweite, wiederholbare, kontextbezogene Bewertung. Metriken: tatsächlicher verifizierter Beitrag, kontrafaktischer Grenznutzen (Ablation einschließlich Koalitionen), einzigartige und seltene Kompetenz, Qualität, Fehlerkorrektur, Latenz, Arbeitsspeicher/VRAM, Speicherplatz, Aktivierung und Einladungen relativ zu fairen Gelegenheiten. Nachfrage und Nutzerzufriedenheit allein sind keine Löschkriterien. Abhängigkeiten, Ausweichpfade, Diversität, Reaktivierungskosten, Stichprobengröße und Unsicherheit berücksichtigen. Keine festgelegte Expertenzahl oder zwingende Reduktionsquote. Kandidaten können längere Zeit koexistieren oder dauerhaft spezialisiert bleiben.

Evolutionäre Selektion ist grundsätzlich reversibel: ACTIVE → DORMANT → ARCHIVED. **Keine physische Löschung von Expertengewichten im normalen evolutionären Betrieb**, auch nicht nach erfolgreicher Konsolidierung oder langjährig geringer Nutzung. Vollständige immutable Artefakte, ABI/Adapter, erforderliche Basismodellreferenzen, Eltern-IDs, Provenienz und Validierung bleiben auf SSD erhalten; Archivierung darf nicht heimlich Abhängigkeiten entfernen. Der kompakte Suchkatalog bleibt erreichbar; vollständige Kompetenzgewichte und gegebenenfalls Wahrnehmungsköpfe werden bei Bedarf geladen. Archivierte Experten sind über Kompetenzsuche, gezielte Einladungen und budgetierte Exploration reaktivierbar. Sicherheits-/Evidenz-Quarantäne bleibt davon getrennt: widerrufene Experten nicht automatisch reaktivieren. Gesetzliche Lösch-/Datenschutzanforderungen separat behandeln.

Residency und evolutionärer Lebenszyklus sind orthogonal: HOT (GPU-resident), WARM (CPU-zugänglich, soweit sinnvoll), COLD (SSD, nicht resident). ACTIVE kann COLD sein; DORMANT/ARCHIVED sind standardmäßig COLD. Auf Strix Halo CPU/GPU-Shared-Memory nach physischer Gesamtbelegung budgetieren, nicht doppelt zählen. Nur tatsächlich freigegebene Tensoren, Cache-Referenzen und Backend-Puffer gelten als Speicherentlastung. Hash-/ABI-Prüfung beim Laden, LRU oder kostenbewusster Cache nach Messung, Prefetch optional, begrenzte parallele Ladevorgänge. Metadatenindex und ausgewählte kleine Köpfe resident halten, ohne alle archivierten Kerne zu laden. SSD-Verbrauch, Ladezeit, Archivintegrität und Backups messen; bei erschöpftem SSD-Budget neue Lernjobs begrenzen oder zusätzlichen Speicher anfordern, nicht automatisch alte Gewichte löschen.

Abnahme: LEARN/CONSOLIDATE dürfen selbst mit vorgeschlagener Ablösung keine Eltern aus dem aktiven Katalog entfernen. Nach Summenexperten-Freigabe sind alle ursprünglichen Experten weiterhin verfügbar; ein Summenexperte, der seltene Fähigkeiten nicht abdeckt, kann keinen automatischen Ruhestand seiner Eltern auslösen. Separate SELECT-Tests mit fairer Exposition, Abhängigkeiten, seltenen Aufgaben, Abbruch, Reaktivierung und unveränderlichen Gewichten. Vergleich gleicher Gesamtpopulation mit/ohne Kandidaten und mit/ohne einzelne Eltern oder Elternkoalition, inklusive veränderter Kommunikation. Physische Löschung im normalen evolutionären Betrieb grundsätzlich ausgeschlossen. Tests für SSD-Auslagerung, Hashprüfung, tatsächliche Speicherfreigabe und Reaktivierung ohne Kompetenzverlust ergänzen. Schwellenwerte erst anhand belastbarer Daten festlegen.

## 5. Frühester vertikaler Nachweis (P0)

A. Kontrollierte, künstliche, zufällig erzeugte Domäne mit konkreten Beziehungen und stabilen IDs, z.B. Zuordnungen in einem fiktiven Katalog, deren Antworten das Startsystem nicht kennt. Unabhängiger datengenerierender Oracle hält korrekte Antworten und Holdouts zurück. Eine kontrollierte Lehrer-Fixture liefert Lernbeispiele; anschließend echte lokale Bootstrap-/Lehrer-Anbindung als separate Abnahme, inkl. falscher oder widersprüchlicher Antworten.

B. Lernjob erfasst verifizierte Beispiele, konstruiert oder trainiert einen kleinen Kandidaten (zunächst zwei getrennte Versuche: DIRECT Assoziationsmatrix und TRAINED kleiner MLP/Adapter), prüft und speichert dessen Gewichte. Auswahl des Verfahrens und Start des Jobs sind zunächst Host-Referenz; Autonomie erst bei einem späteren modellinitiierten Lauf beanspruchen.

C. Nach Neustart: Lehrer, Bootstrap, Faktenspeicher mit Zielantworten, vorheriger Chat/KV-Cache und Netzwerkzugriff im Evaluationsprozess deaktivieren. Nur deklarierter Encoder, Kandidat und unabhängiger Evaluator; Evaluator darf keine Antworten in den Vorwärtslauf einspeisen. Unbekannte Entitäten und Relationstypen mit Abstention prüfen. Kandidaten ablatieren/ersetzen und zeigen, dass der neu erworbene Wissensabruf verschwindet. Getrennt messen: exakter Recall, Paraphrase (nur falls Sprachadapter vorhanden), Transfer auf neue ableitbare Kombinationen.

D. Wissensrevision: ein Fakt wird korrigiert, ein Lehrer liefert Falschaussage, eine Quelle wird zurückgezogen. Bisherige korrekte alte Fälle bleiben erhalten; kein falsches Wissen wird als VERIFIED freigegeben; abhängige Kandidaten deaktiviert oder neu validiert; Rollback und Prozessabsturz vor/nach Veröffentlichung testen.

E. Mindestens drei sequenzielle Wissenspakete als erster Regressionstest; danach mehrstufige Langläufe (z.B. 10/100 Pakete als experimentelle Planwerte) mit Replay, seltenen Fakten und sich ändernden Beziehungen. Gegenüberstellungen: eingefrorenes System; Ledger-only; isolierte Gewichte; Ledger+Gewichte; mit/ohne Replay; mit/ohne Konsolidierung. Gleiche Daten- und Rechenbudgets, vollständige Kosten incl. Lehrer und Evaluation. Vergessen und Kosten über Zeit ausweisen.

F. Erst nach belastbarem Wissensnachweis: bekannte Methoden/Regeln, kombinatorischer Transfer, selbst initiierte Lernjobs, Wiederverwendung erfolgreicher Generatoren, Verbesserung der Lernverfahren und später Änderung kognitiver Strukturen.

Keine zwingende Wartezeit bis zur vollständigen Environment-Abnahme. Benötigte gemeinsame Runtime-/Ledger-/Factory-Verträge zuerst in einem vertikalen Pfad stabilisieren und parallel echte Bootstrap-Anbindung herstellen. Skalierung, universelle semantische Adapter und native Engine bleiben eigenständige Arbeitspakete.

## 6. Messgrößen

Erwerbsrate (neues verifiziertes Wissen / Aufwand), Retention pro Wissenspaket, Transfer auf unbekannte ableitbare Aufgaben, False-Accept/Abstention, Widerspruchs- und Korrekturquote, Zahl veralteter aber noch aktiver Artefakte, Lehrerabhängigkeit, Aufwand für Training/Evaluation/Inferenz, Peak RAM/VRAM, Speicherbedarf/Kompetenz, Lernjob-Ausfall und Recovery. Messung von Kooperation und kausalem Beitrag getrennt von Aktivierungshäufigkeit. Kein versprochener monotoner Intelligenzzuwachs. Hypothese des Lernens des Lernens prüfen durch gleiche neue Wissensdomänen, eingefrorene vs weiterlernende Lernstrategie und gleiche Budgets; Erfolg ist verbesserte Stichproben-/Recheneffizienz ohne Retentionsverlust.

## 7. Forschungsbezüge (erneute gezielte Recherche, 27.11.2026)

- [Parisi et al., Continual lifelong learning with neural networks (2019)](https://doi.org/10.1016/j.neunet.2019.01.012): Überblick über Stabilität/Plastizität, Replay und Complementary Learning Systems. Motivation für zwei Zeitskalen, keine exakte Gehirn-Nachbildung.
- [Mitchell et al., SERAC (ICML 2022)](https://proceedings.mlr.press/v162/mitchell22a.html): externe Edit-Memory, die Vorhersagen beeinflusst; Vergleichsbaseline, kein eigenständiger Gewichtserwerb.
- [Hartvigsen et al., GRACE](https://arxiv.org/abs/2211.11031): lokale latente Key-Value-Adapter ohne Änderung der Basisgewichte; Inspiration für reversible lokale Korrekturen und Gültigkeitsbereiche.
- [Gupta et al., Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://arxiv.org/abs/2401.07453): sequenzielle ROME/MEMIT-Edits können frühere Edits und andere Aufgaben beeinträchtigen; gegen unkontrollierte In-place-Updates.
- [Schwarz et al., Progress & Compress (ICML 2018)](https://proceedings.mlr.press/v80/schwarz18a.html): getrennte Erwerbs- und Konsolidierungszyklen, motiviert Growth+Compression; keine Garantie für unsere offene heterogene Population.
- [Voyager](https://arxiv.org/abs/2305.16291): Curriculum, Code-Skill-Library, Umgebungsfeedback und Selbstprüfung in Minecraft; kein Gewichtstransfer.
- [Knowledge-Centric Self-Improvement (2026, Preprint)](https://arxiv.org/abs/2607.19592): kuratierte persistente Erkenntnisse, evidenzbasierte Beiträge und Transfer; Wissensbasis statt Gewichtsänderung als primärer Verbesserungsgegenstand.
- [Avalanche](https://github.com/ContinualAI/avalanche): PyTorch-Framework für reproduzierbare Continual-Learning-Evaluation; Kandidat für Baselines, keine verpflichtende Laufzeitabhängigkeit.
- [Tracr](https://arxiv.org/abs/2301.05062): formale Programme in Transformer-Gewichte kompilieren; unterstützte formale Operationen, keine beliebige Intelligenz.
- [WeightCode (2026, Preprint)](https://arxiv.org/abs/2611.25476): ausführbare Gewichtsprogramme für bereits trainierte Netze; Forschungsoption, nicht direkter Beweis für unabhängige Experten.
- [Program-as-Weights – Methodenvergleich](https://programasweights.com/research/weight-space-programming/): gelernter Compiler für Adapter eines kompatiblen Interpreters, nicht freie eigenständige Gewichtsarchitektur.

Keine Replikation, vollständige Literaturreview oder Fremdcode-Integration erfolgt. Neue Studien sind Forschungsreferenzen, keine zugesicherten Fähigkeiten.

## 8. Ist-Zustand und offene Blocker

v0.7 hat getrennte Kandidaten/affine Factory, Quellenrevisionen, logisch begrenzte Speichersichten und lokalen HTTP-Testadapter. Die neue Runtime ist nicht der reguläre Chat-Pfad. Keine autonome parametrische Wissensaufnahme, keine unabhängige Lernjob-Evaluation, keine transitive Provenienz, kein echtes LLM, keine Cross-DB-Atomarität und keine garantierten physischen Ressourcenlimits. Bestehende 104 lokale Tests sind keine Abnahme dieses Vertrags. Hier wird ausschließlich der Entwicklungsplan erweitert.
