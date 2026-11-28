# v0.8.0.dev6 – Lernbedarf aus Aufgaben

Status: ausführbarer, begrenzter numerischer Referenzpfad. Ergänzt dev5 ([Lerninitiative](LEARNING_INITIATIVE_v08.md)). Kein produktiver Chat-Ersatz, keine allgemeine autonome Lernfähigkeit. Alle bisherigen Lern-, Konsolidierungs-, Selektions- und SSD-Archivregeln bleiben unverändert.

## Ablauf und Schnittstellen

`TaskLearning(learning)` nimmt eine konkrete typisierte Abfrage `observe(task, domain, entity)` entgegen. Kein erwartetes Ergebnis und kein Orakel werden an den Abfragepfad übergeben. Er verwendet die passenden freigegebenen residenten Wissensexperten (exakte Domain einschließlich Dimensionen), Live-Quellenprüfung und die vorhandene Identitätskodierung. Mehrere Antworten müssen übereinstimmen; Enthaltungen sind keine Widersprüche. Es findet kein Faktenabruf als Ersatz für den Gewichtspfad statt. Der Host gibt Domäne und Entitäts-ID vor, kein allgemeines semantisches Parsing oder dynamische Koalitionsbildung. Das ist eine opt-in Wissensabfrage, kein globaler kognitiver Router.

| Beobachtung | Bedeutung | Neue Domäne lernen? |
|---|---|---|
| ANSWERED | Gewichtspfad hat eine eindeutige Antwort geliefert, noch kein neuer Wahrheitsbeweis | Nein |
| ABSTAINED | Residente Experten enthalten keine eindeutige Antwort | Nein (Erweiterung bestehender Domänen noch offen) |
| CONFLICT | Residente Experten liefern unterschiedliche gültige Antworten | Nein (Konfliktklärung separat) |
| RESERVED_DOMAIN | Fakten, Aufträge, Intakes oder Experten existieren, aber kein nutzbarer residenter Wissensexperte | Nein (Reaktivierung, Fortsetzung oder Korrektur separat) |
| MISSING_DOMAIN | Kein passender Wissensbestand bzw. keine Reservierung vorhanden | Ja, falls Host-Policy die exakte Domäne erlaubt |

Fehler (z.B. Source-Guard, ungültige Eingabe oder ABI-Konflikt) werden nicht als Wissenslücke behandelt. Archivierte und quarantänisierte Experten werden nicht automatisch reaktiviert, überschrieben oder gelöscht. Niedrige Aktivität löst keine Auslagerung aus.

## Persistenz und Lerninitiative

- `task_observations_v08` im Expertenkatalog: eindeutige Task-ID, genaue Domäne, Entity-ID, beobachteter Zustand, beteiligte Experten und Ergebnis. Keine Rohprompts, keine Prüflösungen. Historische Beobachtung ist kein aktueller Faktenbeleg; neue Abfrage benötigt neue Task-ID.
- Index `(topic,domain_sha,outcome,seq)` ermöglicht gezielte Auswahl der jüngsten beobachteten Lücke je zugelassener Domäne. Kein Vollscan des Taskjournals; **die globale Quellenprüfung und bisherige Katalog-/Jobprüfung sind weiterhin nicht skalierungsoptimiert.**
- `propose(cycle, opportunities, proposer)` prüft maximal acht explizit erlaubte Optionen erneut. Nur aktuell unreservierte Domänen mit genau passender MISSING_DOMAIN-Beobachtung sind eligible. Der persistente Plan bindet vollständigen Policy-Hash und Task-IDs, bevor Modell-I/O erfolgt. Leere Pläne lösen keinen Modellaufruf aus. Wiederholung desselben Cycle verwendet den eingefrorenen Plan und das dev5-Receipt; eine neue Beobachtung erfordert einen neuen Cycle. Explizite Wiederholung nach Abbruch vor Receipt kann erneut Modell-I/O auslösen.
- `stage` verlangt den unveränderten Plan und einen zugelassenen Lehrer plus unabhängiges Orakel. Der bestehende Intake-/Recovery-Pfad wird wiederverwendet. Keine automatische Admission; `KnowledgeLearning.promote` prüft weiterhin unabhängig. Modellvorschläge bleiben untrusted. Kein autonomer Scheduler oder unbegrenzte Wiederholungen; ein neuer Cycle muss vom vertrauenswürdigen Host ausdrücklich ausgelöst werden. Ein SKIP bleibt für diesen Cycle persistent.
- Ein bereits gelerntes oder anderweitig reserviertes Thema wird vor Auswahl erneut ausgeschlossen. Der alte dev5-Pfad bleibt für explizite Testaufträge vorhanden; TaskLearning ist kein OS-Sandboxing oder Schutz gegen einen privilegierten Host.

## Echttest mit LM Studio (vorbereitet, nicht hier ausgeführt)

Frischer eigener Datenroot, geladenes kompatibles lokales Modell und installierter Repo-Stand. `MODEL_ID` durch exakte Modell-ID ersetzen. Der 0-basierte Entity-Wert 1 bedeutet Februar; Ausgabe 1 bedeutet 29 Tage. Die Python-Standardbibliothek prüft unabhängig alle zwölf Monate.

```sh
python scripts/learning_initiative_lmstudio.py observe --root Data/task-demo --topic calendar-2024 --entity 1 --task before
python scripts/learning_initiative_lmstudio.py task-propose --root Data/task-demo --model MODEL_ID
python scripts/learning_initiative_lmstudio.py task-stage --root Data/task-demo --model MODEL_ID
# Modellserver kann jetzt ausgeschaltet werden. Falls das Modell SKIP wählt, endet der Lernzyklus ohne Kandidat.
python scripts/learning_initiative_lmstudio.py promote --root Data/task-demo
python scripts/learning_initiative_lmstudio.py observe --root Data/task-demo --topic calendar-2024 --entity 1 --task after
```

Erwartete Folge bei LEARN und korrektem Lehrer: MISSING_DOMAIN → Plan nur calendar-2024 → STAGED → ACTIVE → ANSWERED (1). Neue Abfrage liest Gewichte, nicht Zielantwort aus dem Ledger; Live-Quellenprüfung bleibt aktiv. Das Modell darf SKIP wählen; falsche Fakten werden verweigert. Der Lehrer kann vor Freigabe und Folgeabfrage offline sein. Kein echtes LLM in dieser Testsuite, keine freie autonome Zielbildung.

## Tests und Grenzen

Neun neue Tests: Lernkette mit Wiederanlauf, fehlende Beobachtung, eingefrorener leerer Plan, falsche Domäne und doppelte Task-ID, unzulässige Auswahl und Policy-Wechsel, Archiv/Faktenreservierung, veraltete Lücke, Fehler nicht als Lücke, Kalender-CLI mit simulierten Modellantworten, persistentes SKIP (einige Fälle kombiniert). Gesamtnachweis: `evidence/v08-task-learning-summary.json` und Testlog/JUnit.

Offen: gezieltes Lernen einzelner fehlender Fakten innerhalb bestehender Domänen; Konfliktklärung, automatische Reaktivierungsentscheidung, dauerhaftes Lernbudget/Trigger für gewöhnlichen Chatbetrieb, Indexierung transitiver Quellenabhängigkeiten, gemeinsamer Source-/REMEMBER-Transaktionsvertrag und vollständige Crash-Atomarität. Kein Windows/Strix-Halo/GPU-/Remote-CI-Nachweis. Kein neues allgemeines Regel-/Generalisierungsverfahren. Die drei regulären Expertenzustände ACTIVE (VRAM), LOW_ACTIVE_IMPORTANT (VRAM), ARCHIVED (SSD) bleiben unverändert. Lernen und Konsolidieren bleiben additiv.
