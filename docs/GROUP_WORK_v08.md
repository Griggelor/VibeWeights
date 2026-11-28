# v0.8.0.dev12 – Ausführung freiwilliger neuronaler Arbeitsgruppen

## Ziel und Vertrag

Dev11 zeigt trainierte lokale Beteiligungsentscheidungen. Dev12 verbindet diese Entscheidungen mit der bereits vorhandenen numerischen `Environment`-Runtime. Eine Gruppe ist eine durch Event, Ziel und Gruppenschlüssel definierte Menge aktuell residenter Experten, die sich unabhängig mit ihren freigegebenen Wahrnehmungsköpfen gemeldet haben. Kein zentraler Router legt Reihenfolge, fachliche Rollen oder Signalpfade fest. Die technische Runtime prüft lokale Relevanz gegen einen unveränderlichen Snapshot, führt aktive Gewichte aus und veröffentlicht unabhängig validierte Zwischensignale in Folgerunden.

`GroupWork(PerceptionHeads(...)).execute(...)` verlangt ein aktuelles globales Event, eine bestehende freiwillige Gruppe, einen eindeutigen Run-Identifier, typisierte numerische Startsignale, das Zieltopic, unabhängige Validatoren und eine Orakel-ID. Der vertrauenswürdige Host definiert den Task, Topic-Verträge und Validatoren, nicht die Ausführungsreihenfolge. Pro Teilnehmer wird die aktuell freigegebene Kopfbindung, der Artefakthash und die lokale Entscheidung erneut geprüft. Callback-Mitgliedschaften ohne bestätigende neuronale Kopfentscheidung reichen nicht aus. Archivierte und quarantänisierte Experten werden nicht geladen. Andere residente Experten sind von der Gruppenausführung ausgeschlossen, selbst wenn sie direkt eine Lösung liefern könnten.

Die erste Ausführung unterstützt affine Kerne mit `numeric-events/0.1` und globalem Zielscope. Der Gruppenadapter beschränkt sämtliche Eingaben und Ausgaben auf einen eindeutigen, run-spezifischen Gruppenscope; er übersetzt keine fachlichen Inhalte und berechnet keine Antwort. Er delegiert an `Population.compute`, also die tatsächlichen residenten Kerngewichte. Eine Gruppe darf aus einem oder mehreren Experten bestehen, mehrere Gruppen können dieselben Experten verwenden. Kein Informationsaustausch zwischen getrennten Runs, kein Schreiben ins Wissensledger und keine automatische Kandidatenaufnahme. Weiterhin bleiben Lernen, Konsolidierung und Selektion getrennt.

Die `Environment`-Validatoren erhalten Snapshot und Vorschlag, nicht umgekehrt: der Kern bekommt ausschließlich das Eingangssignal. Unverifizierte Zwischensignale bleiben Hypothesen und können nicht von Folgeexperten konsumiert werden. Konflikt, Stagnation, Budget oder Fehler liefern keinen erfolgreichen Zielwert. Der Gruppenwrapper stuft auch bei einem ansonsten erfolgreichen Environment-Lauf jeden gemeldeten Ausführungsfehler als INVALID ein. Steps, Runden, Signale und Teilnehmerzahl sind begrenzt (max. 1024/64/2048/64); keine Explorationsprobes in diesem Slice, keine physische Rechenzeit- oder Peak-Memory-Garantie.

Ein Run-Receipt speichert Scope, Eingaben, Orakel-ID, Validator-Topics, Mitglieder samt Kern- und Kopf-Hashes, Limits, Resultat, Hypothesen und Trace. Die Katalog-SQLite-Transaktion persistiert den gesamten Receipt mit SHA256. `receipt()` ist ausdrücklich historische Evidenz und keine erneute Freigabe. Run-IDs sind nicht wiederverwendbar. Vor und nach der Ausführung wird der Event-/Gruppenzustand geprüft; der bestehende Source-Guard gilt. Ein unterbrochener Run vor Receipt-Commit muss neu ausgeführt werden. Kein allgemeiner Crash-/Cross-DB-Transaktionsnachweis, keine automatische Wiedereinspielung historischer Ergebnisse. Die Validatoren sind vertrauenswürdige reine Host-Funktionen; kein Sandbox-Schutz vor absichtlich manipulierenden Python-Callbacks.

## Experiment

Ein zuvor freigegebener Wissensexperte publiziert ein KNOWLEDGE-Event. Drei weitere residente, unabhängig abgenommene affine Kerne besitzen tatsächlich trainierte, separat abgenommene Wahrnehmungsköpfe. Zwei Köpfe melden sich für `research`, der dritte lehnt ab. Die beiden Teilnehmer besitzen die Teilfunktionen `mid=2*x+1` und `answer=3*mid+2`; ihre IDs sind alphabetisch gegen die fachliche Reihenfolge angeordnet. Ohne manuell kodierte Reihenfolge entsteht die Folge aus der Verfügbarkeit unabhängig validierter Zwischensignale. Der nicht beteiligte dritte Kern könnte die Gesamtfunktion `6*x+5` direkt berechnen, darf aber nicht einspringen.

Bei Eingabe 4 ist das Gruppenergebnis 29, bei 7 nach Neustart 47. Die Ablation jedes der beiden Teilnehmer einzeln führt zur STAGNATION ohne Zielwert, obwohl der direkte Shortcut resident bleibt. Für diese Ablation wird die bereits vorhandene explizite Selektionsschnittstelle mit Fixture-Evidenz verwendet, nicht ein automatischer Eingriff in die reale Population. Die ursprünglichen Gewichtshashes sind unverändert. Die Tests verifizieren abgelehnte Zwischensignale, Budgetgrenze, Fehlerbehandlung, ungültigen Event nach Quellenrevision, historischen Receipt nach Neustart, beschädigten Receipt und Ablehnung von untrainierter Callback-Gruppenbildung.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_group_work_v08.py
python -m pytest -q
```

```python
from vibeweights.group_work import GroupWork
# heads: bestehende PerceptionHeads-Instanz mit freigegebenen Köpfen und Eventgruppe
work = GroupWork(heads)
receipt = work.execute('run-1', event_id, 'explore', 'research',
    {'input': (4,)}, 'answer',
    {'mid': lambda snapshot, value: value == (9,),
     'answer': lambda snapshot, value: value == (29,)}, oracle_id='independent-fixture')
assert receipt['result']['status'] == 'OK'
```

## Grenzen und nächste Schritte

Dies ist ein erster funktionaler Nachweis einer freiwillig per trainiertem Kopf gebildeten und tatsächlich rechnenden Gruppe. Die eigentlichen Rechenkerne sind hier DIRECT-affine Referenzen, keine importierten Transformerfragmente. Die Wahrnehmung nutzt weiterhin sechs hostdefinierte numerische Eventmerkmale und feste Gruppenlabels; keine freie semantische Aufgabeninterpretation, dynamische Gruppenrekursion oder selbstständige Rekrutierung. Die Taskausführung wird vom Host ausgelöst, nicht durch einen autonomen Scheduler. Zwischenziele und unabhängige Prüffunktionen sind ebenfalls vorgegeben. Keine Generalisierung auf freie Sprachaufgaben, keine echte LLM-Assimilation. Keine automatisierte Übernahme von Gruppenergebnissen in neue Gewichte oder Kooperationserfahrung, keine indexierte transitive Quellenprüfung, kein gemeinsamer REMEMBER-/Learning-Transaktionsvertrag und keine GPU-/Windows-/Remote-CI-Abnahme. Nächste Schritte: überprüfbarer Gruppen-Lernauftrag aus bestätigtem Ergebnis, wiederverwendbare Kooperationskompetenzen, Gruppenrekrutierung, latente ABI und die weiterhin offenen Infrastrukturarbeiten. Fragment-Assimilation bleibt separater Forschungsweg nach [FRAGMENT_ASSIMILATION.md](FRAGMENT_ASSIMILATION.md).
