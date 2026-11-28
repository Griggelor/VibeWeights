# v0.8.0.dev7 – Additive Ergänzung vorhandener Wissensdomänen

## Ziel und Vertrag

Dev6 erkennt fehlende Domänen und Enthaltungen in typisierten numerischen Aufgaben. Dev7 ergänzt den Fall einer einzelnen unbekannten Entität in einer bereits bestehenden Domäne. Die vorhandenen Experten bleiben unverändert. Kein zentraler Router, keine automatische Ablösung, keine evolutionäre Löschung und keine RAM-Zwischenstufe. Die regulären Zustände ACTIVE / LOW_ACTIVE_IMPORTANT / ARCHIVED bleiben unverändert.

`ExtensionLearning(TaskLearning(...))` akzeptiert ausschließlich eine persistierte ABSTAINED-Beobachtung. Der Host autorisiert Ziel, Lehrmodell, Anfrage, Job- und Experten-ID und ein unabhängiges Orakel. Vor und nach Modell-I/O prüft der Source-Guard die aktuellen Quellen, die genaue Domain-Spezifikation und das aktuelle Verhalten der residenten Experten. Jede bereits vorhandene globale Record-Revision zum Zielschlüssel (einschließlich nicht verifizierter, archivierter oder bereits im Lernprozess befindlicher Fakten) blockiert das Einlernen. Dafür sind Korrektur, Fortsetzung oder Reaktivierung erforderlich. Keine automatische Neuinterpretation von Konflikten als Wissenslücke. Abweichende Domänen und untypisierte Experten werden fail-closed behandelt.

Die Lehrantwort darf nur ein JSON-Objekt mit genau `entity` und `label` enthalten. Doppelte Schlüssel, boolesche IDs, andere Entitäten, unzulässige Labels und Abweichungen vom unabhängigen Orakel werden abgewiesen. Das Modell sieht keine Sollwerte des Orakels. Der Host bleibt vertrauenswürdig; keine OS-Sandbox oder unabhängige Verifikation realweltlicher Quellen in dieser Version.

Ein erfolgreicher Intake schreibt genau einen neuen verifizierten Fakt und das persistente Lernintent gemeinsam in die Source-DB. Das Intent bindet Beobachtung, Anfragehash, Modellkennung, Oracle-ID und Lernverfahren. Wiederholung desselben Auftrags nach Neustart setzt den persistierten Intake fort, ohne das Lehrmodell erneut aufzurufen. Änderungen an dieser Policy werden zurückgewiesen. Vor dem Commit darf ein Retry erneut Modell-I/O auslösen. Die anschließende Konstruktion nutzt die bestehende DIRECT- oder SGD-Factory und erzeugt STAGED. `promote` prüft separat das neue Faktum und Enthaltung auf allen übrigen Entitäts-IDs (inkl. Grenzfälle im bestehenden Evaluator). Der alte Experte bleibt resident. Die generischen KnowledgeLearning- und Catalog-APIs sind weiterhin vertrauenswürdige Host-Schnittstellen, keine Berechtigungs-Sandbox.

## Beispiel

```python
from vibeweights.task_learning import TaskLearning
from vibeweights.extension_learning import ExtensionLearning
# learning ist ein vorhandenes KnowledgeLearning-Objekt.
# domain, teacher und oracle sind unabhängig vom Kandidaten definierte Verträge.
tasks = TaskLearning(learning)
obs = tasks.observe('missing-fact', domain, 2)
assert obs['outcome'] == 'ABSTAINED'
ext = ExtensionLearning(tasks)
ext.stage('missing-fact', teacher, request='Provide the approved fact for entity 2',
          job_id='extension-1', expert_id='additional-1', oracle=oracle,
          oracle_id='independent-oracle', method='TRAINED')
assert ext.promote('extension-1', oracle, oracle_id='independent-oracle')
print(tasks.observe('after-extension', domain, 2))
```

Das Orakel darf beim Promovieren die Wahrheit der gesamten Domäne kennen: der neue Ergänzungsexperte soll nur das ihm zugewiesene neue Faktum liefern und auf den anderen IDs abstainieren. `TaskLearning.observe` vereinigt übereinstimmende bekannte Antworten; Widersprüche ergeben CONFLICT und keine Antwort. Das ist keine autonome semantische Gruppenbildung.

## Konfliktprüfung

`review_conflict(review, task, oracle, oracle_id=...)` erfordert eine persistierte CONFLICT-Beobachtung und prüft erneut die aktuellen Ausgaben. Ein unabhängiges Orakel kann abweichende Experten identifizieren; bei unbekanntem Sollwert bleibt der Konflikt UNRESOLVED. Der Bericht ist unveränderlich unter seiner Review-ID. Keine automatische Quellenrevision, Quarantäne, Selektion oder Gewichtsänderung; dafür ist zusätzliche verifizierte Evidenz bzw. ein eigenständiger autorisierter Prozess erforderlich. Der Test injiziert eine fehlerhafte Ausgabe, kein nachgewiesener realweltlicher Quellenkonflikt.

## Abnahme und Grenzen

Zehn neue Tests: DIRECT/TRAINED-Ergänzung, Erhalt des alten Gewichts-Hashes und alter Antworten, Enthaltung des neuen Experten außerhalb seines Zuständigkeitsbereichs, Wiederanlauf ohne erneuten Lehreraufruf, vier fehlerhafte Lehrerantworten, existierender Fakt, Änderung während Modell-I/O, unterbrochener Intake-Auftrag und abweichendes Freigabeorakel, Konfliktprüfung ohne Populationseingriff. Testlauf mit Python/PyTorch CPU und Modell-Doubles. Keine reale LLM-Inferenz, Windows-, GPU- oder Remote-CI-Abnahme. Kein autonomer Scheduler oder freie semantische Lernzielbildung. Aktuell ein Fakt je Auftrag, mit voller affiner Matrix; nicht wirtschaftlich für große Domänen. Skalierbare transitive Quellenindexierung und gemeinsamer REMEMBER-/Learning-Transaktionsvertrag sind weiterhin P0, nicht Bestandteil dieses Slices. Keine allgemeine Generalisierung oder vollständige Konfliktauflösung behauptet.

```sh
python -m pip install -e ".[test]"
python -m pytest -q tests/test_extension_learning_v08.py
python -m pytest -q
```
