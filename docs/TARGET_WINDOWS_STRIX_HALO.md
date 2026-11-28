# Zielsystem: Windows 11 / AMD Strix Halo

Primäre Zielplattform laut Nutzer: Windows 11 auf AMD Strix Halo. Linux bleibt CPU-Referenz, nicht Abnahmeersatz. Konkrete RAM-/UMA-Konfiguration, Treiberversion, PyTorch-Build und GPU-Backend sind noch nicht erhoben. Keine Annahme eines bestimmten verfügbaren HIP-Builds oder einer bestimmten VRAM-Größe.

## Stufe 1: reproduzierbare CPU-Basis

Windows-CI mit Python 3.12/3.13 bleibt verpflichtend, inklusive bytegenauer Artefaktreproduktion. CI 46141227674: Linux beide erfolgreich, Windows jeweils 241/242; einzig `test_head_reproduction` scheitert an CRLF im neu generierten Coalition-JSON. Git-Checkout erzwingt LF. Generatorfix `2a8d1753c555f5602233eea72097f914c434a15c` in dev18, auch in dev19 übernommen. Keine Änderung der Gewichte oder Vergleichsanforderung.

## Stufe 2: explizite Geräteprüfung auf dem Zielrechner

In der gewünschten Python-Umgebung aus dem Repository:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/probe_population_device.py --require-windows --device cpu --output probe-cpu.json
python scripts/probe_population_device.py --require-windows --device cuda:0 --output probe-gpu.json
```

Der letzte Befehl setzt einen funktionierenden PyTorch-GPU-Build voraus. Bei HIP nutzt PyTorch ebenfalls die Schnittstelle `torch.cuda`; ein fehlendes Backend führt explizit zu FAIL, niemals zu CPU-Fallback. Das Skript meldet Gerätetyp, tatsächliches Tensordevice, Gerätebezeichnung, Torch-/HIP-/CUDA-Version, Allocator-Werte (soweit GPU) und Prüfschritte. Es nutzt einen temporären Katalog, verändert keine vorhandenen Experten und benötigt keinen Modell-Download. CI führt die CPU-Variante aus. Die GPU-Ausführung auf Strix Halo steht aus.

## Stufe 3: weitergehende Hardware-Abnahme

- Separat prüfen: Gewichtsresidenz und Archivierung bei wachsender Population, reale UMA-/Allocator-Grenzen, freie Systemressourcen und stabile Wiederaufnahme.
- Hostgetriebene synchrone Affine-Referenz ist noch keine parallele GPU-Kooperation. Keine Durchsatz- oder Latenzziele ohne Messungen.
- CPU-Basis bleibt erhalten. Keine Auswahl oder Installation eines alternativen Backends ohne kompatiblen, reproduzierbaren Testpfad. Weitere Backends und Training auf GPU bedürfen eines eigenen Implementierungs- und Abnahmeabschnitts.
