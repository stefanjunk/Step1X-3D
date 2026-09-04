# Step1X-3D mit zwei NVIDIA-GPUs

Das Image verwendet die vom Projekt getestete Kombination:

- Python 3.10
- CUDA Toolkit 12.4.1 und cuDNN (Development Image)
- PyTorch 2.5.1, Torchvision 0.20.1 und Torchaudio 2.5.1 mit CUDA 12.4
- Kaolin 0.17.0 und Torch Cluster 1.6.3 für PyTorch 2.5/CUDA 12.4
- Gradio 5.5.0 mit Pydantic 2.10.6 (verhindert den booleschen JSON-Schema-Fehler)

Die Web-App verteilt die Geometrie-Pipeline auf `cuda:0` und die
Textur-Pipeline auf `cuda:1`. Das CPU-Offloading ist für den 16-GB-RAM-Host
deaktiviert: Die FP16-Texturgewichte bleiben auf GPU 1, während die großen
FP32-VAE-Schritte mit Tiling auf der freien Reserve von GPU 0 laufen. Die
Web-App reicht außerdem die bereits von der Geometrie-Pipeline freigestellte
RGBA-Eingabe an die Textur-Pipeline weiter und lädt daher kein zweites
Segmentierungsmodell. Nur der Standalone-Texturpfad verwendet bei RGB-Eingaben
BiRefNet kurzzeitig auf `cuda:1`; die gepinnte Modellinstanz wird noch vor SDXL
wieder freigegeben. Gradio verarbeitet jeweils nur eine Generierung gleichzeitig.

## Voraussetzungen

Auf dem Host müssen der NVIDIA-Treiber, Docker und das NVIDIA Container
Toolkit funktionieren. Vor dem Build prüfen:

```bash
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
```

Beide Befehle müssen zwei GPUs anzeigen. Der Container bringt CUDA und die
Python-Bibliotheken mit, kann aber keinen fehlenden oder nicht geladenen
Host-Treiber ersetzen. Plane für Image, Build-Cache und Modellgewichte
mindestens 50 bis 70 GB freien Speicher ein. 16 GB RAM funktionieren knapp und
benötigen ausreichend Swap für die Lastspitzen beim Modellstart; 32 GB oder
mehr verkürzen den Start deutlich.

## Bauen und starten

```bash
cp .env.example .env
mkdir -p .cache/huggingface cache output
docker compose build
docker compose run --rm step1x3d nvidia-smi
docker compose up
```

Die Oberfläche ist danach unter <http://localhost:7861> erreichbar. Der erste
Start dauert deutlich länger, weil Step1X-3D, SDXL, VAE und rembg ihre Gewichte
laden. Hugging-Face-Modelle und das rembg-ONNX-Modell bleiben in den
konfigurierten Docker-Caches erhalten.

Logs und Stoppen:

```bash
docker compose logs -f step1x3d
docker compose down
```

Die Beispiel-Inferenz kann mit derselben Geräteaufteilung gestartet werden:

```bash
docker compose run --rm step1x3d python inference.py
```

## GPU-Architektur

Der Build ist über `TORCH_CUDA_ARCH_LIST=8.9` gezielt auf die beiden RTX 4060
Ti (Ada Lovelace) eingestellt. Das spart viel Buildzeit und Image-Speicher.
Nur bei einem späteren Wechsel des GPU-Typs muss der Wert in `.env` angepasst
und das Image neu gebaut werden:

- Turing/T4/RTX 20: `7.5`
- Ampere A100: `8.0`
- Ampere RTX 30/A10/A40: `8.6`
- Ada RTX 40/L4/L40: `8.9`
- Hopper H100: `9.0`

Nach einer Änderung ist ein erneuter `docker compose build` erforderlich.
Blackwell-GPUs sind nicht Teil dieser CUDA-12.4/PyTorch-2.5.1-Matrix.
