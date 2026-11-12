# Construction-scene analysis on Jetson Orin Nano

The current implementation is a USB **Scout & Inspector** pipeline: YOLO TensorRT
capture/detection worker → replaceable one-frame queue → one SmolVLM2-500M
Inspector worker. OpenCV preview runs on the main thread.

For the provisioned setup, read [the short launch guide](docs/scout-inspector-quickstart.md).

Read [architecture, setup and launch instructions](docs/scout-inspector-architecture.md)
and [current research context](docs/context.md). The new architecture has CPU
contract tests; hardware throughput, quality and memory use remain untested.
Historical Moondream experiments and results are preserved under `experiments/`
and `results/`.

```bash
python3 -m unittest discover -s tests -v
python3 main.py --help
```

No camera or GPU work occurs in these checks. See the setup guide before installing
optional dependencies or running hardware. Generated safety observations are
unverified; this is not a validated hazard/event detection system.
