# Live construction-scene analysis on Jetson Orin Nano

YOLO11n draws object boxes and helps select camera frames. SmolVLM2-500M
describes selected images while capture continues. In the November 24
five-video demonstration, camera delivery continued during all 111 completed
requests. Hazard and fall detection remain unvalidated.

| Start here | What you'll find |
|---|---|
| [Research meeting log](docs/research-log/2026-11-24-meeting-log.md) | Architecture, five MP4s with viewing timestamps, results, and next steps |
| [Detailed five-video results](docs/research-log/2026-11-24-five-video-results.md) | Run settings, timings, and source images behind example captions |
| [Architecture and setup](docs/scout-inspector-architecture.md) | Deployment and runtime details |
| [Research log](docs/research-log/) | Earlier experiments and benchmarks |

The current pipeline is in `src/scout_inspector/`. Launchers and historical
Moondream experiments are in `experiments/`. Saved measurements and recordings
are in `results/`. Local model files and environments are excluded from Git.
