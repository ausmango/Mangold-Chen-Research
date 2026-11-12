"""Hardware launches are explicit; --check does not import GPU libraries."""
import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
import time

from .config import load_config, check_assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/scout_inspector.example.json')
    parser.add_argument('--run', help='New results directory; must not already exist')
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--camera',choices=['usb','csi'],help='USB webcam or existing IMX219 CSI pipeline')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--wait-for-start',action='store_true',help='Preview alignment; S arms timed capture and Inspector')
    parser.add_argument('--source-label',help='Saved source-video label/provenance')
    parser.add_argument('--check', action='store_true', help='Validate local configuration/assets without camera/GPU imports')
    args = parser.parse_args(argv)
    if args.wait_for_start and not args.preview:
        parser.error('--wait-for-start requires --preview')
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be finite and positive')
    try:
        cfg = load_config(args.config)
        if args.camera:
            cfg['camera']['kind'] = args.camera
            if args.camera=='csi':
                cfg['camera'].update(width=640,height=360,fps=30)
        check_assets(cfg)
        if cfg['runtime']['require_tegrastats'] and shutil.which('tegrastats') is None:
            raise FileNotFoundError('tegrastats is required for whole-system RAM measurements')
        modules = ['cv2','numpy','ultralytics']
        if cfg['inspector']['backend'] == 'transformers':
            modules += ['torch','transformers','PIL']
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError('Missing optional modules: '+', '.join(missing)+'; follow setup instructions without upgrading the existing stack')
    except Exception as exc:
        print(f'Preflight failed: {exc}',file=sys.stderr)
        return 2
    if args.check:
        print('Local preflight passed. Camera, CUDA, server/model compatibility and performance remain untested.')
        return 0
    from .backends import camera_factory, YoloScout, inspector_factory
    from .runtime import Pipeline, render
    run = args.run or 'results/scout_inspector_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    pipeline = Pipeline(cfg,run,camera_factory,YoloScout,inspector_factory,wait_for_start=args.wait_for_start)
    pipeline.initialize()
    pipeline.state.update(source_label=args.source_label,wait_for_start=args.wait_for_start)
    pipeline.checkpoint()
    print(f'Results: {Path(run).resolve()}',flush=True)
    print('Align with source paused. Press S in preview, then start playback; alignment is excluded.' if args.wait_for_start else 'Startup is excluded from duration. Timer begins with first delivered camera frame. Start source playback when preview appears.',flush=True)
    old_handler = signal.signal(signal.SIGTERM,lambda *_:pipeline.request_stop('SIGTERM'))
    last_sequence, last_checkpoint = None, 0
    blocked = []
    try:
        pipeline.start()
        startup_deadline = time.perf_counter()+cfg['runtime']['startup_timeout_seconds']
        alignment_started=None
        armed_seen=not args.wait_for_start
        while not pipeline.stop.is_set():
            now = time.perf_counter()
            if not armed_seen and pipeline.armed.is_set():
                startup_deadline=now+cfg['runtime']['startup_timeout_seconds']
                armed_seen=True
            if pipeline.first_frame.is_set():
                if now-pipeline.first_delivery >= args.duration:
                    pipeline.request_stop('duration')
            elif args.wait_for_start and pipeline.scout_ready.is_set() and not pipeline.armed.is_set():
                alignment_started=alignment_started or now
                if now-alignment_started>300:
                    raise TimeoutError('Alignment exceeded five minutes; partial results preserved')
            elif now >= startup_deadline:
                raise TimeoutError('Startup deadline exceeded before first delivered frame')
            if args.preview:
                last_sequence = render(pipeline,last_sequence)
            if now-last_checkpoint >= 1:
                pipeline.checkpoint()
                last_checkpoint = now
            pipeline.stop.wait(0.01)
    except KeyboardInterrupt:
        pipeline.request_stop('KeyboardInterrupt')
    except Exception:
        pipeline._failed('main')
    finally:
        blocked = pipeline.finish()
        signal.signal(signal.SIGTERM,old_handler)
        if args.preview:
            import cv2
            cv2.destroyAllWindows()
    snapshot = pipeline.state.snapshot()
    print(json.dumps(snapshot,indent=2),flush=True)
    failed = bool(blocked or snapshot['errors'])
    if blocked and cfg['runtime']['force_exit_on_shutdown_timeout']:
        print('Native workers did not stop: '+', '.join(blocked)+'. Partial evidence saved; forcing client exit. Externally managed Inspector server remains running.',file=sys.stderr,flush=True)
        os._exit(2)
    return 1 if failed else 0
