"""Three saved Red Machine images, today's prompt; no camera access.

Launch under timeout --kill-after=5s 180s for a hard external deadline.
Reuses the established v2 loader/query path without modifying that script.
"""
import argparse
import json
import re
import shutil
import statistics
import subprocess
import traceback
from pathlib import Path

from moondream_4bit_camera_live import MODEL_ID, REVISION, PROMPT, SETTINGS, save_json, utc

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'results/camera_v2-2~2026-10-05'


def prepare(run):
    manifest = json.loads((BASELINE / 'capture.json').read_text())
    video = manifest['videos'][0]
    run.mkdir(parents=True, exist_ok=False)
    (run / 'frames').mkdir()
    samples = []
    for index in (0, 2, 4):
        original = video['frames'][index]
        sample = dict(original)
        destination = Path('frames') / Path(original['path']).name
        shutil.copy2(BASELINE / original['path'], run / destination)
        sample.update(path=str(destination), source_path=str(BASELINE / original['path']))
        samples.append(sample)
    save_json(run / 'capture.json', {'videos': [{**video, 'frames': samples}]})
    metadata = {
        'created_utc': utc(), 'mode': 'saved_image_control_no_camera',
        'status': 'prepared', 'model': MODEL_ID, 'revision': REVISION,
        'prompt': PROMPT, 'settings': SETTINGS, 'requested_frames': 3,
        'source_baseline': str(BASELINE),
        'capture_to_answer_seconds': None, 'frame_age_seconds': None,
        'limitation': 'Historical images, not live capture or temporal event detection. Original capture timestamps/elapsed times are provenance only. Generated hazard claims are unverified.',
    }
    save_json(run / 'control.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    metadata = prepare(run)
    telemetry = None
    try:
        with (run / 'tegrastats.txt').open('w') as stream:
            telemetry = subprocess.Popen(['tegrastats', '--interval', '500'],
                                         stdout=stream, stderr=subprocess.STDOUT)
        import moondream_4bit_camera_v2 as baseline
        baseline.PROMPT, baseline.SETTINGS = PROMPT, dict(SETTINGS)
        metadata['status'] = 'running'
        save_json(run / 'control.json', metadata)
        baseline.analyze(run)
        metadata['status'] = 'finished'
    except BaseException:
        metadata['status'] = 'failed'
        metadata['error'] = traceback.format_exc()
        print(metadata['error'], flush=True)
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=3)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        analysis_path = run / 'analysis.json'
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
            rows = analysis.get('frames', [])
            times = [r['inference_seconds'] for r in rows]
            metadata.update(
                completed_frames=len(rows),
                mean_inference_seconds=statistics.mean(times) if times else None,
                load_seconds=analysis.get('load_seconds'),
                model_footprint_gib=analysis.get('model_footprint_gib'),
                cuda_peak_allocated_gib=analysis.get('cuda_peak_allocated_gib'),
            )
            report = [
                '# Saved-image control (no live camera)',
                f'Completed: {len(rows)}/3', f'Prompt: {PROMPT}',
                f'Settings: {SETTINGS}',
                'Historical capture times are provenance, not live response delays.',
                'Generated claims are unverified; no temporal event detection.', '',
            ]
            for row in rows:
                report.append(f"- {row['path']}: {row['description']} (inference {row['inference_seconds']:.4f} s)")
            (run / 'report.md').write_text('\n'.join(report), encoding='utf-8')
            if analysis.get('status') != 'finished' or len(rows) != 3:
                metadata['status'] = 'incomplete'
        text = (run / 'tegrastats.txt').read_text(errors='replace') if (run / 'tegrastats.txt').exists() else ''
        samples = [int(a) for a in re.findall(r'RAM (\d+)/\d+MB', text)]
        metadata['ram_samples'] = len(samples)
        metadata['peak_system_ram_mb'] = max(samples, default=None)
        metadata['finished_utc'] = utc()
        save_json(run / 'control.json', metadata)
        print(json.dumps(metadata, indent=2), flush=True)
    return 0 if metadata['status'] == 'finished' else 1


if __name__ == '__main__':
    raise SystemExit(main())
