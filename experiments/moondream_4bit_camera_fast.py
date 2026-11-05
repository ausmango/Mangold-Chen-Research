"""Record six samples per video with unchanged v2 capture; paired short-output test.

No concurrent CSI/model allocation. Analyze each image independently with each
profile; no cross-image encoded cache reuse or new model caching changes.
"""
import argparse
import builtins
import csv
from contextlib import contextmanager
from datetime import datetime
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import statistics
import subprocess
import time
import traceback

from moondream_4bit_camera_live import MODEL_ID, REVISION, PROMPT, save_json, utc

PROFILES = {
    'short32': {'prompt': PROMPT, 'settings': {'max_tokens': 32, 'temperature': 0}},
    'compact16': {
        'prompt': "Name the clearest visible construction hazard in at most six words. If unclear, say 'none clear'. Do not infer motion or hidden hazards.",
        'settings': {'max_tokens': 16, 'temperature': 0},
    },
    'flag4': {
        'prompt': 'Is a construction hazard clearly visible? Answer only YES, NO, or UNCLEAR. Do not infer movement or hidden hazards.',
        'settings': {'max_tokens': 4, 'temperature': 0},
    },
}


def profile_order(selected, index):
    offset = index % len(selected)
    order = selected[offset:] + selected[:offset]
    return order if index % 2 == 0 else list(reversed(order))


@contextmanager
def telemetry(run, phase):
    path = run / f'{phase}_tegrastats.txt'
    if path.exists():
        raise RuntimeError(f'Preserve existing telemetry: {path}')
    process = None
    try:
        with path.open('x') as stream:
            process = subprocess.Popen(['tegrastats', '--interval', '500'],
                                       stdout=stream, stderr=subprocess.STDOUT)
        yield
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def ram_summary(path):
    text = path.read_text(errors='replace') if path.exists() else ''
    samples = [(int(a), int(b)) for a, b in re.findall(r'RAM (\d+)/(\d+)MB', text)]
    return {'samples': len(samples), 'peak_used_mb': max((a for a, _ in samples), default=None),
            'total_mb': samples[-1][1] if samples else None,
            'error': None if samples else 'No RAM samples; inspect telemetry log'}


def summarize(results):
    stats = {}
    for profile in results['profiles']:
        rows = [r for r in results['frames'] if r['profile'] == profile and r['status'] == 'complete']
        stats[profile] = {'completed': len(rows), 'videos': {}}
        for key in ('inference_seconds', 'image_encode_seconds', 'query_seconds',
                    'frame_age_at_inference_start_seconds', 'capture_to_completed_answer_seconds'):
            values = [r[key] for r in rows]
            stats[profile][key] = {
                'mean': statistics.mean(values) if values else None,
                'min': min(values) if values else None, 'max': max(values) if values else None,
                'total': sum(values),
            }
        # First image gets first-use effects; retain all results and a separate exclusion.
        remaining = [r['inference_seconds'] for r in rows if r['pair_index'] > 0]
        stats[profile]['mean_excluding_first_image_seconds'] = statistics.mean(remaining) if remaining else None
        for number in sorted({r['video_number'] for r in rows}):
            times = [r['inference_seconds'] for r in rows if r['video_number'] == number]
            stats[profile]['videos'][str(number)] = {'completed': len(times), 'mean_seconds': statistics.mean(times)}
    results['statistics'] = stats
    # Compare only complete pairs, so interrupted runs do not compare different images.
    pairs = {}
    for row in results['frames']:
        if row['status'] == 'complete':
            pairs.setdefault(row['pair_index'], {})[row['profile']] = row['inference_seconds']
    matched = [p for p in pairs.values() if 'short32' in p and 'compact16' in p]
    results['comparisons_vs_short32'] = {}
    for candidate in results['profiles']:
        if candidate == 'short32':
            continue
        comparable = [p for p in pairs.values() if 'short32' in p and candidate in p]
        if comparable:
            ref = statistics.mean(p['short32'] for p in comparable)
            new = statistics.mean(p[candidate] for p in comparable)
            results['comparisons_vs_short32'][candidate] = {
                'complete_pairs': len(comparable), 'reference_mean_seconds': ref,
                'candidate_mean_seconds': new, 'latency_reduction_percent': 100*(1-new/ref),
                'speed_ratio': ref/new,
            }
    results['paired_comparison'] = {'complete_pairs': len(matched)}
    if matched:
        reference = statistics.mean(p['short32'] for p in matched)
        candidate = statistics.mean(p['compact16'] for p in matched)
        results['paired_comparison'].update(
            short32_mean_seconds=reference, compact16_mean_seconds=candidate,
            latency_reduction_percent=100 * (1 - candidate / reference),
            speed_ratio=reference / candidate,
        )
        nonfirst = [p for i, p in pairs.items() if i > 0 and 'short32' in p and 'compact16' in p]
        if nonfirst:
            warm_ref = statistics.mean(p['short32'] for p in nonfirst)
            warm_new = statistics.mean(p['compact16'] for p in nonfirst)
            results['paired_comparison']['excluding_first_image'] = {
                'pairs': len(nonfirst), 'short32_mean_seconds': warm_ref,
                'compact16_mean_seconds': warm_new,
                'latency_reduction_percent': 100 * (1 - warm_new / warm_ref),
            }


def analyze(run, selected):
    # Validate all inputs before importing the CUDA stack.
    manifest = json.loads((run / 'capture.json').read_text())
    samples = [(video, sample) for video in manifest['videos'] for sample in video['frames']]
    if not samples:
        raise ValueError('No captured frames to analyze')
    for _, sample in samples:
        if not (run / sample['path']).is_file():
            raise FileNotFoundError(run / sample['path'])
    path = run / 'analysis_fast.json'
    if path.exists():
        raise RuntimeError('Analysis already exists; preserve partial results and use a new run')
    results = {
        'created_utc': utc(), 'status': 'initializing', 'model': MODEL_ID, 'revision': REVISION,
        'design': 'Sequential six-frame-per-video capture followed by paired independent-image queries',
        'profiles': {p: PROFILES[p] for p in selected}, 'requested_images': len(samples),
        'requested_queries': len(samples) * len(selected), 'frames': [],
        'limitations': 'Generated claims are unverified. Not temporal event detection or real-time inference. Capture-to-answer includes offline recording/waiting. Retokenized answer length is not exact generation-token instrumentation.',
        'timing': 'Synchronized encode plus query = inference; query receives the encoded image. Each profile re-encodes independently. No disk writes in inference interval. Rotating/reversing profile order, one model, one serial query at a time.',
    }
    results['capture_statistics'] = []
    for video in manifest['videos']:
        timestamps_path = run / Path(video['frames'][0]['path']).parent / 'recording_timestamps.csv' if video['frames'] else None
        times = []
        if timestamps_path is not None and timestamps_path.exists():
            with timestamps_path.open() as stream:
                times = [float(row['elapsed_seconds']) for row in csv.DictReader(stream)]
        span = times[-1] - times[0] if len(times) > 1 else 0
        results['capture_statistics'].append({
            'video_number': video['number'], 'label': video['label'],
            'saved_samples': len(video['frames']), 'recorded_frames': len(times),
            'measured_delivery_fps': (len(times)-1)/span if span > 0 else None,
            'max_delivery_gap_seconds': max((b-a for a,b in zip(times, times[1:])), default=None),
        })
    save_json(path, results)
    torch = None
    try:
        with telemetry(run, 'analysis_fast'):
            import torch
            from PIL import Image
            import transformers.modeling_utils as modeling_utils
            from transformers import AutoModelForCausalLM
            logging.getLogger('pyvips').setLevel(logging.WARNING)
            modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            results['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'torchao', 'Pillow')}
            results['cuda_free_before_load_gib'] = torch.cuda.mem_get_info()[0] / 1024**3
            save_json(path, results)
            print('Loading pinned official 4-bit model (camera must be closed)...', flush=True)
            started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=True, device_map={'': 'cuda'},
            ).eval()
            torch.cuda.synchronize()
            results.update(load_seconds=time.perf_counter() - started,
                           model_footprint_gib=model.get_memory_footprint() / 1024**3,
                           cuda_free_after_load_gib=torch.cuda.mem_get_info()[0] / 1024**3,
                           cuda_allocated_after_load_gib=torch.cuda.memory_allocated() / 1024**3,
                           status='running')
            save_json(path, results)
            for index, (video, sample) in enumerate(samples):
                with Image.open(run / sample['path']) as opened:
                    image = opened.convert('RGB')
                order = profile_order(selected, index)
                for profile in order:
                    definition = PROFILES[profile]
                    row = {**sample, 'pair_index': index, 'profile': profile,
                           'video_number': video['number'], 'video_label': video['label'],
                           'prompt': definition['prompt'], 'settings': definition['settings'],
                           'status': 'started', 'order_within_pair': order.index(profile) + 1}
                    results['frames'].append(row)
                    save_json(path, results)
                    torch.cuda.synchronize()
                    started_utc = datetime.fromisoformat(utc())
                    start = time.perf_counter()
                    with torch.inference_mode():
                        encoded = model.encode_image(image)
                        torch.cuda.synchronize()
                        encoded_at = time.perf_counter()
                        answer = model.query(encoded, definition['prompt'], settings=definition['settings'])['answer']
                        torch.cuda.synchronize()
                        completed = time.perf_counter()
                    del encoded
                    age = (started_utc - datetime.fromisoformat(sample['captured_utc'])).total_seconds()
                    tokenizer = getattr(model, 'tokenizer', None)
                    if tokenizer is None:
                        tokenizer = model.model.tokenizer
                    count = len(tokenizer.encode(answer).ids)
                    row.update(status='complete', answer=answer,
                               inference_started_utc=started_utc.isoformat(), completed_utc=utc(),
                               image_encode_seconds=encoded_at-start, query_seconds=completed-encoded_at,
                               inference_seconds=completed-start,
                               frame_age_at_inference_start_seconds=age,
                               capture_to_completed_answer_seconds=age + completed-start,
                               answer_retokenized_tokens=count,
                               possible_token_cap_reached=count >= definition['settings']['max_tokens'],
                               cuda_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                               cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
                    save_json(path, results)
                    sidecar = run / Path(sample['path']).with_name(Path(sample['path']).stem + f'_{profile}.json')
                    save_json(sidecar, row)
                    print(f"Video {video['number']} frame {sample['number']} {profile}: {completed-start:.3f}s (encode {encoded_at-start:.3f}s, query {completed-encoded_at:.3f}s) | {answer}", flush=True)
            results['status'] = 'finished'
    except KeyboardInterrupt:
        results['status'] = 'interrupted'
    except BaseException:
        results['status'] = 'failed'
        results['error'] = traceback.format_exc()
        print(results['error'], flush=True)
    finally:
        if results['frames'] and results['frames'][-1]['status'] == 'started':
            results['frames'][-1]['status'] = 'failed' if results['status'] == 'failed' else 'interrupted'
        if torch is not None:
            try:
                results['cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
            except Exception:
                results['cuda_metric_error'] = traceback.format_exc()
        results['system_ram'] = ram_summary(run / 'analysis_fast_tegrastats.txt')
        results['finished_utc'] = utc()
        summarize(results)
        save_json(path, results)
        print(json.dumps({'status': results['status'], 'comparison': results['comparisons_vs_short32'], 'statistics': results['statistics'], 'system_ram': results['system_ram']}, indent=2), flush=True)
        print(f'All answers and metrics: {path}', flush=True)
    return 0 if results['status'] == 'finished' else 1


def capture_presets(run):
    """Same five known clips; no terminal metadata entry during capture."""
    from moondream_4bit_camera_v2 import capture
    source = Path(__file__).resolve().parents[1] / 'results/camera_v2-2~2026-10-05/capture.json'
    videos = json.loads(source.read_text())['videos']
    if len(videos) != 5:
        raise ValueError('Expected five baseline video presets')
    responses = iter(value for video in videos for value in (
        video['label'], video.get('source', ''), str(video['duration_requested_seconds'])))
    original_input = builtins.input
    def automatic_input(prompt):
        value = next(responses)
        print(f'{prompt}{value} [automatic preset]', flush=True)
        return value
    try:
        builtins.input = automatic_input
        capture(run)
    finally:
        builtins.input = original_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'analyze'])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error('Profiles must be unique')
    run = args.run.resolve()
    if args.mode == 'analyze':
        return analyze(run, args.profiles)
    run.mkdir(parents=True, exist_ok=False)
    try:
        with telemetry(run, 'capture'):
            capture_presets(run)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
