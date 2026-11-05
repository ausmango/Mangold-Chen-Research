"""Live recording with latest-frame, same-image prompt comparison.

One serial model worker. Encode once per selected image, reuse that image's
existing prefix cache for two prompts, discard it before selecting newest frame.
No cross-frame cache reuse. Full recording/preview stay on main thread.
"""
import argparse
import sys
import importlib.metadata
import json
import statistics
import time
import traceback

import moondream_4bit_camera_live as live
from moondream_4bit_camera_fast import PROFILES

PROFILES_USED = ['short32', 'compact16']


def comparison_worker(run, slot, stop, ready, results):
    results.update(profiles={p: PROFILES[p] for p in PROFILES_USED}, pairs=[],
        cache_design='Existing per-image EncodedImage prefix reused for two serial queries; discarded before selecting next image. No cross-frame cache.',
        comparison_warning='Two-prompt benchmarking reduces update rate. Standalone-equivalent = shared encoding + own query; this estimate excludes the other prompt, not actual wall delay. Capture-to-answer is measured actual wall delay.')
    torch = None
    try:
        import cv2
        import torch
        import transformers.modeling_utils as modeling_utils
        from PIL import Image
        from transformers import AutoModelForCausalLM
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results['cuda_free_before_load_gib'] = torch.cuda.mem_get_info()[0]/1024**3
        live.save_json(run/'analysis.json', results)
        started = time.perf_counter()
        print('Loading pinned model for LIVE prompt comparison...', flush=True)
        model = AutoModelForCausalLM.from_pretrained(live.MODEL_ID, revision=live.REVISION,
            trust_remote_code=True, device_map={'': 'cuda'}).eval()
        torch.cuda.synchronize()
        results.update(load_seconds=time.perf_counter()-started,
            model_footprint_gib=model.get_memory_footprint()/1024**3,
            cuda_free_after_load_gib=torch.cuda.mem_get_info()[0]/1024**3,
            cuda_allocated_after_load_gib=torch.cuda.memory_allocated()/1024**3,
            versions={n: importlib.metadata.version(n) for n in ('torch','transformers','torchao','Pillow')},
            status='model_ready')
        live.save_json(run/'analysis.json', results)
        ready.set()
        previous, pair_index = 0, 0
        while not stop.is_set():
            sample = slot.take(previous, stop)
            if sample is None:
                continue
            sequence, captured, captured_utc, frame = sample
            pair = dict(sequence=sequence, pair_index=pair_index, captured_utc=captured_utc,
                path=f'frames/frame_{sequence:06d}.png',
                skipped_sequences_since_previous=sequence-previous-1, status='started')
            previous = sequence
            results['pairs'].append(pair)
            if not cv2.imwrite(str(run/pair['path']), frame):
                raise RuntimeError('Could not save selected frame')
            live.save_json(run/'analysis.json', results)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            torch.cuda.synchronize()
            encode_start_sequence = slot.count()
            encode_start = time.perf_counter()
            with torch.inference_mode():
                encoded = model.encode_image(image)
            torch.cuda.synchronize()
            encode_seconds = time.perf_counter()-encode_start
            pair['image_encode_seconds'] = encode_seconds
            pair['frame_age_at_encode_start_seconds'] = encode_start-captured
            order = PROFILES_USED if pair_index % 2 == 0 else list(reversed(PROFILES_USED))
            for position, profile in enumerate(order):
                # On capture stop, drain the current pair to preserve matched evidence.
                definition = PROFILES[profile]
                row = {**pair, 'profile': profile, 'order_within_pair': position+1,
                    'prompt': definition['prompt'], 'settings': definition['settings'],
                    'status': 'started'}
                results['frames'].append(row)
                live.save_json(run/'analysis.json', results)
                torch.cuda.synchronize()
                query_start = time.perf_counter()
                start_sequence = slot.count()
                with torch.inference_mode():
                    answer = model.query(encoded, definition['prompt'], settings=definition['settings'])['answer']
                torch.cuda.synchronize()
                completed = time.perf_counter()
                # Actual selected-frame-to-answer time includes preceding prompt.
                row.update(status='complete', answer=answer, completed_utc=live.utc(),
                    query_seconds=completed-query_start,
                    standalone_equivalent_inference_seconds=encode_seconds+completed-query_start,
                    inference_seconds=completed-encode_start,
                    inference_seconds_definition='Elapsed from shared image-encoding start; includes preceding prompt for second answer',
                    frame_age_at_inference_start_seconds=encode_start-captured,
                    frame_age_at_query_start_seconds=query_start-captured,
                    capture_to_completed_answer_seconds=completed-captured,
                    capture_sequence_at_inference_start=encode_start_sequence,
                    capture_sequence_at_query_start=start_sequence,
                    capture_sequence_at_completion=slot.count(),
                    cuda_allocated_gib=torch.cuda.memory_allocated()/1024**3,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                results['latest_answer'] = f'{profile}: {answer.strip()}'
                live.save_json(run/'analysis.json', results)
                live.save_json(run/f'frames/frame_{sequence:06d}_{profile}.json', row)
                print(f"LIVE seq={sequence} {profile}: encode={encode_seconds:.3f}s query={completed-query_start:.3f}s actual camera-to-answer={completed-captured:.3f}s | {answer}", flush=True)
            pair.update(status='complete', cycle_seconds=time.perf_counter()-encode_start)
            del encoded
            pair_index += 1
    except BaseException:
        results['worker_error'] = traceback.format_exc()
        print(results['worker_error'], flush=True)
        results['status'] = 'failed'
        stop.set()
        if results['frames'] and results['frames'][-1]['status']=='started':
            results['frames'][-1]['status']='failed'
    finally:
        if torch is not None:
            try:
                results['cuda_peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
            except Exception:
                results['cuda_metrics_error']=traceback.format_exc()
        results['profile_statistics']={}
        for p in PROFILES_USED:
            rows=[r for r in results['frames'] if r.get('profile')==p and r['status']=='complete']
            results['profile_statistics'][p]={'completed':len(rows)}
            for key in ('query_seconds','standalone_equivalent_inference_seconds','capture_to_completed_answer_seconds'):
                vals=[r[key] for r in rows]
                results['profile_statistics'][p][key]={'mean':statistics.mean(vals) if vals else None,
                    'min':min(vals) if vals else None,'max':max(vals) if vals else None}
        try:
            live.save_json(run/'analysis.json', results)
        except Exception:
            results['worker_persistence_error']=traceback.format_exc()
            print(results['worker_persistence_error'], flush=True)
            stop.set()
        finally:
            ready.set()


original_summary = live.summarize


def comparison_summary(run, results):
    original_summary(run, results)
    results['completed_answers'] = results['completed_frames']
    results['completed_frames'] = len({r['sequence'] for r in results['frames'] if r['status']=='complete'})
    results['completed_pairs'] = sum(p['status']=='complete' for p in results.get('pairs', []))
    live.save_json(run/'analysis.json', results)


if __name__=='__main__':
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--candidate-only', action='store_true')
    selected, remaining = options.parse_known_args()
    if selected.candidate_only:
        PROFILES_USED = ['compact16']
    sys.argv = [sys.argv[0]] + remaining
    live.worker=comparison_worker
    live.summarize=comparison_summary
    raise SystemExit(live.main())
