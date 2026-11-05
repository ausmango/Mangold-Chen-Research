"""CPU-only paired benchmark checks; no camera or CUDA imports."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import moondream_4bit_camera_fast as fast


class FastTests(unittest.TestCase):
    def test_balanced_order(self):
        names = list(fast.PROFILES)
        orders = [tuple(fast.profile_order(names, i)) for i in range(6)]
        self.assertEqual(len(set(orders)), 6)
        for name in names:
            for position in range(3):
                self.assertEqual(sum(order[position] == name for order in orders), 2)

    def test_incomplete_pairs_not_compared(self):
        results = {'profiles': list(fast.PROFILES), 'frames': []}
        for pair, profile, latency in [(0, 'short32', 10), (0, 'compact16', 5), (1, 'short32', 100)]:
            results['frames'].append(dict(pair_index=pair, profile=profile,
                inference_seconds=latency, image_encode_seconds=2, query_seconds=latency-2,
                frame_age_at_inference_start_seconds=20, capture_to_completed_answer_seconds=20+latency,
                status='complete', video_number=1))
        fast.summarize(results)
        self.assertEqual(results['paired_comparison']['complete_pairs'], 1)
        self.assertEqual(results['paired_comparison']['latency_reduction_percent'], 50)

    def exercise_analysis(self, fail=False):
        calls, encodes = [], []
        cuda = types.SimpleNamespace(empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
            mem_get_info=lambda: (5*1024**3, 8*1024**3), synchronize=lambda: None,
            memory_allocated=lambda: 2*1024**3, max_memory_allocated=lambda: 3*1024**3)
        torch = types.ModuleType('torch')
        torch.cuda, torch.inference_mode = cuda, contextlib.nullcontext
        modeling = types.ModuleType('transformers.modeling_utils')
        transformers = types.ModuleType('transformers')
        transformers.modeling_utils = modeling
        def encode(image):
            marker = object()
            encodes.append(marker)
            return marker
        def query(encoded, prompt, settings):
            self.assertIs(encoded, encodes[-1])
            calls.append(settings['max_tokens'])
            if fail and len(calls) == 2:
                raise RuntimeError('synthetic query failure')
            return {'answer': 'Uneven ground'}
        model = types.SimpleNamespace(encode_image=encode, query=query,
            model=types.SimpleNamespace(tokenizer=types.SimpleNamespace(
                encode=lambda answer: types.SimpleNamespace(ids=[1, 2]))),
            get_memory_footprint=lambda: 1024**3)
        model.eval = lambda: model
        transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **kw: model)
        pil = types.ModuleType('PIL')
        @contextlib.contextmanager
        def image_open(path):
            yield types.SimpleNamespace(convert=lambda mode: object())
        pil.Image = types.SimpleNamespace(open=image_open)
        @contextlib.contextmanager
        def fake_telemetry(run, phase):
            (run / f'{phase}_tegrastats.txt').write_text('RAM 5000/7620MB\n')
            yield
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            frames = []
            for i in range(2):
                path = f'frame_{i}.jpg'
                (run/path).write_bytes(b'synthetic')
                frames.append(dict(number=i+1, path=path, captured_utc='2026-10-22T00:00:00+00:00'))
            fast.save_json(run/'capture.json', {'videos': [dict(number=1, label='test', frames=frames)]})
            modules = {'torch': torch, 'transformers': transformers,
                       'transformers.modeling_utils': modeling, 'PIL': pil}
            with patch.dict(sys.modules, modules), patch.object(fast, 'telemetry', fake_telemetry), \
                    patch.object(fast.importlib.metadata, 'version', return_value='synthetic'), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = fast.analyze(run, list(fast.PROFILES))
            saved = json.loads((run/'analysis_fast.json').read_text())
            self.assertEqual(saved['system_ram']['peak_used_mb'], 5000)
            if fail:
                self.assertEqual(status, 1)
                self.assertEqual(saved['frames'][0]['status'], 'complete')
                self.assertEqual(saved['frames'][1]['status'], 'failed')
                self.assertIn('synthetic query failure', saved['error'])
            else:
                self.assertEqual(status, 0)
                self.assertEqual(len(encodes), 6)  # Re-encode independently per profile.
                self.assertEqual(calls, [32, 16, 4, 32, 4, 16])
                for row in saved['frames']:
                    self.assertAlmostEqual(row['inference_seconds'], row['image_encode_seconds']+row['query_seconds'])
                    self.assertAlmostEqual(row['capture_to_completed_answer_seconds'], row['frame_age_at_inference_start_seconds']+row['inference_seconds'], places=5)
                    self.assertTrue((run / f"frame_{row['number']-1}_{row['profile']}.json").exists())

    def test_profiles_persistence_and_timing(self):
        self.exercise_analysis()

    def test_failure_preserves_completed_answer(self):
        self.exercise_analysis(fail=True)


if __name__ == '__main__':
    unittest.main()
