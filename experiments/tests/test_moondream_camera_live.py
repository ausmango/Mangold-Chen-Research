"""CPU-only concurrency/failure checks. No camera, CUDA, or model imports."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import moondream_4bit_camera_live as live


class WorkerTests(unittest.TestCase):
    def exercise(self, fail=False):
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        stop, ready = threading.Event(), threading.Event()
        slot = live.LatestFrame()
        calls = []

        def query(image, prompt, settings):
            calls.append(image)
            if len(calls) == 1:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test timed out')
                if fail:
                    raise RuntimeError('synthetic query failure')
            else:
                second.set()
                stop.set()
            return {'answer': 'Synthetic answer'}

        cuda = types.SimpleNamespace(
            empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
            mem_get_info=lambda: (5 * 1024**3, 8 * 1024**3),
            synchronize=lambda: None, memory_allocated=lambda: 1024**3,
            max_memory_allocated=lambda: 2 * 1024**3,
        )
        torch = types.ModuleType('torch')
        torch.cuda, torch.inference_mode = cuda, contextlib.nullcontext
        modeling = types.ModuleType('transformers.modeling_utils')
        transformers = types.ModuleType('transformers')
        transformers.modeling_utils = modeling
        model = types.SimpleNamespace(eval=lambda: model,
                                      get_memory_footprint=lambda: 1024**3,
                                      query=query)
        transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **kw: model)
        cv2 = types.ModuleType('cv2')
        def write(path, frame):
            Path(path).write_bytes(b'synthetic frame')
            return True
        cv2.imwrite, cv2.cvtColor, cv2.COLOR_BGR2RGB = write, lambda f, c: f, 1
        pil = types.ModuleType('PIL')
        pil.Image = types.SimpleNamespace(fromarray=lambda f: f)
        modules = {'torch': torch, 'transformers': transformers,
                   'transformers.modeling_utils': modeling, 'cv2': cv2, 'PIL': pil}
        results = {'frames': []}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules), \
                patch.object(live.importlib.metadata, 'version', return_value='synthetic'), \
                contextlib.redirect_stdout(io.StringIO()):
            run = Path(folder)
            (run / 'frames').mkdir()
            thread = threading.Thread(target=live.worker,
                                      args=(run, slot, stop, ready, results))
            thread.start()
            try:
                self.assertTrue(ready.wait(3))
                slot.publish('first', time.perf_counter(), live.utc())
                self.assertTrue(entered.wait(3))
                # Producer continues during blocked inference. Only newest survives.
                for i in range(2, 101):
                    slot.publish(f'frame-{i}', time.perf_counter(), live.utc())
                self.assertEqual(slot.count(), 100)
                # Frame and metadata exist before a query completes.
                pending = json.loads((run / 'analysis.json').read_text())
                self.assertEqual(pending['frames'][0]['status'], 'started')
                self.assertTrue((run / pending['frames'][0]['path']).exists())
                release.set()
                thread.join(3)
                self.assertFalse(thread.is_alive())
                saved = json.loads((run / 'analysis.json').read_text())
                if fail:
                    self.assertIn('synthetic query failure', saved['worker_error'])
                    self.assertEqual(saved['frames'][0]['status'], 'failed')
                    self.assertTrue(stop.is_set())
                else:
                    self.assertTrue(second.is_set())
                    self.assertEqual(calls, ['first', 'frame-100'])
                    self.assertEqual([r['sequence'] for r in saved['frames']], [1, 100])
                    self.assertEqual(saved['frames'][1]['skipped_sequences_since_previous'], 98)
                    for row in saved['frames']:
                        self.assertEqual(row['prompt'], live.PROMPT)
                        self.assertEqual(row['settings'], live.SETTINGS)
                        self.assertEqual(row['revision'], live.REVISION)
                        self.assertAlmostEqual(
                            row['capture_to_completed_answer_seconds'],
                            row['frame_age_at_inference_start_seconds'] + row['inference_seconds'])
                        self.assertTrue((run / Path(row['path']).with_suffix('.json')).exists())
            finally:
                stop.set()
                release.set()
                with slot.condition:
                    slot.condition.notify_all()
                thread.join(3)

    def test_latest_frame_and_timings(self):
        self.exercise()

    def test_worker_failure_preserves_frame(self):
        self.exercise(fail=True)

    def test_tegrastats_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'tegrastats.txt').write_text('RAM 5400/7620MB\nRAM 5500/7620MB\n')
            (run / 'capture_timestamps.csv').write_text(
                'sequence,captured_utc,monotonic_seconds,read_seconds\n'
                '1,synthetic,10,0.01\n2,synthetic,10.04,0.02\n3,synthetic,10.10,0.03\n')
            results = {'frames': []}
            live.summarize(run, results)
            self.assertAlmostEqual(results['camera_capture']['measured_fps'], 20)
            self.assertAlmostEqual(results['camera_capture']['max_delivery_gap_seconds'], 0.06)
            self.assertAlmostEqual(results['camera_capture']['mean_read_seconds'], 0.02)
            self.assertEqual(results['system_ram']['peak_used_mb'], 5500)
            self.assertIsNone(results['summary']['inference_seconds']['mean'])


class MainFailureTests(unittest.TestCase):
    def test_initialization_failure_returns_failure_after_cleanup(self):
        cv2 = types.ModuleType('cv2')
        cv2.destroyAllWindows = lambda: None
        verified = []
        def first_read():
            verified.append(True)
            return True, object()
        cv2.CAP_GSTREAMER = 0
        cv2.VideoCapture = lambda *a: types.SimpleNamespace(
            isOpened=lambda: True, read=first_read, release=lambda: None)

        def failed_worker(run, slot, stop, ready, results):
            self.assertTrue(verified, 'camera must deliver before model loading')
            results['worker_error'] = 'synthetic load failure'
            stop.set()
            ready.set()

        telemetry = types.SimpleNamespace(terminate=lambda: None, wait=lambda **kw: None)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'cv2': cv2}), \
                patch.object(live, 'worker', failed_worker), \
                patch.object(live.subprocess, 'Popen', return_value=telemetry), \
                patch.object(live.signal, 'signal'), \
                patch.object(sys, 'argv', ['live', '--run', str(Path(folder) / 'run')]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(live.main(), 1)
            saved = json.loads((Path(folder) / 'run/analysis.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('synthetic load failure', saved['worker_error'])
            self.assertEqual(saved['completed_frames'], 0)

    def test_camera_failure_does_not_load_model(self):
        cv2 = types.ModuleType('cv2')
        cv2.CAP_GSTREAMER = 0
        cv2.destroyAllWindows = lambda: None
        cv2.VideoCapture = lambda *a: types.SimpleNamespace(
            isOpened=lambda: True, read=lambda: (False, None), release=lambda: None)
        telemetry = types.SimpleNamespace(terminate=lambda: None, wait=lambda **kw: None)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'cv2': cv2}), \
                patch.object(live, 'worker') as model_worker, \
                patch.object(live.subprocess, 'Popen', return_value=telemetry), \
                patch.object(live.signal, 'signal'), \
                patch.object(sys, 'argv', ['live', '--run', str(Path(folder) / 'run')]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(live.main(), 1)
            model_worker.assert_not_called()
            saved = json.loads((Path(folder) / 'run/analysis.json').read_text())
            self.assertIn('no frame before model loading', saved['main_error'])

    def test_cleanup_and_missing_telemetry_preserve_failure(self):
        cv2 = types.ModuleType('cv2')
        def failed_cleanup():
            raise RuntimeError('synthetic window cleanup failure')
        cv2.destroyAllWindows = failed_cleanup
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'cv2': cv2}), \
                patch.object(live.subprocess, 'Popen', side_effect=FileNotFoundError('tegrastats')), \
                patch.object(live.signal, 'signal'), \
                patch.object(sys, 'argv', ['live', '--run', str(Path(folder) / 'run')]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(live.main(), 1)
            saved = json.loads((Path(folder) / 'run/analysis.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('cleanup_errors', saved)
            self.assertEqual(saved['system_ram']['samples'], 0)
            self.assertIsNone(saved['system_ram']['peak_used_mb'])


class CameraOnlyTests(unittest.TestCase):
    def test_alignment_and_recording_without_model(self):
        class Frame:
            shape = (360, 640, 3)
            def copy(self):
                return self
        frame = Frame()
        calls = []
        camera = types.SimpleNamespace(
            isOpened=lambda: True, read=lambda: (True, frame),
            release=lambda: calls.append('camera_release'))
        writer = types.SimpleNamespace(
            isOpened=lambda: True, write=lambda f: calls.append('write'),
            release=lambda: calls.append('writer_release'))
        cv2 = types.ModuleType('cv2')
        cv2.VideoCapture = lambda *a: camera
        cv2.VideoWriter = lambda *a: writer
        cv2.VideoWriter_fourcc = lambda *a: 0
        cv2.CAP_GSTREAMER, cv2.FONT_HERSHEY_SIMPLEX = 0, 0
        cv2.putText = lambda *a: None
        cv2.imshow = lambda *a: None
        keys = iter([ord('s'), 0, ord('q')])
        cv2.waitKey = lambda *a: next(keys)
        cv2.destroyAllWindows = lambda: calls.append('windows_closed')
        def save_image(path, image):
            Path(path).write_bytes(b'synthetic')
            return True
        cv2.imwrite = save_image
        telemetry = types.SimpleNamespace(terminate=lambda: None, wait=lambda **kw: None)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'cv2': cv2}), \
                patch.object(live, 'worker', side_effect=AssertionError('model worker must not run')), \
                patch.object(live.subprocess, 'Popen', return_value=telemetry), \
                patch.object(live.signal, 'signal'), \
                patch.object(sys, 'argv', ['live', '--run', str(Path(folder) / 'run'),
                                          '--camera-only', '--wait-for-start']), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(live.main(), 0)
            saved = json.loads((Path(folder) / 'run/analysis.json').read_text())
            self.assertEqual(saved['mode'], 'camera_only')
            self.assertEqual(saved['camera_capture']['delivered_frames'], 2)
            self.assertEqual(saved['recording_statistics']['submitted_frames'], 2)
            self.assertEqual(saved['stop_reason'], 'preview_q')
            self.assertEqual(saved['frames'], [])
            self.assertNotIn('load_seconds', saved)
            self.assertEqual(calls.count('write'), 2)
            self.assertIn('camera_release', calls)
            self.assertIn('writer_release', calls)
            self.assertTrue((Path(folder) / 'run/setup_first_frame.png').exists())



if __name__ == '__main__':
    unittest.main()
