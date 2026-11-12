"""CPU-only concurrency contracts; no camera, CUDA or model imports."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.scout_inspector.config import DEFAULTS, load_config
from src.scout_inspector.core import Detection, LatestQueue, TriggerRules
from src.scout_inspector.runtime import Pipeline, telemetry_summary


class FakeFrame:
    shape = (480,640,3)


class Camera:
    def __init__(self, cfg):
        self.closed = False
    def open(self):
        return {'fake':True}
    def read(self):
        time.sleep(0.002)
        return FakeFrame()
    def close(self):
        self.closed = True


class Scout:
    labels = ['person','car','truck','bus']
    def __init__(self,cfg):
        pass
    def load(self):
        return {'fake':True}
    def predict(self,frame):
        return [Detection('person',.9,(.1,.1,.4,.9)),Detection('truck',.9,(.4,.2,.9,.9))]
    def close(self):
        pass


class Inspector:
    def __init__(self,cfg):
        self.active = self.peak = 0
        self.entered = threading.Event()
        self.release = threading.Event()
    def load(self):
        return {'fake':True}
    def infer(self,image,prompt,stop):
        self.active += 1
        self.peak = max(self.peak,self.active)
        self.entered.set()
        self.release.wait(1)
        self.active -= 1
        return {'answer':'Unverified observation','cancelled':False}
    def memory_stats(self):
        return {'fake':True}
    def close(self):
        pass


class Contracts(unittest.TestCase):
    def test_single_latest_slot(self):
        bridge = LatestQueue()
        for i in range(1000):
            self.assertEqual(bridge.replace(i),int(i>0))
        self.assertEqual(bridge.queue.maxsize,1)
        self.assertEqual(bridge.take(),999)
        bridge.done()
        self.assertEqual(bridge.queue.unfinished_tasks,0)

    def test_missing_ppe_labels_rejected(self):
        cfg = copy.deepcopy(DEFAULTS['triggers'])
        cfg['missing_helmet_enabled'] = True
        with self.assertRaises(ValueError):
            TriggerRules(cfg).validate_labels(Scout.labels)

    def test_nominal_scene_no_trigger(self):
        self.assertFalse(TriggerRules(DEFAULTS['triggers']).evaluate([],time.perf_counter(),480))

    def test_configuration_requires_revision(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'config.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(json.dumps({'inspector':{'revision':'a'*40}}))
            self.assertEqual(load_config(path)['inspector']['image_size'],384)

    def test_capture_continues_during_serial_inspection(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg['triggers']['min_trigger_interval_ms'] = 0
        camera,inspector = Camera({}),Inspector({})
        with tempfile.TemporaryDirectory() as root:
            pipeline = Pipeline(cfg,Path(root)/'run',lambda _:camera,Scout,lambda _:inspector,
                                prepare=lambda frame,size:(b'fake',{}),
                                save_image=lambda path,image:path.write_bytes(image))
            pipeline.initialize()
            pipeline.start(use_tegrastats=False)
            try:
                self.assertTrue(inspector.entered.wait(1))
                before = pipeline.state.snapshot()['counters']['captured_frames']
                time.sleep(.08)
                after = pipeline.state.snapshot()['counters']['captured_frames']
                self.assertGreater(after,before+5)
                self.assertEqual(pipeline.bridge.queue.qsize(),1)
                self.assertGreater(pipeline.state.snapshot()['counters']['replaced_pending_frames'],0)
            finally:
                pipeline.request_stop('test')
                inspector.release.set()
                self.assertEqual(pipeline.finish(),[])
            self.assertEqual(inspector.peak,1)
            self.assertTrue(camera.closed)
            rows = [json.loads(line) for line in (pipeline.run/'inspections.jsonl').read_text().splitlines()]
            complete = [row for row in rows if row['status']=='complete']
            self.assertTrue(complete)
            row = complete[0]
            for key in ['prompt','settings','captured_utc','frame_age_at_inference_start_ms','inference_ms','capture_to_completed_answer_ms']:
                self.assertIn(key,row)
            self.assertTrue((pipeline.run/row['frame_path']).exists())
            from experiments.summarize_scout_inspector import summarize
            report = summarize(pipeline.run)
            self.assertEqual(report['completed_answers'],1)
            self.assertGreaterEqual(report['answers_with_capture_overlap'],1)

    def test_alignment_excludes_inference_and_timer_until_armed(self):
        cfg = copy.deepcopy(DEFAULTS)
        inspector = Inspector({})
        with tempfile.TemporaryDirectory() as root:
            pipeline = Pipeline(cfg,Path(root)/'run',Camera,Scout,lambda _:inspector,
                                prepare=lambda frame,size:(b'fake',{}),
                                save_image=lambda path,image:path.write_bytes(image),wait_for_start=True)
            pipeline.initialize()
            pipeline.start(use_tegrastats=False)
            try:
                deadline=time.perf_counter()+1
                while pipeline.state.snapshot()['counters'].get('alignment_frames',0)<3 and time.perf_counter()<deadline:
                    time.sleep(.01)
                self.assertGreater(pipeline.state.snapshot()['counters'].get('alignment_frames',0),0)
                self.assertFalse(pipeline.first_frame.is_set())
                self.assertFalse(inspector.entered.is_set())
                self.assertEqual(pipeline.state.snapshot()['counters'].get('captured_frames',0),0)
                pipeline.arm()
                self.assertTrue(pipeline.first_frame.wait(1))
                self.assertTrue(inspector.entered.wait(1))
            finally:
                pipeline.request_stop('test')
                inspector.release.set()
                self.assertEqual(pipeline.finish(),[])

    def test_worker_failure_preserves_partial_run(self):
        class Broken(Scout):
            def load(self):
                raise RuntimeError('intentional fake engine failure')
        with tempfile.TemporaryDirectory() as root:
            pipeline = Pipeline(copy.deepcopy(DEFAULTS),Path(root)/'run',Camera,Broken,Inspector)
            pipeline.initialize()
            pipeline.start(use_tegrastats=False)
            self.assertTrue(pipeline.stop.wait(1))
            self.assertEqual(pipeline.finish(),[])
            saved = json.loads((pipeline.run/'run.json').read_text())
            self.assertEqual(saved['metadata']['status'],'failed')
            self.assertIn('intentional fake engine failure',str(saved['errors']))

    def test_stale_frames_do_not_reach_model(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg['inspector']['max_frame_age_ms'] = .01
        inspector = Inspector({})
        with tempfile.TemporaryDirectory() as root:
            pipeline = Pipeline(cfg,Path(root)/'run',Camera,Scout,lambda _:inspector,
                                prepare=lambda frame,size:(b'fake',{}),
                                save_image=lambda path,image:path.write_bytes(image))
            pipeline.initialize()
            pipeline.start(use_tegrastats=False)
            try:
                deadline = time.perf_counter()+1
                while pipeline.state.snapshot()['counters'].get('stale_frames_discarded',0)==0 and time.perf_counter()<deadline:
                    time.sleep(.01)
                self.assertGreater(pipeline.state.snapshot()['counters'].get('stale_frames_discarded',0),0)
                self.assertFalse(inspector.entered.is_set())
            finally:
                pipeline.request_stop('test')
                inspector.release.set()
                pipeline.finish()

    def test_native_call_has_bounded_join(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg['runtime']['shutdown_timeout_seconds'] = .02
        release,entered = threading.Event(),threading.Event()
        class BlockedCamera(Camera):
            def read(self):
                entered.set()
                release.wait(2)
                return FakeFrame()
        with tempfile.TemporaryDirectory() as root:
            pipeline = Pipeline(cfg,Path(root)/'run',BlockedCamera,Scout,Inspector)
            pipeline.initialize()
            pipeline.start(use_tegrastats=False)
            self.assertTrue(entered.wait(1))
            pipeline.request_stop('test')
            before = time.perf_counter()
            try:
                self.assertIn('scout',pipeline.finish())
                self.assertLess(time.perf_counter()-before,.5)
            finally:
                release.set()
                for worker in pipeline.threads:
                    worker.join(1)

    def test_http_preserves_exact_submission_and_completed_answer_on_stop(self):
        from src.scout_inspector.backends import LlamaCppInspector
        import base64
        cfg = copy.deepcopy(DEFAULTS['inspector'])
        client = LlamaCppInspector(cfg)
        client.api_model = 'fake-local-model'
        stop = threading.Event()
        stop.set()
        class Encoded(bytes):
            def tobytes(self):
                return bytes(self)
        encoded = Encoded(b'exact-fake-jpeg')
        def request(route,payload):
            self.assertEqual(route,'/chat/completions')
            self.assertEqual(payload['max_tokens'],32)
            self.assertEqual(payload['temperature'],0)
            content = payload['messages'][0]['content']
            self.assertEqual(content[0]['text'],'exact prompt')
            self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]),encoded)
            return {'choices':[{'message':{'content':'saved answer'},'finish_reason':'stop'}],
                    'usage':{'completion_tokens':2}}
        client._request = request
        with patch.dict('sys.modules',{'cv2':SimpleNamespace(IMWRITE_JPEG_QUALITY=1,imencode=lambda *args:(True,encoded))}):
            result = client.infer(None,'exact prompt',stop)
        self.assertEqual(result['_submitted_jpeg_bytes'],encoded)
        self.assertFalse(result['cancelled'])
        self.assertTrue(result['stop_requested_during_request'])
        self.assertEqual(result['usage']['completion_tokens'],2)

    def test_tegrastats_peak_distinct_from_cuda(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'tegrastats.txt'
            path.write_text('RAM 2000/7620MB\nRAM 3456/7620MB\n')
            summary = telemetry_summary(path)
            self.assertIn(3456,summary.values())


if __name__ == '__main__':
    unittest.main()
