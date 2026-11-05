"""No hardware: ensure same-image cache pairing and latest-frame replacement."""
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
import moondream_4bit_live_compare as compare


class CompareTests(unittest.TestCase):
    def exercise(self, fail=False):
        slot = compare.live.LatestFrame()
        stop, ready = threading.Event(), threading.Event()
        slot.publish('first', time.perf_counter(), compare.live.utc())
        encoded_inputs, queries = [], []
        def encode(image):
            encoded_inputs.append(image)
            return ('encoded', len(encoded_inputs))
        def query(encoded, prompt, settings):
            queries.append(encoded)
            if len(queries)==1:
                for index in range(2, 51):
                    slot.publish(f'frame{index}', time.perf_counter(), compare.live.utc())
            if fail and len(queries)==2:
                raise RuntimeError('synthetic query error')
            if len(queries)==4:
                stop.set()
            return {'answer':'Uneven ground'}
        model = types.SimpleNamespace(encode_image=encode, query=query,
                                     get_memory_footprint=lambda: 1024**3)
        model.eval=lambda: model
        cuda=types.SimpleNamespace(empty_cache=lambda:None, reset_peak_memory_stats=lambda:None,
            synchronize=lambda:None, mem_get_info=lambda:(5*1024**3,8*1024**3),
            memory_allocated=lambda:2*1024**3,max_memory_allocated=lambda:3*1024**3)
        torch=types.ModuleType('torch');torch.cuda=cuda;torch.inference_mode=contextlib.nullcontext
        transformer=types.ModuleType('transformers')
        modeling=types.ModuleType('transformers.modeling_utils');transformer.modeling_utils=modeling
        transformer.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a,**kw:model)
        cv2=types.ModuleType('cv2');cv2.COLOR_BGR2RGB=0;cv2.cvtColor=lambda f,c:f
        def write(path,frame):
            Path(path).write_bytes(b'frame')
            return True
        cv2.imwrite=write
        pil=types.ModuleType('PIL');pil.Image=types.SimpleNamespace(fromarray=lambda f:f)
        modules={'torch':torch,'transformers':transformer,'transformers.modeling_utils':modeling,'cv2':cv2,'PIL':pil}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules,modules), \
                patch.object(compare.importlib.metadata,'version',return_value='synthetic'), \
                contextlib.redirect_stdout(io.StringIO()):
            run=Path(folder);(run/'frames').mkdir()
            results={'frames':[]}
            compare.comparison_worker(run,slot,stop,ready,results)
            saved=json.loads((run/'analysis.json').read_text())
            self.assertTrue(ready.is_set())
            self.assertEqual(queries[:2],[('encoded',1),('encoded',1)])
            if fail:
                self.assertIn('synthetic query error', saved['worker_error'])
                self.assertEqual(saved['frames'][0]['status'],'complete')
                self.assertEqual(saved['frames'][1]['status'],'failed')
            else:
                self.assertEqual(encoded_inputs,['first','frame50'])
                self.assertEqual(queries,[('encoded',1),('encoded',1),('encoded',2),('encoded',2)])
                self.assertEqual([p['sequence'] for p in saved['pairs']],[1,50])
                self.assertEqual([r['profile'] for r in saved['frames']],['short32','compact16','compact16','short32'])
                for row in saved['frames']:
                    self.assertAlmostEqual(row['frame_age_at_inference_start_seconds']+row['inference_seconds'],row['capture_to_completed_answer_seconds'])
                    self.assertTrue((run / f"frames/frame_{row['sequence']:06d}_{row['profile']}.json").exists())

    def test_cache_pair_and_latest_frame(self):
        self.exercise()

    def test_partial_error_preservation(self):
        self.exercise(fail=True)


if __name__=='__main__':
    unittest.main()
