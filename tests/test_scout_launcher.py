"""Offline managed-launcher lifecycle tests; every hardware executable is fake."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def exercise(self, fail):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for folder in ['experiments','configs','models/inspector','models/scout/source',
                           '.venv-scout/bin','dependencies/llama.cpp/build/bin','fakebin']:
                (root/folder).mkdir(parents=True,exist_ok=True)
            for name in ['run_scout_inspector.sh','summarize_scout_inspector.py']:
                shutil.copy(ROOT/'experiments'/name,root/'experiments'/name)
            shutil.copy(ROOT/'configs/scout_inspector.example.json',root/'configs/scout_inspector.local.json')
            for name in ['model-Q4_K_M.gguf','mmproj.gguf']:
                (root/'models/inspector'/name).write_bytes(b'fake')
            (root/'models/provision-manifest.json').write_text('{}')
            (root/'models/scout/source/yolo11n.onnx').write_bytes(b'fake')
            stub=root/'.venv-scout/bin/python'
            stub.write_text('#!'+sys.executable+'\n'+r'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args and args[0]=='-u': args=args[1:]
if args[0]=='-':
    sys.stdin.read()  # Skip sockets/HTTP in this offline lifecycle test.
    sys.exit(0)
if args[0] in ['experiments/export_scout.py','experiments/build_scout_from_onnx.py']:
    Path('models/scout/yolo11n-fp16.engine').write_bytes(b'fake')
    Path('export_executed.txt').write_text('fake export; no GPU')
    sys.exit(0)
if args[0]=='main.py':
    if '--check' in args: sys.exit(0)
    run=Path(args[args.index('--run')+1]); run.mkdir()
    failed=os.environ.get('SCOUT_TEST_FAIL_CLIENT')=='1'
    state={'metadata':{'status':'failed' if failed else 'finished','camera_metrics':{},'system_ram':{}},
           'counters':{},'errors':['intentional fake client failure'] if failed else []}
    (run/'run.json').write_text(json.dumps(state))
    (run/'scout.jsonl').write_text(json.dumps({'delivered_monotonic':2})+'\n')
    answer={'status':'complete','sequence':1,'completed_monotonic':2.5,'inference_ms':1000,
            'frame_age_at_inference_start_ms':500,'capture_to_completed_answer_ms':1500}
    (run/'inspections.jsonl').write_text(json.dumps(answer)+'\n')
    sys.exit(3 if failed else 0)
os.execv(sys.executable,[sys.executable]+args)
''')
            stub.chmod(0o755)
            for name,path in [('server',root/'dependencies/llama.cpp/build/bin/llama-server'),
                              ('tegra',root/'fakebin/tegrastats')]:
                path.write_text('#!/bin/bash\necho $$ > '+name+'_pid.txt\ntrap "exit 0" TERM\nwhile true; do echo "RAM 1234/7620MB"; sleep .1; done\n')
                path.chmod(0o755)
            env=os.environ.copy()
            env['PATH']=str(root/'fakebin')+':'+env['PATH']
            env['SCOUT_TEST_FAIL_CLIENT']=str(int(fail))
            result=subprocess.run(['bash','experiments/run_scout_inspector.sh'],cwd=root,env=env,
                                  capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,3 if fail else 0,result.stdout+result.stderr)
            run=next((root/'results').iterdir())
            self.assertEqual((run/'exit_status.txt').read_text().strip(),str(result.returncode))
            self.assertTrue((root/'export_executed.txt').exists())
            summary=json.loads(next(run.glob('summary_*.json')).read_text())
            self.assertEqual(summary['completed_answers'],1)
            self.assertEqual(summary['answers_with_capture_overlap'],1)
            for name in ['server','tegra']:
                pid=int((root/(name+'_pid.txt')).read_text())
                with self.assertRaises(ProcessLookupError): os.kill(pid,0)

    def test_success_cleans_owned_processes_and_summarizes(self):
        self.exercise(False)

    def test_failure_preserves_exit_and_partial_summary(self):
        self.exercise(True)


if __name__=='__main__': unittest.main()
