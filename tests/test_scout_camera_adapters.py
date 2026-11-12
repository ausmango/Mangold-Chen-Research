"""Adapter selection and pipeline checks without opening any camera."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from src.scout_inspector.backends import CsiCamera, UsbCamera, camera_factory, resolve_usb_device
from src.scout_inspector.config import DEFAULTS


class CameraAdapterTests(unittest.TestCase):
    def test_csi_pipeline_is_exact_legacy_pipeline(self):
        source=Path(__file__).resolve().parents[1]/'experiments/moondream_4bit_camera_v2.py'
        tree=ast.parse(source.read_text())
        capture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
        assignment=next(n for n in capture.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pipeline' for t in n.targets))
        self.assertEqual(CsiCamera.PIPELINE,ast.literal_eval(assignment.value))

    def test_factory_selects_without_opening_camera(self):
        cfg=copy.deepcopy(DEFAULTS['camera'])
        self.assertIsInstance(camera_factory(cfg),UsbCamera)
        cfg['kind']='csi'
        self.assertIsInstance(camera_factory(cfg),CsiCamera)

    def test_auto_usb_does_not_silently_fall_back_to_csi(self):
        with patch('pathlib.Path.glob',return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'use --camera csi'):
                resolve_usb_device('auto')


if __name__=='__main__': unittest.main()
