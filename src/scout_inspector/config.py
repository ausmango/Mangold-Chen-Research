"""Strict JSON configuration; no hardware/model imports."""
import copy
import json
import math
import re
from pathlib import Path

DEFAULTS = {
    'camera': {'kind': 'usb', 'device': 'auto', 'width': 640, 'height': 480, 'fps': 30,
               'fourcc': 'MJPG', 'buffer_size': 1},
    'scout': {'weights': 'models/scout/yolo11n-fp16.engine', 'imgsz': 640,
              'device': '0', 'confidence': 0.4, 'max_detections': 50},
    'triggers': {'person_classes': ['person'],
                 'target_classes': ['car', 'truck', 'bus'],
                 'proximity_enabled': True, 'proximity_gap_fraction': 0.08,
                 'presence_classes': [], 'motion_enabled': False,
                 'motion_speed_fraction_per_second': 0.8, 'motion_match_fraction': 0.25,
                 'missing_helmet_enabled': False, 'helmet_classes': ['helmet'],
                 'ppe_min_person_height_px': 100, 'min_trigger_interval_ms': 250},
    'inspector': {'backend': 'llama_cpp', 'base_url': 'http://127.0.0.1:8080/v1',
                  'model_id': 'ggml-org/SmolVLM2-500M-Video-Instruct-GGUF',
                  'revision': None, 'model_path': 'models/inspector/model-Q4_K_M.gguf',
                  'mmproj_path': 'models/inspector/mmproj.gguf',
                  'image_size': 384, 'max_frame_age_ms': 1000,
                  'max_answer_age_ms': 3000, 'max_tokens': 32, 'temperature': 0,
                  'request_timeout_seconds': 30, 'generation_timeout_seconds': 20,
                  'prompt': 'Describe only clearly visible conditions in this construction image. In one short sentence, state the observation, location, and a possible safety concern if supported. If unclear, say so. Treat Scout triggers as unverified hints; do not invent hidden hazards, movement, missing PPE, or OSHA violations.'},
    'runtime': {'startup_timeout_seconds': 120, 'shutdown_timeout_seconds': 5,
                'force_exit_on_shutdown_timeout': True, 'require_tegrastats': True,
                'idle_sleep_seconds': 0.01},
}


def merge(default, supplied):
    if not isinstance(supplied, dict):
        raise ValueError('Configuration sections must be objects')
    unknown = set(supplied)-set(default)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        result[key] = merge(default[key], value) if isinstance(default[key], dict) else value
    return result


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def load_config(path):
    cfg = merge(DEFAULTS, json.loads(Path(path).read_text()))
    for section, keys in [('camera', ['width','height','fps','buffer_size']),
                          ('scout',['imgsz','max_detections']),
                          ('inspector',['image_size','max_frame_age_ms','max_answer_age_ms','max_tokens',
                                        'request_timeout_seconds','generation_timeout_seconds']),
                          ('runtime',['startup_timeout_seconds','shutdown_timeout_seconds','idle_sleep_seconds'])]:
        for key in keys:
            positive(cfg[section][key], f'{section}.{key}')
    for section, keys in [('camera',['width','height','fps','buffer_size']),
                          ('scout',['imgsz','max_detections']),('inspector',['image_size','max_tokens'])]:
        if any(not isinstance(cfg[section][k],int) or isinstance(cfg[section][k],bool) for k in keys):
            raise ValueError(f'{section} sizes/counts must be integers')
    if cfg['camera']['kind'] not in ['usb','csi']:
        raise ValueError('camera.kind must be usb or csi')
    if not 0 < cfg['scout']['confidence'] <= 1:
        raise ValueError('scout.confidence must be in (0,1]')
    for key in ['proximity_gap_fraction','motion_speed_fraction_per_second','motion_match_fraction','ppe_min_person_height_px']:
        positive(cfg['triggers'][key], key)
    interval = cfg['triggers']['min_trigger_interval_ms']
    if not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval < 0:
        raise ValueError('min_trigger_interval_ms must be finite and nonnegative')
    for key in ['person_classes','target_classes','presence_classes','helmet_classes']:
        value = cfg['triggers'][key]
        if not isinstance(value,list) or not all(isinstance(v,str) and v.strip() for v in value):
            raise ValueError(f'{key} must be a list of class names')
    if not cfg['triggers']['person_classes']:
        raise ValueError('person_classes must not be empty')
    if cfg['triggers']['proximity_enabled'] and not cfg['triggers']['target_classes']:
        raise ValueError('Proximity requires target_classes')
    if cfg['inspector']['backend'] not in ['llama_cpp','transformers']:
        raise ValueError('inspector.backend must be llama_cpp or transformers')
    if cfg['inspector']['model_id'] not in ['ggml-org/SmolVLM2-500M-Video-Instruct-GGUF',
                                          'HuggingFaceTB/SmolVLM2-500M-Video-Instruct']:
        raise ValueError('Initial supported Inspector is SmolVLM2-500M; other models need a validated adapter')
    revision = cfg['inspector']['revision']
    if not isinstance(revision,str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Set inspector.revision to the exact artifact revision (not main/latest)')
    for section, keys in [('triggers',['proximity_enabled','motion_enabled','missing_helmet_enabled']),
                          ('runtime',['force_exit_on_shutdown_timeout','require_tegrastats'])]:
        if any(type(cfg[section][key]) is not bool for key in keys):
            raise ValueError(f'{section} switches must be booleans')
    if not isinstance(cfg['camera']['fourcc'],str) or len(cfg['camera']['fourcc']) != 4:
        raise ValueError('camera.fourcc must contain four characters')
    if cfg['inspector']['temperature'] != 0:
        raise ValueError('Initial benchmark requires deterministic temperature=0')
    if not isinstance(cfg['inspector']['prompt'],str) or not cfg['inspector']['prompt'].strip():
        raise ValueError('inspector.prompt is required')
    if cfg['inspector']['image_size'] > 512:
        raise ValueError('Keep initial Inspector image_size <=512 to bound submitted pixels')
    return cfg


def check_assets(cfg):
    required = [Path(cfg['scout']['weights']), Path(cfg['inspector']['model_path'])]
    if required[0].suffix != '.engine':
        raise ValueError('Scout requires a locally exported TensorRT .engine; no automatic model downloads')
    if cfg['inspector']['backend'] == 'llama_cpp':
        required.append(Path(cfg['inspector']['mmproj_path']))
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
    else:
        if not required[0].is_file() or not required[1].is_dir():
            raise FileNotFoundError('Need local Scout engine and a local Inspector snapshot directory')
    return required
