"""Lazy runtime adapters. Importing this module loads no camera/CUDA/model code."""
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

from .core import Detection


def artifact(path):
    path = Path(path).resolve()
    if path.is_dir():
        files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in ['.json','.safetensors','.bin','.model'])
        if not files:
            raise ValueError(f'No model/config artifacts in {path}')
        return {'path': str(path), 'files': [artifact(p) for p in files]}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def resolve_usb_device(device):
    """Read sysfs only; never mistake the legacy IMX219 CSI node for USB."""
    def is_usb(path):
        node = Path('/sys/class/video4linux')/Path(path).resolve().name
        location = str((node/'device').resolve())
        return any(part.startswith('usb') for part in Path(location).parts)
    if device == 'auto':
        stable = sorted(Path('/dev/v4l/by-id').glob('*video-index0'))
        for path in stable:
            if is_usb(path): return str(path)
        for node in sorted(Path('/sys/class/video4linux').glob('video*'),key=lambda p:int(p.name[5:])):
            path = '/dev/'+node.name
            if is_usb(path): return path
        raise RuntimeError('No USB camera found. The IMX219 is CSI; use --camera csi for that camera, or connect a USB webcam.')
    path = '/dev/video'+str(device) if isinstance(device,int) else str(device)
    if not is_usb(path):
        raise RuntimeError(f'{path} is not identified as USB. Use --camera csi for the IMX219.')
    return path


class UsbCamera:
    def __init__(self, config):
        self.config = config
        self.capture = None

    def open(self):
        import cv2
        cfg = self.config
        device = resolve_usb_device(cfg['device'])
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"USB camera {cfg['device']} could not open with V4L2")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg['fourcc']))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])
        self.capture.set(cv2.CAP_PROP_FPS, cfg['fps'])
        buffer_accepted = self.capture.set(cv2.CAP_PROP_BUFFERSIZE, cfg['buffer_size'])
        return {'backend': self.capture.getBackendName(), 'selected_device':device,
                'reported_width': self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                'reported_height': self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'reported_fps': self.capture.get(cv2.CAP_PROP_FPS),
                'buffer_request_accepted': bool(buffer_accepted),
                'note': 'Requested/reported camera FPS is not measured capture throughput; USB/driver buffering remains unmeasured'}

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(self.config.get('kind','usb').upper()+' camera read failed')
        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class CsiCamera(UsbCamera):
    """The original camera_v2 CSI pipeline, unchanged; opened only on explicit launch."""
    PIPELINE = (
        'nvarguscamerasrc sensor-id=0 ! '
        'video/x-raw(memory:NVMM),width=1280,height=720,'
        'framerate=30/1,format=NV12 ! '
        'nvvidconv ! video/x-raw,width=640,height=360,format=BGRx ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    )

    def open(self):
        import cv2
        self.capture = cv2.VideoCapture(self.PIPELINE,cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError('CSI camera could not open with the preserved camera_v2 pipeline')
        return {'backend':'GStreamer CSI','pipeline':self.PIPELINE,
                'reported_width':self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                'reported_height':self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'reported_fps':self.capture.get(cv2.CAP_PROP_FPS),
                'note':'Preserved 1280x720/30 sensor request, 640x360 BGR output; delivered rate measured separately'}


def camera_factory(config):
    return CsiCamera(config) if config['kind']=='csi' else UsbCamera(config)


class YoloScout:
    def __init__(self, config):
        self.config = config

    def load(self):
        # Suppress Ultralytics' automatic dependency installer, not errors.
        os.environ['YOLO_AUTOINSTALL'] = 'False'
        from ultralytics import YOLO
        cfg = self.config
        if not Path(cfg['weights']).is_file():
            raise FileNotFoundError(cfg['weights'])
        self.model = YOLO(cfg['weights'], task='detect')
        # Explicit synthetic warmup BEFORE opening USB capture. Not a benchmark.
        import numpy as np
        self.predict(np.zeros((cfg['imgsz'],cfg['imgsz'],3), dtype=np.uint8))
        names = self.model.names
        self.labels = list(names.values()) if isinstance(names,dict) else list(names)
        metadata = {'adapter': 'ultralytics TensorRT', 'ultralytics': importlib.metadata.version('ultralytics'),
                    'engine': artifact(cfg['weights']), 'classes': self.labels,
                    'device': cfg['device'], 'imgsz': cfg['imgsz'],
                    'precision': 'See engine export manifest; not inferred from file name'}
        sidecar = Path(cfg['weights']).with_suffix('.json')
        if sidecar.exists():
            metadata['export_manifest'] = json.loads(sidecar.read_text())
        return metadata

    def predict(self, frame):
        cfg = self.config
        result = self.model.predict(source=frame, imgsz=cfg['imgsz'], conf=cfg['confidence'],
                                    max_det=cfg['max_detections'], device=cfg['device'],
                                    verbose=False, stream=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxyn.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        confidence = result.boxes.conf.cpu().tolist()
        names = result.names
        detections = []
        for box, label, score in zip(boxes, classes, confidence):
            box = tuple(min(1.0,max(0.0,float(x))) for x in box)
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(Detection(names[int(label)],float(score),box))
        return detections

    def close(self):
        self.model = None


class LlamaCppInspector:
    """One HTTP request in flight to a separately managed, local CUDA server."""
    def __init__(self, config):
        self.config = config
        parsed = urllib.parse.urlparse(config['base_url'])
        if parsed.scheme != 'http' or parsed.hostname not in ['127.0.0.1','localhost','::1']:
            raise ValueError('Initial Inspector endpoint must be a local HTTP llama.cpp server')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, route, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.config['base_url'].rstrip('/')+route, data=body,
                                         headers={'Content-Type':'application/json'})
        with self.opener.open(request, timeout=self.config['request_timeout_seconds']) as response:
            content = response.read(1024*1024+1)
        if len(content) > 1024*1024:
            raise RuntimeError('Inspector response exceeds 1 MiB limit')
        return json.loads(content)

    def load(self):
        info = self._request('/models')
        models = info.get('data', [])
        if len(models) != 1 or not models[0].get('id'):
            raise RuntimeError('Configure one loaded llama.cpp Inspector model before starting')
        self.api_model = models[0]['id']
        return {'backend':'llama.cpp local HTTP (externally managed CUDA server)',
                'api_model_id': self.api_model, 'model_id': self.config['model_id'],
                'model_load_ms':None, 'model_footprint_bytes':None,
                'load_timing_scope':'Client connection/artifact verification only; external server model load time must come from server log',
                'revision': self.config['revision'], 'model_artifact':artifact(self.config['model_path']),
                'projector_artifact':artifact(self.config['mmproj_path']),
                'note':'Artifact hashes record configured files; server must be launched with these exact files. Projector may remain higher precision. Server memory is not measured by client PyTorch statistics.'}

    def infer(self, image, prompt, stop):
        import cv2
        ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok:
            raise RuntimeError('Could not encode Inspector image')
        url = 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode('ascii')
        response = self._request('/chat/completions', {
            'model':self.api_model, 'temperature':self.config['temperature'],
            'max_tokens':self.config['max_tokens'], 'stream':False,
            'messages':[{'role':'user','content':[{'type':'text','text':prompt},
                                                {'type':'image_url','image_url':{'url':url}}]}],
        })
        choice = response['choices'][0]
        answer = choice['message']['content']
        if not isinstance(answer,str):
            raise RuntimeError('Inspector returned non-text content')
        return {'answer':answer, 'usage':response.get('usage'),
                'server_timings':response.get('timings'), 'finish_reason':choice.get('finish_reason'),
                'cancelled':False, 'stop_requested_during_request':stop.is_set(),
                '_submitted_jpeg_bytes':encoded.tobytes(), 'memory':self.memory_stats()}

    def memory_stats(self):
        return {'cuda_allocated_bytes':None, 'cuda_peak_allocated_bytes':None,
                'note':'External llama.cpp server; client cannot report its CUDA allocation. Use whole-system tegrastats.'}

    def close(self):
        pass  # Never terminate a server this client did not start.


class TransformersInspector:
    """Local SmolVLM2 FP16 fallback, no weight downloads or generic BnB patch."""
    def __init__(self, config):
        self.config = config

    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        torch.cuda.reset_peak_memory_stats()
        path = self.config['model_path']
        self.processor = AutoProcessor.from_pretrained(path,local_files_only=True)
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.size = {'longest_edge':self.config['image_size']}
        model_load_started = time.perf_counter()
        self.model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='sdpa', device_map={'':'cuda'}).eval()
        torch.cuda.synchronize()
        return {'backend':'Transformers FP16 CUDA/SDPA', 'precision':'FP16, NOT INT4',
                'model_load_ms':1000*(time.perf_counter()-model_load_started),
                'model_id':self.config['model_id'],'revision':self.config['revision'],
                'artifact':artifact(path),'model_footprint_bytes':self.model.get_memory_footprint(),
                'image_splitting':False,'processor_longest_edge':self.config['image_size'],
                'processor_native_patch_size':getattr(self.processor.image_processor,'max_image_size',None),
                'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers']},
                'memory':self.memory_stats()}

    def infer(self, image, prompt, stop):
        from PIL import Image
        import cv2
        from transformers import StoppingCriteria, StoppingCriteriaList
        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()
        rgb = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        text = self.processor.apply_chat_template([{'role':'user','content':[
            {'type':'image'},{'type':'text','text':prompt}]}],add_generation_prompt=True,tokenize=False)
        inputs = self.processor(text=text,images=[rgb],return_tensors='pt').to('cuda')
        for key,value in inputs.items():
            if self.torch.is_floating_point(value):
                inputs[key] = value.to(self.torch.float16)
        count = inputs['input_ids'].shape[-1]
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.config['max_tokens'],
                max_time=self.config['generation_timeout_seconds'],
                stopping_criteria=StoppingCriteriaList([Cancel()]))
        self.torch.cuda.synchronize()
        generated = output[0,count:]
        answer = self.processor.decode(generated,skip_special_tokens=True)
        return {'answer':answer, 'processor_input_tokens':int(count),
                'generated_token_ids_count':int(generated.shape[0]),
                'token_count_note':'Generated-ID count may include EOS; processor input length is not a separate measured visual-encoder token count.',
                'cancelled':stop.is_set(),'memory':self.memory_stats()}

    def memory_stats(self):
        return {'cuda_allocated_bytes':self.torch.cuda.memory_allocated(),
                'cuda_peak_allocated_bytes':self.torch.cuda.max_memory_allocated(),
                'note':'PyTorch process allocation includes Scout allocations tracked by PyTorch; TensorRT and other processes are not covered'}

    def close(self):
        self.model = None
        self.processor = None


def inspector_factory(config):
    return LlamaCppInspector(config) if config['backend']=='llama_cpp' else TransformersInspector(config)
