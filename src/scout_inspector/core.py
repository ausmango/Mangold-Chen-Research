"""CPU-only contracts, trigger rules, and the single-slot memory bridge."""
from dataclasses import asdict, dataclass
import math
import queue
import threading
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # Coordinates normalized to the ORIGINAL camera image.
    box: tuple

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        if not all(math.isfinite(x) for x in (*self.box, self.confidence)):
            raise ValueError('Nonfinite detection')
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('Detection box must be normalized, nonempty and within image')


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_monotonic: float
    captured_utc: str
    image: Any
    reasons: tuple
    detections: tuple
    source_shape: tuple
    transform: dict
    scout_ms: float


class LatestQueue:
    """Exactly queue.Queue(maxsize=1), one producer, one consumer, no blocking put.

    Producer lock serializes replacement if additional callers are introduced.
    Consumer races with discard are handled. task_done balances dropped entries.
    """
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)
        self._producer_lock = threading.Lock()

    def replace(self, packet):
        dropped = 0
        with self._producer_lock:
            try:
                self.queue.put_nowait(packet)
                return dropped
            except queue.Full:
                pass
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                dropped = 1
            except queue.Empty:
                pass  # Consumer removed it between Full and get_nowait.
            self.queue.put_nowait(packet)
        return dropped

    def take(self):
        return self.queue.get_nowait()

    def done(self):
        self.queue.task_done()

    def clear(self):
        try:
            self.queue.get_nowait()
            self.queue.task_done()
            return 1
        except queue.Empty:
            return 0


class State:
    """Only counters, latest preview and latest result; no image history."""
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = {}
        self.metadata = {}
        self.preview = None
        self.answer = None
        self.errors = []

    def count(self, key, amount=1):
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + amount

    def update(self, **values):
        with self.lock:
            self.metadata.update(values)

    def set_preview(self, value):
        with self.lock:
            self.preview = value

    def set_answer(self, value):
        with self.lock:
            self.answer = value

    def display(self):
        with self.lock:
            return self.preview, self.answer

    def error(self, worker, detail):
        with self.lock:
            self.errors.append({'worker': worker, 'traceback': detail})

    def snapshot(self):
        with self.lock:
            return {'counters': dict(self.counters), 'metadata': dict(self.metadata),
                    'errors': list(self.errors)}


class TriggerRules:
    def __init__(self, config):
        self.config = config
        self.previous = []
        self.previous_time = None

    def validate_labels(self, available):
        labels = set(available)
        required = set(self.config['person_classes'])
        if self.config['proximity_enabled']:
            required.update(self.config['target_classes'])
        required.update(self.config['presence_classes'])
        if self.config['missing_helmet_enabled']:
            required.update(self.config['helmet_classes'])
        missing = required - labels
        if missing:
            raise ValueError(f'Detector lacks configured classes {sorted(missing)}; use a matching custom detector/config. Stock COCO has no general machinery/PPE labels.')

    @staticmethod
    def box_distance(a, b):
        dx = max(a[0] - b[2], b[0] - a[2], 0)
        dy = max(a[1] - b[3], b[1] - a[3], 0)
        return math.hypot(dx, dy)

    @staticmethod
    def center(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    def evaluate(self, detections, now, source_height):
        cfg = self.config
        people = [d for d in detections if d.label in cfg['person_classes']]
        targets = [d for d in detections if d.label in cfg['target_classes']]
        reasons = []
        if cfg['proximity_enabled']:
            for person in people:
                for target in targets:
                    distance = self.box_distance(person.box, target.box)
                    if distance <= cfg['proximity_gap_fraction']:
                        reasons.append({'rule': 'image_plane_proximity', 'target': target.label,
                                        'gap_fraction': distance,
                                        'observation': 'person detection near vehicle/equipment detection in image coordinates; physical separation unknown'})
        for label in sorted({d.label for d in detections} & set(cfg['presence_classes'])):
            reasons.append({'rule': 'class_presence', 'label': label})
        centers = [self.center(p.box) for p in people]
        dt = now-self.previous_time if self.previous_time is not None else None
        if cfg['motion_enabled'] and dt is not None and 0.01 <= dt <= 0.5:
            for center in centers:
                if self.previous:
                    distance = min(math.dist(center, old) for old in self.previous)
                    if distance <= cfg['motion_match_fraction'] and distance/dt >= cfg['motion_speed_fraction_per_second']:
                        reasons.append({'rule': 'rapid_image_plane_change',
                                        'speed_fraction_per_second': distance/dt,
                                        'observation': 'nearest-centroid heuristic, not tracked identity, physical speed, or verified event'})
        self.previous, self.previous_time = centers, now
        if cfg['missing_helmet_enabled']:
            helmets = [self.center(d.box) for d in detections if d.label in cfg['helmet_classes']]
            for person in people:
                x1,y1,x2,y2 = person.box
                if (y2-y1)*source_height < cfg['ppe_min_person_height_px']:
                    continue
                if not any(x1 <= x <= x2 and y1 <= y <= y1+0.35*(y2-y1) for x,y in helmets):
                    reasons.append({'rule': 'no_helmet_detection_in_estimated_head_region',
                                    'observation': 'possible review trigger only; occlusion/missed detections can explain absence'})
        return tuple(reasons[:16])


def detection_dicts(detections):
    return [asdict(d) for d in detections]


def prepare_image(frame, size):
    """Uniform square grid with aspect-preserving letterbox; never crop context."""
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    scale = min(size/width, size/height)
    new_w, new_h = max(1, round(width*scale)), max(1, round(height*scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size-new_w)//2, (size-new_h)//2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas, {'method': 'letterbox', 'scale': scale, 'left': left, 'top': top,
                    'output_width': size, 'output_height': size}
