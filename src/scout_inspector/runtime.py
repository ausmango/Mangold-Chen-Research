"""Two independent workers; main owns UI, lifecycle, and bounded shutdown."""
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone

from .core import LatestQueue, State, TriggerRules, FramePacket, detection_dicts, prepare_image


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class EventLog:
    def __init__(self, path):
        self.stream = path.open('x',encoding='utf-8',buffering=1)

    def emit(self, **value):
        self.stream.write(json.dumps({'logged_utc':utc(),**value},allow_nan=False)+'\n')

    def close(self):
        self.stream.close()


def system_memory():
    wanted = ['MemTotal','MemAvailable','SwapTotal','SwapFree']
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,_,value = line.partition(':')
        if key in wanted:
            values[key+'_bytes'] = int(value.split()[0])*1024
    return values


def telemetry_summary(path):
    count, peak, total = 0, None, None
    if path.exists():
        with path.open(errors='replace') as stream:
            for line in stream:
                match = re.search(r'RAM (\d+)/(\d+)MB',line)
                if match:
                    used,total = map(int,match.groups())
                    peak = max(peak or 0,used)
                    count += 1
    return {'source':'tegrastats, whole system, 500 ms samples', 'samples':count,
            'peak_used_mb':peak,'total_mb':total,
            'error':None if count else 'No whole-system RAM samples recorded'}


class Pipeline:
    def __init__(self, cfg, run, camera_factory, scout_factory, inspector_factory,
                 prepare=prepare_image, save_image=None, wait_for_start=False):
        self.cfg, self.run = cfg, Path(run)
        self.camera_factory,self.scout_factory,self.inspector_factory = camera_factory,scout_factory,inspector_factory
        self.prepare = prepare
        self.save_image = save_image or self._save_image
        self.bridge,self.state = LatestQueue(),State()
        self.stop = threading.Event()
        self.armed = threading.Event()
        if not wait_for_start:
            self.armed.set()
        self.inspector_ready,self.scout_ready,self.first_frame = (threading.Event() for _ in range(3))
        self.threads = []
        self.telemetry = None
        self.stop_reason = None
        self.first_delivery,self.last_delivery = None,None
        self.max_capture_gap = 0

    @staticmethod
    def _save_image(path, image):
        import cv2
        if not cv2.imwrite(str(path),image):
            raise RuntimeError(f'Could not persist analyzed image: {path}')

    def initialize(self):
        # Existing folders are NEVER reused, even after a failed initialization.
        self.run.mkdir(parents=True,exist_ok=False)
        (self.run/'frames').mkdir()
        save_json(self.run/'config.json',self.cfg)
        self.state.update(status='initializing',created_utc=utc(),system_before=system_memory(),
            design=self.cfg['camera']['kind'].upper()+' YOLO Scout + latest-triggered-frame Queue(maxsize=1) + serial Inspector',
            timestamp_scope='Host delivery after camera.read, not sensor exposure; capture-to-answer excludes unknown driver buffering',
            limitations='Unverified generated observations, not OSHA compliance findings or validated temporal event detection. GPU contention and native-call blocking can still occur.')
        self.checkpoint()

    def start(self, use_tegrastats=True):
        if use_tegrastats:
            executable = shutil.which('tegrastats')
            if executable is None and self.cfg['runtime']['require_tegrastats']:
                raise FileNotFoundError('tegrastats required but not found; do not substitute CUDA allocation for system RAM')
            if executable:
                with (self.run/'tegrastats.txt').open('x') as stream:
                    self.telemetry = subprocess.Popen([executable,'--interval','500'],stdout=stream,stderr=subprocess.STDOUT)
        self.threads = [threading.Thread(target=self._inspector,name='inspector',daemon=True),
                        threading.Thread(target=self._scout,name='scout',daemon=True)]
        for thread in self.threads:
            thread.start()

    def arm(self):
        if not self.armed.is_set():
            self.state.update(armed_utc=utc())
            self.armed.set()
            print('Starting timed capture on next delivered frame. Start source playback now.',flush=True)

    def request_stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason
        self.stop.set()

    def checkpoint(self):
        save_json(self.run/'run.json',self.state.snapshot())

    def _failed(self, worker):
        detail = traceback.format_exc()
        self.state.error(worker,detail)
        print(f'{worker.upper()} ERROR:\n{detail}',flush=True)
        self.request_stop(worker+'_error')

    def _scout(self):
        camera,detector,log = None,None,None
        try:
            log = EventLog(self.run/'scout.jsonl')
            # Initialization waits for Inspector LOAD, never for its generation.
            while not self.inspector_ready.is_set() and not self.stop.wait(0.01):
                pass
            if self.stop.is_set():
                return
            detector = self.scout_factory(self.cfg['scout'])
            started = time.perf_counter()
            metadata = detector.load()
            rules = TriggerRules(self.cfg['triggers'])
            rules.validate_labels(detector.labels)
            self.state.update(scout={**metadata,'load_ms':1000*(time.perf_counter()-started)})
            camera = self.camera_factory(self.cfg['camera'])
            self.state.update(camera=camera.open())
            self.scout_ready.set()
            sequence,last_offer = 0,float('-inf')
            while not self.stop.is_set():
                read_started = time.perf_counter()
                frame = camera.read()
                delivered, timestamp = time.perf_counter(),utc()
                if self.stop.is_set():
                    break
                sequence += 1
                if not self.armed.is_set():
                    detections = tuple(detector.predict(frame))
                    self.state.set_preview((sequence,frame,detections,timestamp))
                    self.state.count('alignment_frames')
                    continue
                if self.first_delivery is None:
                    self.first_delivery = delivered
                    self.state.update(status='running',first_frame_utc=timestamp)
                    self.first_frame.set()
                if self.last_delivery is not None:
                    self.max_capture_gap = max(self.max_capture_gap,delivered-self.last_delivery)
                self.last_delivery = delivered
                self.state.count('captured_frames')
                started = time.perf_counter()
                detections = tuple(detector.predict(frame))
                detected = time.perf_counter()
                reasons = rules.evaluate(detections,delivered,frame.shape[0])
                self.state.set_preview((sequence,frame,detections,timestamp))
                replaced = 0
                if reasons:
                    self.state.count('triggered_frames')
                    interval = self.cfg['triggers']['min_trigger_interval_ms']/1000
                    if delivered-last_offer >= interval:
                        image,transform = self.prepare(frame,self.cfg['inspector']['image_size'])
                        packet = FramePacket(sequence,delivered,timestamp,image,reasons,detections,
                                             tuple(frame.shape),transform,1000*(detected-started))
                        # No put(), join(), HTTP request or Inspector lock here.
                        replaced = self.bridge.replace(packet)
                        self.state.count('queued_frames')
                        self.state.count('replaced_pending_frames',replaced)
                        last_offer = delivered
                    else:
                        self.state.count('trigger_cooldown_frames')
                else:
                    self.state.count('nominal_frames_discarded')
                end = time.perf_counter()
                if end-read_started > 1/self.cfg['camera']['fps']:
                    self.state.count('scout_frame_budget_misses')
                log.emit(sequence=sequence,captured_utc=timestamp,delivered_monotonic=delivered,
                    read_ms=1000*(delivered-read_started),detector_ms=1000*(detected-started),
                    loop_ms=1000*(end-read_started),replaced_pending=replaced,
                    triggers=reasons,detections=detection_dicts(detections))
        except Exception:
            self._failed('scout')
        finally:
            for obj in (camera,detector):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        self._failed('scout_cleanup')
            if log:
                log.close()
            self.scout_ready.set()

    def _inspector(self):
        backend,log = None,None
        try:
            log = EventLog(self.run/'inspections.jsonl')
            backend = self.inspector_factory(self.cfg['inspector'])
            started = time.perf_counter()
            metadata = backend.load()
            self.state.update(inspector={**metadata,'load_ms':1000*(time.perf_counter()-started)})
            self.inspector_ready.set()
            while not self.stop.is_set():
                try:
                    packet = self.bridge.take()
                except queue.Empty:
                    # Interruptible equivalent of the requested 10 ms idle sleep.
                    self.stop.wait(self.cfg['runtime']['idle_sleep_seconds'])
                    continue
                try:
                    age = 1000*(time.perf_counter()-packet.captured_monotonic)
                    if age > self.cfg['inspector']['max_frame_age_ms']:
                        self.state.count('stale_frames_discarded')
                        log.emit(status='stale_discarded',sequence=packet.sequence,frame_age_ms=age)
                        continue
                    frame_path = Path('frames')/f'frame_{packet.sequence:08d}.png'
                    self.save_image(self.run/frame_path,packet.image)
                    prompt = self.cfg['inspector']['prompt']+'\nScout review hints (not verified findings): '+json.dumps(packet.reasons[:2],separators=(',',':'))
                    row = {'status':'started','sequence':packet.sequence,'frame_path':str(frame_path),
                           'captured_utc':packet.captured_utc,'captured_monotonic':packet.captured_monotonic,
                           'source_shape':packet.source_shape,
                           'transform':packet.transform,'triggers':packet.reasons,
                           'detections':detection_dicts(packet.detections),'prompt':prompt,
                           'settings':{'max_tokens':self.cfg['inspector']['max_tokens'],
                                       'temperature':self.cfg['inspector']['temperature']}}
                    log.emit(**row)
                    started = time.perf_counter()
                    start_age = 1000*(started-packet.captured_monotonic)
                    if start_age > self.cfg['inspector']['max_frame_age_ms']:
                        self.state.count('stale_frames_discarded')
                        log.emit(status='stale_after_persistence',sequence=packet.sequence,frame_age_ms=start_age)
                        continue
                    row['inference_started_utc'] = utc()
                    row['inference_started_monotonic'] = started
                    result = backend.infer(packet.image,prompt,self.stop)
                    completed = time.perf_counter()
                    jpeg = result.pop('_submitted_jpeg_bytes',None)
                    if jpeg is not None:
                        transport_path=frame_path.with_suffix('.jpg')
                        (self.run/transport_path).write_bytes(jpeg)
                        row['submitted_image_path']=str(transport_path)
                    delay = 1000*(completed-packet.captured_monotonic)
                    row.update(**result,status='cancelled' if result.get('cancelled') else 'complete',
                        completed_utc=utc(),completed_monotonic=completed,
                        scout_ms=packet.scout_ms,inference_ms=1000*(completed-started),
                        frame_age_at_inference_start_ms=start_age,
                        capture_to_completed_answer_ms=delay,
                        answer_is_stale=delay > self.cfg['inspector']['max_answer_age_ms'],
                        assessment='Unverified model-generated observation; not an OSHA violation determination')
                    log.emit(**row)
                    save_json(self.run/frame_path.with_suffix('.json'),row)
                    self.state.set_answer(row)
                    self.state.count('completed_answers' if row['status']=='complete' else 'cancelled_answers')
                    print(json.dumps({k:row[k] for k in ['sequence','answer','inference_ms',
                        'frame_age_at_inference_start_ms','capture_to_completed_answer_ms','answer_is_stale']}),flush=True)
                finally:
                    self.bridge.done()
        except Exception:
            self._failed('inspector')
        finally:
            if backend:
                try:
                    self.state.update(inspector_final_memory=backend.memory_stats())
                    backend.close()
                except Exception:
                    self._failed('inspector_cleanup')
            if log:
                log.close()
            self.inspector_ready.set()

    def finish(self):
        self.stop.set()
        deadline=time.perf_counter()+self.cfg['runtime']['shutdown_timeout_seconds']
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=max(0,deadline-time.perf_counter()))
        blocked=[t.name for t in self.threads if t.is_alive()]
        if not blocked:
            self.state.count('pending_discarded_on_stop',self.bridge.clear())
        if self.telemetry:
            self.telemetry.terminate()
            try:
                self.telemetry.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.telemetry.kill()
                self.telemetry.wait()
        snapshot=self.state.snapshot()
        count=snapshot['counters'].get('captured_frames',0)
        span=self.last_delivery-self.first_delivery if count>1 else 0
        self.state.update(status='failed' if snapshot['errors'] or blocked else
                          'finished' if self.stop_reason=='duration' else 'stopped',
            finished_utc=utc(),stop_reason=self.stop_reason,blocked_workers=blocked,
            camera_metrics={'delivered_frames':count,'first_to_last_seconds':span,
                'measured_fps':(count-1)/span if span>0 else None,
                'max_delivery_gap_ms':1000*self.max_capture_gap if count>1 else None},
            system_ram=telemetry_summary(self.run/'tegrastats.txt'))
        self.checkpoint()
        return blocked


def render(pipeline, last_sequence):
    """Called ONLY by main. Workers never call imshow/waitKey/destroyAllWindows."""
    import cv2
    preview,answer=pipeline.state.display()
    if preview is not None and preview[0]!=last_sequence:
        sequence,frame,detections,_=preview
        view=frame.copy()
        h,w=view.shape[:2]
        for detection in detections:
            x1,y1,x2,y2=detection.box
            cv2.rectangle(view,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),(0,255,0),1)
            cv2.putText(view,detection.label,(int(x1*w),max(12,int(y1*h))),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
        cv2.putText(view,f"Scout seq {sequence} | {'S: start | ' if not pipeline.armed.is_set() else ''}Q: stop",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        if answer:
            age=time.perf_counter()-answer['completed_monotonic']+answer['capture_to_completed_answer_ms']/1000
            stale=age*1000>pipeline.cfg['inspector']['max_answer_age_ms']
            label=f"{'STALE ' if stale else ''}unverified, source age {age:.1f}s: {answer['answer']}"
            for line in range(3):
                text=label[line*80:(line+1)*80]
                cv2.putText(view,text,(8,h-45+line*15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,255),1)
        cv2.imshow('Scout & Inspector (unverified observations)',view)
        last_sequence=sequence
    key=cv2.waitKey(1)&0xff
    if key==ord('q'):
        pipeline.request_stop('preview_q')
    elif key==ord('s'):
        pipeline.arm()
    return last_sequence
