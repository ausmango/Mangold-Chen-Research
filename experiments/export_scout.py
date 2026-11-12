"""Explicit hardware export utility. Never invoked by pipeline/preflight."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import datetime


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True,help='Existing local YOLO11n or YOLOv8n .pt file')
    parser.add_argument('--revision',required=True,help='Exact source checkpoint release/commit provenance')
    parser.add_argument('--output',required=True)
    parser.add_argument('--precision',choices=['fp16','int8'],default='fp16')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--workspace-gib',type=float,default=.5)
    parser.add_argument('--calibration-data',help='Local dataset YAML; INT8 requires representative local images')
    args = parser.parse_args()
    source,target = Path(args.weights).resolve(),Path(args.output).resolve()
    if not source.is_file() or source.suffix != '.pt':
        parser.error('Supply an existing local .pt checkpoint; no automatic download')
    if target.suffix != '.engine' or target.exists() or target.with_suffix('.json').exists():
        parser.error('Output must be a NEW .engine path and manifest')
    if source.with_suffix('.engine').exists():
        parser.error('Export would overwrite the engine beside the checkpoint; use a fresh checkpoint directory')
    if args.imgsz <= 0 or not math.isfinite(args.workspace_gib) or args.workspace_gib <= 0:
        parser.error('Image size and workspace must be positive')
    if args.precision=='int8' and (not args.calibration_data or not Path(args.calibration_data).is_file()):
        parser.error('INT8 requires --calibration-data pointing to a local YAML dataset')
    if importlib.metadata.version('ultralytics') != '8.3.162':
        parser.error('This export recipe targets ultralytics 8.3.162; validate other versions separately')
    os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    settings = dict(format='engine',imgsz=args.imgsz,batch=1,device=0,
                    half=args.precision=='fp16',int8=args.precision=='int8',
                    workspace=args.workspace_gib,dynamic=False,simplify=False,opset=17)
    if args.precision=='int8':
        settings['data']=str(Path(args.calibration_data).resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    exported = Path(YOLO(str(source),task='detect').export(**settings)).resolve()
    if exported != target:
        shutil.move(str(exported),str(target))
    manifest = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source':str(source),'source_revision':args.revision,'source_sha256':sha(source),
                'engine':str(target),'engine_sha256':sha(target),'engine_bytes':target.stat().st_size,
                'configured_precision':args.precision,'settings':settings,
                'versions':{name:importlib.metadata.version(name) for name in ['ultralytics','torch']},
                'calibration_yaml_sha256':sha(args.calibration_data) if args.calibration_data else None,
                'note':'Engine built on this Jetson. Requested precision is not proof every layer uses it. Retain calibration dataset separately; YAML hash is not an image dataset hash.'}
    import tensorrt
    manifest['versions']['tensorrt']=tensorrt.__version__
    with target.with_suffix('.json').open('x') as stream:
        json.dump(manifest,stream,indent=2)
    print(target)


if __name__=='__main__':
    main()
