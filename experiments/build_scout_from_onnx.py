"""Build a local engine without loading PyTorch, using installed trtexec."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def wrap_engine(raw,output,metadata):
    header=json.dumps(metadata).encode()
    with Path(output).open('xb') as stream,Path(raw).open('rb') as source:
        stream.write(len(header).to_bytes(4,'little'))
        stream.write(header)
        for chunk in iter(lambda:source.read(1024*1024),b''): stream.write(chunk)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx',default='models/scout/source/yolo11n.onnx')
    parser.add_argument('--output',default='models/scout/yolo11n-fp16.engine')
    parser.add_argument('--trtexec',default='/usr/src/tensorrt/bin/trtexec')
    args=parser.parse_args()
    source,output=Path(args.onnx),Path(args.output)
    if not source.is_file(): parser.error('Need the existing local ONNX file; no automatic export/download')
    if output.exists() or output.with_suffix('.json').exists(): parser.error('Refusing to overwrite engine/provenance')
    output.parent.mkdir(parents=True,exist_ok=True)
    # ONNX metadata parsing is CPU-only and does not import Torch.
    import onnx
    model=onnx.load(str(source),load_external_data=False)
    values={item.key:item.value for item in model.metadata_props}
    names=ast.literal_eval(values['names'])
    metadata={'names':names,'stride':int(values.get('stride','32')),'task':'detect',
              'batch':1,'imgsz':[640,640]}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    raw=output.with_name(output.name+'.'+stamp+'.raw.partial')
    if raw.exists(): parser.error('Partial raw engine exists; preserve it in the failed result folder before retrying')
    cmd=[args.trtexec,'--onnx='+str(source),'--saveEngine='+str(raw),'--fp16',
         '--memPoolSize=workspace:128','--builderOptimizationLevel=0','--avgTiming=1','--skipInference']
    print('Standalone TensorRT build; PyTorch is not loaded. Camera stays closed.',flush=True)
    print('Command:', ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True,timeout=600)
    if not raw.is_file() or raw.stat().st_size==0: raise RuntimeError('No valid serialized engine produced')
    # Metadata prefix is required for the Ultralytics loader to retain COCO labels.
    wrapped=output.with_name(output.name+'.'+stamp+'.wrapped.partial')
    wrap_engine(raw,wrapped,metadata)
    if output.exists(): raise RuntimeError('Another build created output; refusing overwrite')
    wrapped.rename(output)
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'builder':'standalone trtexec, no PyTorch process','command':cmd,
              'precision':'requested FP16, some layers may remain FP32',
              'source_onnx':str(source),'source_onnx_sha256':sha(source),
              'engine':str(output),'engine_sha256':sha(output),'engine_bytes':output.stat().st_size,
              'note':'Low builder effort/workspace prioritizes a first working test; not a maximal-throughput tuned engine. Build profiling still executes GPU kernels despite skipInference.'}
    with output.with_suffix('.json').open('x') as stream: json.dump(manifest,stream,indent=2)
    print('Engine ready:',output,flush=True)

if __name__=='__main__': main()
