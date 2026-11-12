"""Download pinned assets/source only. No camera, model loading or GPU use."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HF_REV='ccd7aae53bcb1997355c2f094959e72b3642ce17'
LLAMA_REV='2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c'

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        print(f'Existing artifact retained: {path}',flush=True)
    else:
        partial=path.with_suffix(path.suffix+'.partial')
        print(f'Downloading {url}',flush=True)
        with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as stream:
            while True:
                chunk=response.read(4*1024*1024)
                if not chunk: break
                stream.write(chunk)
        partial.rename(path)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
    return {'path':str(path.relative_to(ROOT)),'url':url,'bytes':path.stat().st_size,'sha256':digest.hexdigest()}

def main():
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'inspector_id':'ggml-org/SmolVLM2-500M-Video-Instruct-GGUF',
              'inspector_revision':HF_REV,'llama_cpp_revision':LLAMA_REV,'artifacts':[]}
    for name,local in [('SmolVLM2-500M-Video-Instruct-f16.gguf','model-f16.gguf'),
                       ('mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf','mmproj.gguf')]:
        manifest['artifacts'].append(download(f'https://huggingface.co/{manifest["inspector_id"]}/resolve/{HF_REV}/{name}',ROOT/'models/inspector'/local))
    manifest['artifacts'].append(download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',ROOT/'models/scout/source/yolo11n.pt'))
    archive=ROOT/f'dependencies/llama.cpp-{LLAMA_REV}.tar.gz'
    manifest['artifacts'].append(download(f'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{LLAMA_REV}',archive))
    target=ROOT/'dependencies/llama.cpp'
    marker=target/'PROVISIONED_REVISION'
    if not marker.exists():
        if any(target.iterdir()): raise RuntimeError('Refusing to overwrite existing llama.cpp source')
        with tarfile.open(archive) as tf:
            prefix=f'llama.cpp-{LLAMA_REV}/'
            for member in tf.getmembers():
                if not member.name.startswith(prefix): continue
                relative=member.name[len(prefix):]
                if not relative: continue
                if member.issym() or member.islnk(): raise RuntimeError('Unexpected archive link')
                destination=(target/relative).resolve()
                if target.resolve() not in destination.parents: raise RuntimeError('Unsafe archive path')
                member.name=relative
                tf.extract(member,target)
        marker.write_text(LLAMA_REV+'\n')
    elif marker.read_text().strip()!=LLAMA_REV:
        raise RuntimeError('Existing llama.cpp revision differs')
    path=ROOT/'models/provision-manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['inspector_revision']!=HF_REV: raise RuntimeError('Existing provenance differs')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    config_path=ROOT/'configs/scout_inspector.local.json'
    if not config_path.exists():
        cfg=json.loads((ROOT/'configs/scout_inspector.example.json').read_text())
        cfg['inspector']['revision']=HF_REV
        cfg['triggers']['presence_classes']=['person']
        cfg['triggers']['proximity_enabled']=False
        config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__': main()
