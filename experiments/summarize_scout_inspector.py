"""Summarize saved evidence only; no camera/model imports or benchmark reruns."""
import argparse
import bisect
import datetime
import json
from pathlib import Path
import re
import statistics


def lines(path):
    rows=[]
    if not path.exists(): return rows
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass  # Interrupted final JSONL row is not a completed result.
    return rows


def stats(values):
    return {'samples':len(values),'mean':statistics.mean(values) if values else None,
            'min':min(values) if values else None,'max':max(values) if values else None}


def summarize(run):
    run=Path(run)
    client=run/'client' if (run/'client').is_dir() else run
    state=json.loads((client/'run.json').read_text())
    answers=[r for r in lines(client/'inspections.jsonl') if r.get('status')=='complete']
    captures=sorted(r['delivered_monotonic'] for r in lines(client/'scout.jsonl') if 'delivered_monotonic' in r)
    overlaps=[]
    for row in answers:
        end=row['completed_monotonic']
        start=row.get('inference_started_monotonic',end-row['inference_ms']/1000)
        overlaps.append({'sequence':row['sequence'],
                         'camera_deliveries_during_inference':max(0,bisect.bisect_left(captures,end)-bisect.bisect_right(captures,start))})
    full_ram=[]
    full_log=client.parent/'whole_session_tegrastats.txt'
    if full_log.exists():
        full_ram=[int(x) for x in re.findall(r'RAM (\d+)/\d+MB',full_log.read_text(errors='replace'))]
    return {'source':str(client.resolve()),'status':state['metadata'].get('status'),
            'completed_answers':len(answers),
            'inference_ms':stats([r['inference_ms'] for r in answers]),
            'frame_age_at_inference_start_ms':stats([r['frame_age_at_inference_start_ms'] for r in answers]),
            'capture_to_completed_answer_ms':stats([r['capture_to_completed_answer_ms'] for r in answers]),
            'stale_answers':sum(bool(r.get('answer_is_stale')) for r in answers),
            'capture_overlap':overlaps,
            'answers_with_capture_overlap':sum(r['camera_deliveries_during_inference']>0 for r in overlaps),
            'camera_metrics':state['metadata'].get('camera_metrics'),
            'client_system_ram':state['metadata'].get('system_ram'),
            'whole_session_peak_system_ram_mb':max(full_ram) if full_ram else None,
            'counters':state.get('counters'), 'errors':state.get('errors'),
            'limitations':'Overlap demonstrates host-delivery scheduling, not simultaneous GPU execution, verified accuracy, sensor exposure freshness or guaranteed real-time answers.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args=parser.parse_args()
    result=summarize(args.run)
    run=Path(args.run)
    name='summary_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
    path=run/name
    with path.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    print('Summary saved:',path)

if __name__=='__main__': main()
