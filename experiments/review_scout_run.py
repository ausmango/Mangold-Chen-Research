"""Offline evidence extraction for a completed live run; never opens camera/CUDA."""
import argparse
import bisect
import datetime
import json
from pathlib import Path
import re
import statistics as st


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    return {'n':len(values),'mean':st.mean(values) if values else None,
            'median':st.median(values) if values else None,
            'min':min(values) if values else None,'max':max(values) if values else None}


def review(run):
    run=Path(run); client=run/'client'
    state=json.loads((client/'run.json').read_text())
    records=rows(client/'inspections.jsonl')
    answers=[r for r in records if r['status']=='complete']
    captures=rows(client/'scout.jsonl')
    times=[r['delivered_monotonic'] for r in captures]
    first=times[0];span=times[-1]-first
    active=[];idle=[];overlaps=[]
    windows=[(r['inference_started_monotonic'],r['completed_monotonic']) for r in answers]
    for r in captures:
        (active if any(a<r['delivered_monotonic']<b for a,b in windows) else idle).append(r)
    for r,(a,b) in zip(answers,windows):
        overlaps.append({'sequence':r['sequence'],'delivered_frames':bisect.bisect_left(times,b)-bisect.bisect_right(times,a),
                         'captured_at_s':r['captured_monotonic']-first,'started_at_s':a-first,'completed_at_s':b-first})
    busy=sum(b-a for a,b in windows)
    # All recorded requests are nonoverlapping and inside this capture span.
    events=sorted([(a,1) for a,b in windows]+[(b,-1) for a,b in windows])
    current=maximum=0
    for _,delta in events: current+=delta;maximum=max(maximum,current)
    def ram(path):
        text=path.read_text();values=[int(x) for x in re.findall(r'RAM (\d+)/\d+MB',text)]
        return stats(values)
    baseline=json.loads(Path('results/camera_v2-2~2026-10-05/analysis.json').read_text())
    october=json.loads(Path('results/camera_analysis_20261022T163529601772802/analysis_fast.json').read_text())
    oldmean=st.mean(r['inference_seconds'] for r in baseline['frames'])
    octmean=october['statistics']['compact16']['inference_seconds']['mean']
    newmean=st.mean(r['inference_ms'] for r in answers)/1000
    fullram=ram(run/'whole_session_tegrastats.txt')
    modelbytes=state['metadata']['inspector']['model_artifact']['bytes']
    projbytes=state['metadata']['inspector']['projector_artifact']['bytes']
    rawlog=(run/'inspector_server.log').read_text()
    def timestamp(line):
        a,b,c,d=map(int,line.split()[0].split('.'));return a*60+b+c/1000+d/1e6
    loadbegin=next(l for l in rawlog.splitlines() if 'loading model ' in l)
    loadend=next(l for l in rawlog.splitlines() if 'model loaded' in l)
    construction_ids={209,229,292,459,774} if run.name=='scout_live_20261112T181800536659876' else set()
    evidence={
      'run':str(run),'exit_status':int((run/'exit_status.txt').read_text()),'state':state,
      'capture_seconds':span,'completed_answers':len(answers),'answers_per_capture_second':len(answers)/span,
      'inference_ms':stats([r['inference_ms'] for r in answers]),
      'warm_after_first_inference_ms':stats([r['inference_ms'] for r in answers[1:]]),
      'construction_subset_sequences':[r['sequence'] for r in answers if r['sequence'] in construction_ids],
      'construction_subset_inference_ms':stats([r['inference_ms'] for r in answers if r['sequence'] in construction_ids]),
      'frame_age_ms':stats([r['frame_age_at_inference_start_ms'] for r in answers]),
      'capture_to_answer_ms':stats([r['capture_to_completed_answer_ms'] for r in answers]),
      'server_prompt_ms':stats([r['server_timings']['prompt_ms'] for r in answers]),
      'server_decode_ms':stats([r['server_timings']['predicted_ms'] for r in answers]),
      'warm_server_prompt_ms':stats([r['server_timings']['prompt_ms'] for r in answers[1:]]),
      'warm_server_decode_ms':stats([r['server_timings']['predicted_ms'] for r in answers[1:]]),
      'stale_answers':sum(r['answer_is_stale'] for r in answers),
      'length_finished_answers':sum(r['finish_reason']=='length' for r in answers),
      'output_tokens_total':sum(r['usage']['completion_tokens'] for r in answers),
      'prompt_tokens_per_request':[r['usage']['prompt_tokens'] for r in answers],
      'cached_tokens_per_request':[r['usage']['prompt_tokens_details']['cached_tokens'] for r in answers],
      'overlap_per_answer':overlaps,'max_overlapping_inspector_calls':maximum,
      'active_inference_seconds':busy,'inference_occupancy_fraction':busy/span,
      'captured_during_inference':len(active),'captured_outside_inference':len(idle),
      'delivery_rate_during_inference_hz':len(active)/busy,
      'delivery_rate_outside_inference_hz':len(idle)/(span-busy),
      'detector_ms_during_inference':stats([r['detector_ms'] for r in active]),
      'detector_ms_outside_inference':stats([r['detector_ms'] for r in idle]),
      'last_answer_captured_at_s':answers[-1]['captured_monotonic']-first,
      'last_answer_completed_at_s':answers[-1]['completed_monotonic']-first,
      'no_new_completed_answer_tail_seconds':times[-1]-answers[-1]['completed_monotonic'],
      'full_session_ram_mb':fullram,'client_ram_mb':ram(client/'tegrastats.txt'),
      'server_logged_load_interval_seconds':timestamp(loadend)-timestamp(loadbegin),
      'inspector_checkpoint_bytes':modelbytes+projbytes,'inspector_checkpoint_mib':(modelbytes+projbytes)/2**20,
      'evidence_complete':all((client/r['frame_path']).is_file() and (client/r['submitted_image_path']).is_file()
                              and (client/Path(r['frame_path']).with_suffix('.json')).is_file() for r in answers),
      'comparisons_descriptive_only':{
        'oct5_mean_seconds':oldmean,'oct22_mean_seconds':octmean,
        'latency_reduction_vs_oct5_percent':100*(1-newmean/oldmean),
        'latency_reduction_vs_oct22_percent':100*(1-newmean/octmean),
        'ratio_oct5_to_current':oldmean/newmean,'ratio_oct22_to_current':octmean/newmean,
        'observed_peak_ram_difference_vs_5400_percent':100*(1-fullram['max']/5400)},
      'answers':answers}
    return evidence


def plot(e,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    windows=e['overlap_per_answer']
    for i,r in enumerate(windows):
        axes[0].broken_barh([(r['started_at_s'],r['completed_at_s']-r['started_at_s'])],(i-.35,.7),facecolors='steelblue')
    axes[0].set(yticks=range(len(windows)),yticklabels=[str(r['sequence']) for r in windows],
                xlabel='Seconds since first camera delivery',ylabel='Analyzed sequence',
                title='Inspector requests; capture continued in every blue interval',xlim=(0,e['capture_seconds']))
    axes[0].grid(axis='x',alpha=.3)
    age=[r['frame_age_at_inference_start_ms']/1000 for r in e['answers']]
    latency=[r['inference_ms']/1000 for r in e['answers']]
    axes[1].bar(range(len(age)),age,label='Frame age before inference',color='darkorange')
    axes[1].bar(range(len(age)),latency,bottom=age,label='Client inference call',color='steelblue')
    axes[1].axhline(3,color='firebrick',linestyle='--',label='Configured stale-answer threshold')
    axes[1].set(xticks=range(len(age)),xticklabels=[str(r['sequence']) for r in e['answers']],
                xlabel='Analyzed sequence',ylabel='Capture-to-answer seconds',title='Measured answer delay and its components')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'latency_timeline.png',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run');parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();e=review(args.run)
    out=Path(args.run)/('review_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    (out/'metrics.json').write_text(json.dumps(e,indent=2)+'\n')
    if args.plot: plot(e,out)
    print(out)
    print(json.dumps({k:v for k,v in e.items() if k not in ['state','answers']},indent=2))

if __name__=='__main__':main()
