"""Complete qualified local checks around the initial final selections."""
import datetime,itertools
import core as c
import validate as v
np=c.np

def vec(r):return np.array([r['amplitudes'][k] for k in c.KEYS])

def prepare():
    path=c.DATA/'polish_plan.json'
    if path.exists():return c.read(path)
    assert c.read(c.DATA/'initial_final_validation_complete.json')['completed']
    assert not (c.DATA/'banks/selected_motion.json').exists()
    selection=c.read(c.DATA/'validated_selection.json');local=c.read(c.DATA/'local_complete.json')
    checked=v.update_validation_index();index={r['id']:r for r in c.nominal()};states=[];covered=[]
    for key,cell in selection['waves'].items():
        if not cell['qualified']:continue
        row=cell['selected']['nominal'];starts=[s for s in local if s['wave']==float(key) and s['center']]
        def coverage(s):
            return s['stop']=='no_material_improvement_at_declared_resolution' and row['id'] in {s['history'][-1]['before'],s['history'][-1]['after']}
        if any(coverage(s) for s in starts):covered.append(float(key));continue
        closest=min(starts,key=lambda s:np.linalg.norm((vec(s['center'])-vec(row))/c.HIGH))
        step=min(.01,closest['step']) if closest['center']['id']==row['id'] else .01
        direction=None
        for s in starts:
            if s['center']['id']!=row['id'] or not s['history'] or s['history'][-1]['gain_mm']<=.2:continue
            h=s['history'][-1];old=index[h['before']]
            if checked.get(old['id'],{}).get('qualified') and checked[row['id']]['half']['D']>checked[old['id']]['half']['D']:
                direction=(vec(row)-vec(old)).tolist();break
        states.append(dict(wave=float(key),center=row,step=step,initial_direction=direction,history=[],stop=None))
    protocol=c.ROOT/'POLISH_PROTOCOL.md'
    protocol.write_text('''# Qualified local checks after the primary search

The primary six-poll search is complete. Its completion receipt distinguishes resolution stops from poll-budget stops. This supplement checks the initially selected, fully qualified gaits whose selected point was not already covered by a final no-material-gain poll at the declared resolution. It does not restart weaker secondary basins.

The plant, established per-wave PD, amplitudes and bounds, initialization, 1 Hz, nominal 6 s at 0.125 ms, 12 s validation, 6 s at 0.0625 ms, engineering gates, and <=10% half-step displacement disagreement remain unchanged. Nominal fixed-forward displacement is the objective. No contact loss or shape/crossover preference is used.

Start each selected gait at normalized step min(0.01, its applicable saved step); use 0.01 if the selected point differs from the saved center. Poll six signed coordinate directions. On the first supplemental poll only, if the immediately preceding successful move has both endpoints fully qualified and also improves the finer-step displacement, test one, two and four times that successful displacement beyond the current point, clipped to the same bounds. The finer-step direction check guides proposals only; it does not change selection criteria. Archive the exact starting state and tasks.

For every nominal candidate that improves its current qualified center by more than 0.2 mm, complete any missing long/half-step checks. A nominally faster point cannot move the center unless fully qualified. Existing exact conditions and qualifications are reused. Lower nominal scores cannot improve the declared objective and need no extra qualification in this supplement. Accept the best fully qualified improvement. With no qualified gain above 0.2 mm, halve the step down to 0.005; at 0.005 record no material qualified improvement in the tested directions. This is a directional resolution check, not a continuous/global optimality proof. At most six supplemental polls are allowed; any remaining budget stop stays explicit.

The initial ranked selection and primary search records are retained. The current selection is refreshed from every completed qualified evaluation, with the count and maximum nominal advantage of faster untested candidates retained. All selected trajectory replays and final reporting occur afterward.
''',encoding='utf-8')
    c.write(c.DATA/'selection_before_polish.json',selection)
    plan=dict(time=datetime.datetime.now().astimezone().isoformat(),max_polls=6,material_gain_mm=.2,
        protocol=str(protocol),protocol_sha256=c.sha(protocol),initial_selection_sha256=c.sha(c.DATA/'selection_before_polish.json'),
        already_covered_waves=covered,states=states,physics_scoring_or_gates_changed=False)
    c.write(path,plan);return plan

def refresh_selection():
    checked=v.update_validation_index();rows=c.nominal();waves={}
    for wave in c.WAVES:
        rr=sorted([r for r in rows if r['wave']==wave and r['feasible']],key=lambda r:r['D'],reverse=True)
        selected=next((checked[r['id']] for r in rr if checked.get(r['id'],{}).get('qualified')),None)
        pending=[r for r in rr if r['id'] not in checked and (selected is None or r['D']>selected['nominal']['D'])]
        waves[f'{wave:g}']=dict(selected=selected,qualified=selected is not None,nominal_feasible_count=len(rr),
            candidates_tested=sum(r['id'] in checked for r in rr),higher_untested=len(pending),
            higher_untested_max_gain_mm=max([r['D']-selected['nominal']['D'] for r in pending],default=0.) if selected else None)
    result=dict(waves=waves,global_optimum_proved=False,validation_completed_for_declared_candidates=True,
        primary_selection_retained=str(c.DATA/'selection_before_polish.json'),qualified_polish_complete=True)
    c.write(c.DATA/'validated_selection.json',result);return result

def main():
    plan=prepare();states=plan['states'];axes=[sign*np.eye(3)[j] for j in range(3) for sign in [-1,1]]
    assert c.sha(c.DATA/'selection_before_polish.json')==plan['initial_selection_sha256']
    for poll in range(plan['max_polls']):
        assert c.sha(plan['protocol'])==plan['protocol_sha256']
        saved=c.DATA/f'polish_after_{poll:02d}.json'
        if saved.exists():states=c.read(saved);continue
        tasks=[];ids={}
        for s in states:
            if s['stop']:continue
            points=[np.clip(vec(s['center'])+s['step']*c.HIGH*d,0,c.HIGH) for d in axes]
            if poll==0 and s['initial_direction'] is not None:
                points += [np.clip(vec(s['center'])+multiple*np.array(s['initial_direction']),0,c.HIGH) for multiple in [1,2,4]]
            proposed=[c.task(s['wave'],x,f'polish_nominal_{poll:02d}') for x in points]
            ids[s['wave']]=list(dict.fromkeys(c.identity(t) for t in proposed));tasks.extend(proposed)
        if not tasks:break
        got={r['id']:r for r in c.bank(tasks,f'polish_nominal_{poll:02d}',12)}
        checked=v.update_validation_index();qualification=[]
        for s in states:
            if s['stop']:continue
            for key in ids[s['wave']]:
                r=got[key]
                if r['feasible'] and r['D']>s['center']['D']+.2 and key not in checked:
                    qualification += v.pair_tasks(r,f'polish_validation_{poll:02d}')
        bank=c.DATA/'banks'/f'polish_validation_{poll:02d}.json'
        if bank.exists():qualification=c.read(bank)
        if qualification:c.bank(qualification,f'polish_validation_{poll:02d}',12)
        checked=v.update_validation_index()
        for s in states:
            if s['stop']:continue
            before=s['center'];qualified=[got[key] for key in ids[s['wave']] if checked.get(key,{}).get('qualified')]
            best=max([before,*qualified],key=lambda r:r['D']);gain=best['D']-before['D']
            s['history'].append(dict(poll=poll,step=s['step'],before=before['id'],after=best['id'],gain_mm=gain,
                proposed_ids=ids[s['wave']],qualified_improver_ids=[r['id'] for r in qualified if r['D']>before['D']+.2]))
            s['center']=best
            if gain<=.2:
                if s['step']<=.005+1e-12:s['stop']='no_material_qualified_improvement_at_declared_resolution'
                else:s['step']/=2
        c.write(saved,states)
        print('Qualified polish poll complete: '+str([(s['wave'],s['center']['D'],s['step'],s['stop']) for s in states]),flush=True)
    for s in states:
        if not s['stop']:s['stop']='supplemental_poll_budget_reached_not_converged'
    c.write(c.DATA/'polish_complete.json',dict(completed=True,states=states,already_covered_waves=plan['already_covered_waves']))
    result=refresh_selection()
    print('Qualified polish complete: '+str({w:cell['selected']['nominal']['D'] if cell['qualified'] else None for w,cell in result['waves'].items()}),flush=True)

if __name__=='__main__':main()
