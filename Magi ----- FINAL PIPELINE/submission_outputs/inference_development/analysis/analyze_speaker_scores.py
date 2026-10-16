"""Explain saved speaker scores using the unchanged official token aligner."""
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'submission_outputs/inference_development'
DEST=OUT/'analysis'
spec=importlib.util.spec_from_file_location('official_score',ROOT/'TaskData/score.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
references=score.load_jsonl(ROOT/'TaskData/development/labels.jsonl',reference=True)
predictions=score.load_jsonl(OUT/'predictions.jsonl')
report=json.loads((ROOT/'score.json').read_text(encoding='utf-8'))


def indexed_tokens(rows):
    return [(text,speaker,index) for index,row in enumerate(rows) for text,speaker in score.tokens([row])]


def analyze(gold_pages,pred_pages):
    pairs=[];gcounts=Counter();pcounts=Counter();gall=pall=0
    for page,(gold,pred) in enumerate(zip(gold_pages,pred_pages)):
        gt,pt=indexed_tokens(gold),indexed_tokens(pred)
        gall+=len(gt);pall+=len(pt)
        gcounts.update(t[1] for t in gt if t[1] is not None)
        pcounts.update(t[1] for t in pt)
        pairs.extend((g,p,w,page) for g,p,w in score.align_tokens(gt,pt))
    for g,p,w,page in pairs:
        if g[1] is None:pcounts[p[1]]-=1
    pcounts=+pcounts
    labels=[(g[1],p[1],w) for g,p,w,page in pairs if g[1] is not None]
    mapping=score.best_mapping(labels)
    correct=sum(w for g,p,w in labels if mapping.get(p)==g)
    aligned=sum(w for g,p,w in labels)
    balanced_pairs=[(g,p,2*w/(gcounts[g]+pcounts[p])) for g,p,w in labels]
    balanced_map=score.best_mapping(balanced_pairs)
    balanced_sum=sum(w for g,p,w in balanced_pairs if balanced_map.get(p)==g)
    extras=sum(balanced_map.get(p) not in gcounts for p in pcounts)
    metrics=dict(matched_token_coverage=sum(x[2] for x in pairs)/gall if gall else (1. if not pall else 0.),
                 speaker_accuracy_on_matched=correct/aligned if aligned else (0. if gcounts else None),
                 joint_f1=2*correct/(sum(gcounts.values())+sum(pcounts.values())) if gcounts else None,
                 balanced_joint_f1=min(1.,balanced_sum/(len(gcounts)+extras)) if gcounts else None)
    return metrics,pairs,gcounts,pcounts,mapping,balanced_map,extras,aligned,correct


def save_csv(name,rows):
    if not rows:return
    with (DEST/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


summary=Counter();errors=Counter();group_status=Counter();roles=Counter();identities=[];sequences=[];group_errors=[]
raw_metrics=defaultdict(list);upper=[];pair_total=0;known_total=0;pred_total=0
for number,(sid,gold) in enumerate(references.items(),1):
    pred=predictions.get(sid,[[],[],[]])
    metrics,pairs,gcounts,pcounts,mapping,bmap,extras,aligned,correct=analyze(gold,pred)
    for key,value in metrics.items():
        expected=report['sequences'][sid][key]
        assert (value is None and expected is None) or math.isclose(value,expected,abs_tol=1e-10), (sid,key,value,expected)
    path=OUT/'sequences'/sid
    groups=json.loads((path/'group_predictions.json').read_text())
    by_page={page:sorted([g for g in groups if g['page_index']==page],key=lambda g:g['text_index']) for page in range(3)}
    assert all(len(by_page[p])==len(pred[p]) for p in range(3)),sid
    for group in groups:
        group_status[group['speaker_status']]+=1;roles[group['text_role']]+=1
        summary['groups']+=1;summary['tail_overrides']+=bool(group['tail_override'])
        summary['merged_groups']+=len(group['box_ids'])>1
    group_labels=defaultdict(Counter);group_wrong=Counter();group_weight=Counter();matrix=Counter()
    for g,p,w,page in pairs:
        if g[1] is None:continue
        matrix[g[1],p[1]]+=w;group_labels[page,p[2]][g[1]]+=w;group_weight[page,p[2]]+=w
        if mapping.get(p[1])!=g[1]:
            group_wrong[page,p[2]]+=w
            kind='narration_as_spoken' if g[1]=='NARRATION' and p[1]!='NARRATION' else 'spoken_as_narration' if g[1]!='NARRATION' and p[1]=='NARRATION' else 'wrong_spoken_identity'
            errors[kind]+=w
            group=by_page[page][p[2]]
            if kind=='narration_as_spoken' and group['tail_override']:errors['narration_as_spoken_with_tail_override']+=w
            if p[1]=='UNKNOWN':errors['wrong_tokens_predicted_UNKNOWN']+=w
    inverse={g:p for p,g in bmap.items()}
    for gold_id,count in gcounts.items():
        predicted_id=inverse.get(gold_id)
        overlap=matrix[gold_id,predicted_id] if predicted_id is not None else 0.
        value=2*overlap/(count+pcounts[predicted_id]) if predicted_id is not None else 0.
        identities.append(dict(sequence_id=sid,gold_speaker=gold_id,gold_tokens=count,predicted_speaker=predicted_id,
                               predicted_tokens=pcounts.get(predicted_id,0),matched_correct_weight=overlap,identity_f1=value))
        summary['gold_identities']+=1;summary['zero_f1_identities']+=value==0
        summary['small_identities_le_10_tokens']+=count<=10
        summary['zero_f1_small_identities']+=count<=10 and value==0
        if gold_id=='NARRATION':
            summary['sequences_with_narration']+=1;summary['zero_f1_narration_identities']+=value==0
        significant=[p for p in pcounts if matrix[gold_id,p]>=max(3,.1*count)]
        summary['fragmentation_indicators']+=gold_id!='NARRATION' and len(significant)>1
    for predicted_id,count in pcounts.items():
        significant=[g for g in gcounts if matrix[g,predicted_id]>=max(3,.1*count)]
        summary['collapse_indicators']+=predicted_id!='NARRATION' and len(significant)>1
    for (page,index),labels in group_labels.items():
        group=by_page[page][index]
        mixed=len([g for g,w in labels.items() if w>=1])>1
        summary['merged_groups_with_mixed_reference_speakers']+=len(group['box_ids'])>1 and mixed
        if group_wrong[page,index]:
            group_errors.append(dict(sequence_id=sid,box_id=group['box_id'],predicted_speaker=pred[page][index]['speaker'],
                 speaker_status=group['speaker_status'],text_role=group['text_role'],tail_override=group['tail_override'],
                 members=len(group['box_ids']),reference_speakers=json.dumps(dict(labels)),
                 wrong_matched_weight=group_wrong[page,index],matched_known_weight=group_weight[page,index],
                 reason=group['reason'],text=pred[page][index]['text']))
    summary['speaker_scored_sequences']+=bool(gcounts)
    if gcounts:
        summary['extra_predicted_identities_on_scored_sequences']+=extras
        summary['scored_sequences_with_extra_identities']+=extras>0
        for label in pcounts:
            if bmap.get(label) not in gcounts:
                category='NARRATION' if label=='NARRATION' else 'UNKNOWN' if label=='UNKNOWN' else 'character'
                summary['extra_label_type_'+category]+=1
    summary['gold_narration_entries']+=sum(row['speaker']=='NARRATION' for page in gold for row in page)
    summary['predicted_narration_entries']+=sum(row['speaker']=='NARRATION' for page in pred for row in page)
    summary['sequences_predicted_with_narration']+=any(row['speaker']=='NARRATION' for page in pred for row in page)
    summary['extra_predicted_identities']+=extras
    summary['sequences_with_extra_identities']+=extras>0
    pair_total+=aligned;known_total+=sum(gcounts.values());pred_total+=sum(pcounts.values())
    if gcounts:upper.append(2*aligned/(sum(gcounts.values())+sum(pcounts.values())))
    sequences.append(dict(sequence_id=sid,**metrics,gold_identities=len(gcounts),predicted_identities=len(pcounts),
                          extra_predicted_identities=extras,known_gold_tokens=sum(gcounts.values()),
                          aligned_known_weight=aligned,correct_speaker_weight=correct))
    raw=score.load_jsonl(path/'transcript_raw.jsonl')[sid]
    raw_values=analyze(gold,raw)[0]
    for key,value in raw_values.items():
        if value is not None:raw_metrics[key].append(value)
    if number%20==0:print('Analyzed',number,'/',len(references),flush=True)

result=dict(reference_sequences=len(references),predicted_sequences=len(predictions),missing_sequences=report['missing_sequences'],
            reported_macro=report['macro'],summary=dict(summary),group_speaker_status=dict(group_status),group_roles=dict(roles),
            matched_known_token_weight=pair_total,correct_speaker_token_weight=sum(r['correct_speaker_weight'] for r in sequences),
            wrong_matched_weight_by_type=dict(errors),known_gold_token_count=known_total,scored_predicted_token_count=pred_total,
            raw_before_cleanup_macro={k:sum(v)/len(v) for k,v in raw_metrics.items()},
            aligned_token_speaker_oracle_joint_f1_macro=sum(upper)/len(upper),
            notes=['All four token/speaker metrics reproduced the saved report per sequence. Character-order score was read from score.json.',
                   'Oracle changes speaker assignments on already aligned tokens; it is a diagnostic ceiling, not a measured model improvement.',
                   'Fragmentation/collapse indicators use overlap >= max(3 tokens, 10% of the identity token count); they do not isolate wrong links from clustering.',
                   'Mixed-reference merge flags rely on token alignment and need image inspection before attributing a cause.'],
            sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'TaskData/score.py',ROOT/'TaskData/development/labels.jsonl',OUT/'predictions.jsonl',ROOT/'score.json']})
DEST.mkdir(exist_ok=True)
(DEST/'speaker_error_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
save_csv('per_identity_f1.csv',identities)
save_csv('per_sequence_diagnostics.csv',sorted(sequences,key=lambda r:r['balanced_joint_f1'] or 0))
save_csv('speaker_error_groups.csv',sorted(group_errors,key=lambda r:r['wrong_matched_weight'],reverse=True))
print(json.dumps(result,indent=2),flush=True)
