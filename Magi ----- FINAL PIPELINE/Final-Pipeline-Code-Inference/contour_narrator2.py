"""Plain functions copied from testContourNarrator2; no model loading or notebook execution.
Pipeline settings can override CHECK, REGION and FAINT_REGION before analyze_page.
"""
import math
import importlib.util
from pathlib import Path
import cv2
import numpy as np

path=Path(__file__).with_name('contour_grouping.py')
spec=importlib.util.spec_from_file_location('full_contour_grouping',path)
contours=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contours)
CHECK=dict(contours.CHECK,expand=.20)
COLOR_TOLERANCE=25
MIN_SOLID_FRACTION=.85
REGION=dict(min_text_coverage=.90,min_outside_fraction=.15,min_area_ratio=1.15,
            min_white_fraction=.85,min_solid_fraction=MIN_SOLID_FRACTION,color_tolerance=COLOR_TOLERANCE,
            min_visible_points=16,min_curved_fraction=.12,min_curve_run=6,min_sharp_fraction=.12,min_sharp_run=6,
            curve_arc_text_scale=.35,curve_arc_min_px=10,curve_turn_min_deg=20,
            curve_direction_consistency=.65,curve_bulge_text_scale=.025,
            rectangle_fill=.94,box_axis_tolerance_deg=12,box_axis_fraction=.90,edge_margin=2)
FAINT_REGION=dict(local_window=15,min_contrast=7,max_line_gray=245,close_kernel=3,canny_low=8,canny_high=24,
                  suspicious_crop_fraction=.25,min_visible_support=.65,
                  caption_crop_fraction=.10,caption_visible_fraction=.80)

def union_box(boxes):
    return [min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes)]

def longest_run(values):
    # Boundary samples are circular; a curve can cross the first/last sample.
    longest=current=0
    for value in np.r_[values,values]:
        current=current+1 if value else 0
        longest=max(longest,current)
    return min(longest,len(values))

def box_like_profile(contour,gray,evidence=None):
    # Simplify small jagged edges, then look for two perpendicular families of straight sides.
    perimeter=cv2.arcLength(contour,True)
    polygon=cv2.approxPolyDP(contour,max(1.,.008*perimeter),True)[:,0].astype(float)
    starts=polygon;ends=np.roll(polygon,-1,axis=0)
    vectors=ends-starts;lengths=np.linalg.norm(vectors,axis=1)
    height,width=gray.shape;edge=REGION['edge_margin']
    nearby=cv2.dilate(cv2.Canny(gray,25,80) if evidence is None else evidence,np.ones((5,5),np.uint8))>0
    valid=lengths>=max(4.,.02*perimeter)
    # Only use image-supported segments inside the crop; crop borders are never box evidence.
    for i,(a,b) in enumerate(zip(starts,ends)):
        samples=np.rint(a[None,:]+np.linspace(0,1,9)[:,None]*(b-a)).astype(int)
        inside=((samples[:,0]>edge)&(samples[:,0]<width-1-edge)&
                (samples[:,1]>edge)&(samples[:,1]<height-1-edge))
        valid[i] &= bool(inside.all() and nearby[samples[:,1],samples[:,0]].mean()>=.75)
    if np.count_nonzero(valid)<3:return False,0.
    angles=np.degrees(np.arctan2(vectors[valid,1],vectors[valid,0]))%180
    weights=lengths[valid];best=0.;box_like=False
    for axis in angles:
        difference=np.abs((angles-axis+90)%180-90)
        parallel=difference<=REGION['box_axis_tolerance_deg']
        perpendicular=np.abs(difference-90)<=REGION['box_axis_tolerance_deg']
        fraction=float(weights[parallel|perpendicular].sum()/weights.sum())
        best=max(best,fraction)
        # Both directions must contribute; one long straight edge is insufficient.
        if (fraction>=REGION['box_axis_fraction'] and weights[parallel].sum()/weights.sum()>=.15
            and weights[perpendicular].sum()/weights.sum()>=.15):box_like=True
    return box_like,best

def substantial_curve(profile,visible,text_scale):
    # Measure real smooth arcs in pixels, relative to member text size, not total group perimeter.
    points=profile['boundary_points']
    smooth=sum(weight*np.roll(points,shift,axis=0)
        for shift,weight in [(-2,1),(-1,4),(0,6),(1,4),(2,1)])/16
    curved=visible & (profile['boundary_labels']=='curved')
    tangent=np.roll(smooth,-1,axis=0)-np.roll(smooth,1,axis=0)
    angles=np.arctan2(tangent[:,1],tangent[:,0])
    count=len(points);starts=np.flatnonzero(curved & ~np.roll(curved,1))
    if curved.all():starts=np.array([0])
    best=dict(curve_arc_px=0.,curve_turn_deg=0.,curve_bulge_px=0.,curve_consistency=0.)
    substantial=False
    for start in starts:
        indices=[]
        for step in range(count):
            index=(int(start)+step)%count
            if not curved[index]:break
            indices.append(index)
        if len(indices)<3:continue
        arc=smooth[indices]
        length=float(np.linalg.norm(np.diff(arc,axis=0),axis=1).sum())
        turns=np.diff(np.unwrap(angles[indices]))
        turn=float(abs(np.degrees(turns.sum())))
        consistency=float(abs(turns.sum())/max(1e-8,np.abs(turns).sum()))
        chord=arc[-1]-arc[0];offset=arc-arc[0]
        bulge=float(np.max(np.abs(chord[0]*offset[:,1]-chord[1]*offset[:,0]))/max(1e-8,np.linalg.norm(chord)))
        accepted=(length>=max(REGION['curve_arc_min_px'],REGION['curve_arc_text_scale']*text_scale)
                  and turn>=REGION['curve_turn_min_deg']
                  and consistency>=REGION['curve_direction_consistency']
                  and bulge>=max(1.,REGION['curve_bulge_text_scale']*text_scale))
        if accepted or (not substantial and length>best['curve_arc_px']):
            best=dict(curve_arc_px=length,curve_turn_deg=turn,curve_bulge_px=bulge,curve_consistency=consistency)
        substantial=substantial or accepted
    return bool(substantial),best

def region_pass(candidate,page_image,text_boxes,target_index,use_faint=False):
    target_indices=[target_index] if isinstance(target_index,int) else list(target_index)
    a,b,c,d=candidate['box'];dx=(c-a)*CHECK['expand'];dy=(d-b)*CHECK['expand']
    bounds=[max(0,math.floor(a-dx)),max(0,math.floor(b-dy)),
            min(page_image.width,math.ceil(c+dx)),min(page_image.height,math.ceil(d+dy))]
    crop=np.asarray(page_image.crop(bounds));gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    height,width=gray.shape
    target=np.zeros_like(gray,np.uint8);excluded=np.zeros_like(gray,np.uint8);member_masks=[]
    for i,box in enumerate(text_boxes):
        a,b,c,d=box
        x1,y1=max(0,math.floor(a-bounds[0])),max(0,math.floor(b-bounds[1]))
        x2,y2=min(width,math.ceil(c-bounds[0])),min(height,math.ceil(d-bounds[1]))
        if x2<=x1 or y2<=y1:continue
        excluded[y1:y2,x1:x2]=255
        if i in target_indices:
            target[y1:y2,x1:x2]=255
            member=np.zeros_like(gray,np.uint8);member[y1:y2,x1:x2]=255;member_masks.append(member)
    target_area=max(1,np.count_nonzero(target))
    samples=crop[target>0]
    if not len(samples):return dict(crop=crop,bounds=bounds,regions=[],best=None,barrier=np.zeros_like(gray))
    # Find candidate white and dominant-color backgrounds; no enclosing ink outline is needed.
    binaries=[np.uint8(gray>=CHECK['white_threshold'])*255]
    quantized=samples.astype(np.int32)//32
    colors,counts=np.unique(quantized,axis=0,return_counts=True)
    for index in np.argsort(counts)[-3:]:
        color=np.median(samples[(quantized==colors[index]).all(axis=1)],axis=0)
        similar=np.linalg.norm(crop.astype(float)-color,axis=2)<=REGION['color_tolerance']
        binaries.append(np.uint8(similar)*255)
    # Visible color/ink transitions support shape; image borders never do.
    edges=cv2.Canny(gray,25,80)
    barrier=np.zeros_like(gray)
    if use_faint:
        # Lines only need to be darker than their local background, not globally black.
        window=FAINT_REGION['local_window']
        background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((window,window),np.uint8))
        contrast=background.astype(np.int16)-gray.astype(np.int16)
        barrier=np.uint8((gray<CHECK['dark_threshold']) |
            ((contrast>=FAINT_REGION['min_contrast']) & (gray<=FAINT_REGION['max_line_gray'])))*255
        # Low-gradient edges bridge thin grey arcs missed by the local-contrast mask.
        faint_edges=cv2.Canny(gray,FAINT_REGION['canny_low'],FAINT_REGION['canny_high'])
        barrier=cv2.bitwise_or(barrier,faint_edges)
        kernel=FAINT_REGION['close_kernel']
        barrier=cv2.morphologyEx(barrier,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
        edges=cv2.bitwise_or(edges,barrier)
    nearby=cv2.dilate(edges,np.ones((5,5),np.uint8))>0
    records=[]
    for binary in binaries:
        # On retry, never close the WHITE mask across a faint outline.
        repaired=cv2.bitwise_and(binary,cv2.bitwise_not(barrier)) if use_faint else cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,components,stats,_=cv2.connectedComponentsWithStats(repaired)
        ids=np.unique(components[(target>0)&(repaired>0)])
        for component_id in ids:
            if component_id==0:continue
            component=np.uint8(components==component_id)*255
            found,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not found:continue
            contour=max(found,key=cv2.contourArea)
            mask=np.zeros_like(gray,np.uint8)
            cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)  # Fill letter-sized holes for coverage only.
            coverage=np.count_nonzero((mask>0)&(target>0))/target_area
            # Every member must be covered; a large text must not hide a missed small one.
            if any(np.count_nonzero((mask>0)&(member>0))/max(1,np.count_nonzero(member))<REGION['min_text_coverage'] for member in member_masks):continue
            area=np.count_nonzero(mask)
            outside_fraction=np.count_nonzero((mask>0)&(target==0))/max(1,area)
            if (coverage<REGION['min_text_coverage'] or area<target_area*REGION['min_area_ratio']
                or outside_fraction<REGION['min_outside_fraction']):continue
            usable=(cv2.erode(mask,np.ones((3,3),np.uint8))>0)&(excluded==0)
            pixels=crop[usable].astype(float)
            if len(pixels)<20:continue
            white_fraction=float((gray[usable]>=CHECK['white_threshold']).mean())
            median=np.median(pixels,axis=0)
            solid_fraction=float((np.linalg.norm(pixels-median,axis=1)<=REGION['color_tolerance']).mean())
            white=white_fraction>=REGION['min_white_fraction']
            if not white and solid_fraction<REGION['min_solid_fraction']:continue
            profile=contours.boundary_profile(contour,CHECK)
            points=np.rint(profile['boundary_points']).astype(int)
            edge=REGION['edge_margin']
            on_edge=((points[:,0]<=edge)|(points[:,1]<=edge)|
                     (points[:,0]>=width-1-edge)|(points[:,1]>=height-1-edge))
            # Chord-based curvature near crop corners would be artificial; exclude that neighborhood too.
            near_edge=np.maximum.reduce([np.roll(on_edge,shift) for shift in range(-CHECK['curve_window'],CHECK['curve_window']+1)])
            visible=~near_edge & nearby[points[:,1],points[:,0]]
            labels=profile['boundary_labels']
            curved=visible & (labels=='curved')
            flat=visible & (labels=='flat')
            sharp=visible & (labels=='sharp')
            visible_count=np.count_nonzero(visible)
            curve_fraction=np.count_nonzero(curved)/max(1,visible_count)
            flat_fraction=np.count_nonzero(flat)/max(1,visible_count)
            sharp_fraction=np.count_nonzero(sharp)/max(1,visible_count)
            sharp_run=longest_run(sharp)
            rect=cv2.minAreaRect(contour);fill=cv2.contourArea(contour)/max(1,rect[1][0]*rect[1][1])
            curve_run=longest_run(curved)
            enough=visible_count>=REGION['min_visible_points']
            rounded=enough and curve_fraction>=REGION['min_curved_fraction'] and curve_run>=REGION['min_curve_run']
            # Rectangular boxes keep their caption rule; ordinary rectangle corners are not speech evidence.
            box_like,box_axis_fraction=box_like_profile(contour,gray,edges)
            rectangular=enough and box_like  # Fill ratio alone is NOT caption evidence.
            sharp_sides=enough and sharp_fraction>=REGION['min_sharp_fraction'] and sharp_run>=REGION['min_sharp_run']
            expressive=rounded or sharp_sides
            # A long curved side beats a box-like approximation; ordinary small corners do not.
            text_scale=float(np.median([min(text_boxes[i][2]-text_boxes[i][0],text_boxes[i][3]-text_boxes[i][1]) for i in target_indices]))
            strong_curves,arc_metrics=substantial_curve(profile,visible,text_scale)
            if white and strong_curves:
                role,reason='SPOKEN','white region with substantial curved sides; overrides box-like approximation'
            elif rectangular:
                role='NARRATED'
                reason='box-like region with straight perpendicular sides' if box_like else 'white/solid region with mainly straight rectangular sides'
            elif white and expressive:
                role='SPOKEN'
                reason='white region with meaningful visible curved sides' if rounded else 'white region with meaningful visible sharp edges'
            elif expressive:role,reason='NARRATED','curved/sharp solid-color region; not white'
            else:role,reason='NARRATED','region exists; insufficient visible shape evidence'
            support=visible_count/max(1,np.count_nonzero(~near_edge))
            crop_fraction=float(on_edge.mean())
            visible_fraction=visible_count/len(points)
            leaked=crop_fraction>FAINT_REGION['suspicious_crop_fraction'] and visible_fraction<FAINT_REGION['min_visible_support']
            positive=bool((white and (expressive or strong_curves)) or rectangular)
            # A caption claim needs enough actual sides; panel/crop fragments must be retried.
            caption_uncertain=bool(role=='NARRATED' and rectangular and crop_fraction>FAINT_REGION['caption_crop_fraction']
                                   and visible_fraction<FAINT_REGION['caption_visible_fraction'])
            review=bool(leaked or caption_uncertain or not positive)
            if leaked:role,reason='NARRATED','region may have leaked outside the bubble; uncertain extraction'
            elif caption_uncertain:role,reason='NARRATED','box-like evidence is incomplete near crop borders; requires faint retry'
            records.append(dict(mask=mask,usable=usable,contour=contour,points=points,visible=visible,
                curved=curved,flat=flat,sharp=sharp,on_edge=on_edge,coverage=float(coverage),area=int(area),
                white_fraction=white_fraction,solid_fraction=solid_fraction,outside_fraction=float(outside_fraction),
                curved_fraction=float(curve_fraction),flat_fraction=float(flat_fraction),sharp_fraction=float(sharp_fraction),sharp_run=int(sharp_run),box_axis_fraction=float(box_axis_fraction),box_like=bool(box_like),strong_curves=bool(strong_curves),
                visible_points=int(visible_count),curve_run=int(curve_run),rectangularity=float(fill),
                **arc_metrics,text_role=role,reason=reason,shape_decided=bool(positive and not review),needs_review=review,
                crop_edge_fraction=crop_fraction,visible_support=float(support),visible_fraction=float(visible_fraction),leak_suspected=bool(leaked),caption_uncertain=caption_uncertain,
                extraction_pass='faint_barrier' if use_faint else 'normal'))
    # Prefer a shape-supported region, then coverage and the tighter background mass.
    records.sort(key=lambda r:(not r['shape_decided'],-r['coverage'],r['area']))
    return dict(crop=crop,bounds=bounds,regions=records,best=records[0] if records else None,barrier=barrier)

def find_background_regions(candidate,page_image,text_boxes,target_index):
    first=region_pass(candidate,page_image,text_boxes,target_index)
    # Keep confident first-pass decisions; retry only missing, uncertain or leaking regions.
    if first['best'] is not None and not first['best']['needs_review']:
        first.update(retry_attempted=False,first_pass=first['best']['reason'])
        return first
    retry=region_pass(candidate,page_image,text_boxes,target_index,use_faint=True)
    chosen=retry if retry['best'] is not None else first
    chosen.update(retry_attempted=True,first_pass=first['best']['reason'] if first['best'] else 'no qualifying region',
                  retry_reason=retry['best']['reason'] if retry['best'] else 'no qualifying region after faint barriers',
                  retry_barrier=retry['barrier'])
    return chosen

def best_option(options):
    options.sort(key=lambda o:(not o['region']['shape_decided'],-o['region']['coverage'],o['region']['area'],str(o['candidate_id'])))
    return options[0] if options else None

def analyze_page(candidates,image,texts,PAGE_NUMBER):
    """Group surviving OCR, label shared/separate regions, and return serializable rows plus viewer masks."""
    grouping_records=[dict(t,delete_ocr=False) for t in texts]
    page_groups=contours.inspect_page_groups(candidates,image,grouping_records,CHECK)
    lookup={t['box_id']:i for i,t in enumerate(texts)}
    units=[];assigned=set()
    memberships={bid:[] for bid in lookup}
    for proposal in page_groups['merge_candidates']:
        for bid in proposal['text_ids']:
            if bid in memberships:memberships[bid].append(proposal['group_id'])
    for proposal in page_groups['merge_candidates']:
        # First proposal wins deterministically; ambiguous membership remains visible for review.
        indices=sorted((lookup[bid] for bid in proposal['text_ids'] if bid in lookup and bid not in assigned),
                       key=lambda i:texts[i]['text_index'])
        if len(indices)<2:continue
        ids=[texts[i]['box_id'] for i in indices];assigned.update(ids)
        units.append(dict(group_id=f"PG{PAGE_NUMBER}:"+proposal['group_id'],member_indices=indices,
            candidate_ids=proposal['candidate_ids'],grouping_source='overlap_candidate',
            needs_review=any(len(memberships[bid])>1 for bid in ids)))
    for i,text in enumerate(texts):
        if text['box_id'] not in assigned:
            units.append(dict(group_id=text['box_id'],member_indices=[i],candidate_ids=[],
                grouping_source='standalone',needs_review=len(memberships[text['box_id']])>1))
    units.sort(key=lambda unit:texts[unit['member_indices'][0]]['text_index'])
    results=[];debug=[]
    text_boxes=[t['box'] for t in texts]
    for unit in units:
        indices=unit['member_indices'];members=[texts[i] for i in indices]
        box=union_box([m['box'] for m in members])
        attempts=[];options=[];selected=[]
        if unit['grouping_source']=='overlap_candidate':
            # Test the combined YOLO crop, not a rectangle drawn around the text.
            candidate=dict(box=union_box([candidates[i]['box'] for i in unit['candidate_ids']]))
            checked=find_background_regions(candidate,image,text_boxes,indices)
            attempts.append(dict(candidate_id=unit['candidate_ids'],result=checked))
            if checked['best'] is not None:
                selected=[dict(candidate_id=unit['candidate_ids'],result=checked,region=checked['best'])]
            else:
                # Disconnected masses: inspect each member in that same combined crop.
                for index in indices:
                    checked=find_background_regions(candidate,image,text_boxes,index)
                    attempts.append(dict(candidate_id=unit['candidate_ids'],result=checked,member_id=texts[index]['box_id']))
                    if checked['best'] is not None:
                        selected.append(dict(candidate_id=unit['candidate_ids'],result=checked,region=checked['best']))
        else:
            index=indices[0];area=max(1,(box[2]-box[0])*(box[3]-box[1]))
            for candidate_id,candidate in enumerate(candidates):
                if contours.overlap(box,candidate['box'])/area<.8:continue
                checked=find_background_regions(candidate,image,text_boxes,index)
                attempts.append(dict(candidate_id=candidate_id,result=checked))
                if checked['best'] is not None:
                    options.append(dict(candidate_id=candidate_id,result=checked,region=checked['best']))
            chosen=best_option(options)
            if chosen:selected=[chosen]
        chosen=selected[0] if selected else None
        row=dict(box_id=unit['group_id'],box_ids=[m['box_id'] for m in members],
            member_text_boxes=[m['box'] for m in members],text_index=members[0]['text_index'],
            text=' '.join(m['text'].strip() for m in members if m['text'].strip()),
            magi_text=' '.join(str(m['magi_text'] or '').strip() for m in members),
            paddle_text=' '.join(str(m['paddle_text'] or '').strip() for m in members),
            text_box=box,text_role='NARRATED',reason='no substantial white/solid-color region around text',candidate_id=None,
            candidate_ids=unit['candidate_ids'],grouping_source=unit['grouping_source'],needs_review=unit['needs_review'],
            contour_points=[],visible_segments=[],crop_bounds=None,ocr_agreement=members[0]['ocr_agreement'] if len(members)==1 else None,
            member_ocr_agreement=[m['ocr_agreement'] for m in members],detection_pass='group_background_region',region_predictions=[])
        if chosen:
            region=chosen['region']
            row.update({k:region[k] for k in ['text_role','reason','coverage','white_fraction','solid_fraction','outside_fraction',
                'curved_fraction','flat_fraction','sharp_fraction','sharp_run','box_axis_fraction','box_like','strong_curves',
                'visible_points','curve_run','rectangularity','curve_arc_px','curve_turn_deg','curve_bulge_px','curve_consistency','crop_edge_fraction','visible_support','visible_fraction','leak_suspected','caption_uncertain','extraction_pass']})
            row.update(candidate_id=chosen['candidate_id'],crop_bounds=chosen['result']['bounds'],
                retry_attempted=chosen['result']['retry_attempted'])
            row['needs_review']=row['needs_review'] or any(o['region']['needs_review'] for o in selected)
            for option in selected:
                region=option['region'];offset=np.array(option['result']['bounds'][:2]);points=region['points']+offset
                row['contour_points'].extend((region['contour'][:,0]+offset).tolist())
                row['visible_segments'].extend([[points[i].tolist(),points[(i+1)%len(points)].tolist()] for i in range(len(points))
                    if region['visible'][i] and region['visible'][(i+1)%len(points)]])
                row['region_predictions'].append({k:region[k] for k in ['text_role','reason','coverage','curved_fraction','sharp_fraction','box_like','strong_curves','curve_arc_px','curve_turn_deg','curve_bulge_px','curve_consistency','needs_review','extraction_pass']})
            roles={r['text_role'] for r in row['region_predictions']}
            split_attempts=[a for a in attempts if 'member_id' in a]
            missing=any(a['result']['best'] is None for a in split_attempts)
            if len(roles)>1 or missing:
                row.update(text_role='NARRATED',needs_review=True,
                    reason='group has conflicting region labels' if len(roles)>1 else 'group has members without a qualifying region')
            elif len(selected)>1:
                row['reason']='separate regions agree: '+row['text_role']
                if any(o['region']['needs_review'] for o in selected):row.update(text_role='NARRATED',reason='separate regions include uncertain shape evidence',needs_review=True)
            row['metrics_scope']='shared region' if not split_attempts else 'first member region; see region_predictions'
        elif not attempts:row['reason']='no YOLO bubble matched this surviving Magi text'
        results.append(row);debug.append(dict(attempts=attempts,chosen=chosen))
    emitted=[bid for row in results for bid in row['box_ids']]
    assert len(emitted)==len(set(emitted)) and set(emitted)=={t['box_id'] for t in texts},'Missing/duplicated surviving text'
    return dict(rows=results,grouping=page_groups),debug
