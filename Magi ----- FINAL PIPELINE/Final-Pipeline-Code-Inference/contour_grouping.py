"""Reuse the contour checks from checkingContour.ipynb for transcript grouping only.
Individual boundaries determine which overlapping candidates need grouping.
Merged boundary success is never required; no narration/speaker overrides.
"""
import math
import cv2
import numpy as np

CHECK = dict(expand=0.20, dark_threshold=180, white_threshold=210,
             close_kernel=3, text_padding=2, min_text_coverage=0.95,
             min_white_fraction=0.85, min_border_dark_fraction=0.20,
             min_candidate_iou=0.35, max_area_ratio=1.7,
             rectangle_fill=0.88, rounded_max_fill=0.90, min_solidity=0.85,
             boundary_samples=256, curve_window=12, straight_tolerance_px=0.8,
             straight_tolerance_fraction=0.015, sharp_turn_deg=18,
             mixed_min_flat=0.15, mixed_min_curved=0.12,
             crop_edge_margin_px=1, max_crop_edge_fraction=0.02,
             min_supported_boundary_fraction=0.85, max_unsupported_fraction=0.10)

FAINT = dict(enabled=True, local_window=15, min_local_contrast=7,
             max_line_gray=245, close_kernel=3)

def boundary_profile(contour, settings):
    """Sample a closed boundary evenly; distinguish straight, smooth-curved and sharp sections."""
    points = contour[:, 0].astype(float)
    points = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, lengths > 1e-8]
    points = points[keep]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0., np.cumsum(lengths)]
    samples = np.linspace(0, distance[-1], settings['boundary_samples'], endpoint=False)
    boundary = np.column_stack([np.interp(samples, distance, points[:, axis]) for axis in [0, 1]])
    # Mild periodic smoothing suppresses pixel stair-steps without fitting an ellipse.
    smooth = sum(weight*np.roll(boundary, shift, axis=0)
                 for shift,weight in [(-2,1),(-1,4),(0,6),(1,4),(2,1)])/16
    previous, following = smooth-np.roll(smooth,1,axis=0), np.roll(smooth,-1,axis=0)-smooth
    cosine = (previous*following).sum(1)/np.maximum(1e-8,
                np.linalg.norm(previous,axis=1)*np.linalg.norm(following,axis=1))
    turning = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    window = settings['curve_window']
    a,b = np.roll(smooth,window,axis=0),np.roll(smooth,-window,axis=0)
    chord = b-a
    chord_length = np.linalg.norm(chord,axis=1)
    offset = smooth-a
    deviation = np.abs(chord[:,0]*offset[:,1]-chord[:,1]*offset[:,0])/np.maximum(1e-8,chord_length)
    tolerance = np.maximum(settings['straight_tolerance_px'],
                           settings['straight_tolerance_fraction']*chord_length)
    # A sharp corner is not itself evidence of a smooth curve. Mark its neighborhood separately.
    sharp = np.maximum.reduce([np.roll(turning,shift) for shift in range(-window,window+1)]) > settings['sharp_turn_deg']
    flat = (deviation <= tolerance) & ~sharp
    curved = (deviation > tolerance) & ~sharp
    labels = np.full(len(boundary),'sharp',dtype=object)
    labels[flat],labels[curved] = 'flat','curved'
    return dict(boundary_points=boundary,boundary_labels=labels,
                flat_fraction=float(flat.mean()),curved_fraction=float(curved.mean()),
                sharp_fraction=float(sharp.mean()))

def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))

def faint_ink_mask(gray, settings):
    """Recover gray strokes darker than their local background; never invent a crop border."""
    window=settings['local_window']
    # A local bright background exposes faint gray outlines beside white bubble interiors.
    background=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((window,window),np.uint8))
    contrast=background.astype(np.int16)-gray.astype(np.int16)
    return np.uint8((gray < settings['dark_threshold']) |
                   ((contrast >= settings['min_local_contrast']) &
                    (gray <= settings['max_line_gray'])))*255

def drawn_boundary_evidence(profile, gray, text_mask, settings):
    """Crop-border segments supply no outline evidence, even when OpenCV closes them."""
    points=np.rint(profile['boundary_points']).astype(int)
    height,width=gray.shape
    edge_margin=settings['crop_edge_margin_px']
    on_edge=((points[:,0]<=edge_margin)|(points[:,1]<=edge_margin)
             |(points[:,0]>=width-1-edge_margin)|(points[:,1]>=height-1-edge_margin))
    ink=faint_ink_mask(gray,settings)>0 if settings.get('faint_retry') else gray<settings['dark_threshold']
    raw_dark=np.uint8(ink & (text_mask==0))
    # Look for actual ink near each contour point, allowing a small white/black contour offset.
    raw_dark[:edge_margin+1,:]=0
    raw_dark[-edge_margin-1:,:]=0
    raw_dark[:,:edge_margin+1]=0
    raw_dark[:,-edge_margin-1:]=0
    nearby=cv2.dilate(raw_dark,np.ones((5,5),np.uint8))
    supported=(nearby[points[:,1],points[:,0]]>0)&~on_edge
    # Measure the longest unsupported run around the complete circular boundary.
    missing=~supported
    longest=current=0
    for value in np.r_[missing,missing]:
        current=current+1 if value else 0
        longest=max(longest,current)
    longest=min(longest,len(points))
    labels=profile['boundary_labels'].copy()
    labels[on_edge]='crop_edge'
    valid=max(1,np.count_nonzero(~on_edge))
    return dict(boundary_labels=labels,
        flat_fraction=float(np.count_nonzero(labels=='flat')/valid),
        curved_fraction=float(np.count_nonzero(labels=='curved')/valid),
        sharp_fraction=float(np.count_nonzero(labels=='sharp')/valid),
        crop_edge_fraction=float(on_edge.mean()),
        supported_boundary_fraction=float(supported.mean()),
        longest_unsupported_fraction=float(longest/len(points)),
        crop_truncated=bool(on_edge.mean()>settings['max_crop_edge_fraction']))

def check_candidate(candidate, page_image, text_boxes, settings, required_text_indices=None):
    x1,y1,x2,y2 = candidate['box']
    dx,dy = (x2-x1)*settings['expand'],(y2-y1)*settings['expand']
    bounds = [max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),
              min(page_image.width,math.ceil(x2+dx)),min(page_image.height,math.ceil(y2+dy))]
    crop = np.asarray(page_image.crop(bounds))
    gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    dark = (faint_ink_mask(gray,settings) if settings.get('faint_retry')
            else np.uint8(gray < settings['dark_threshold'])*255)
    kernel = np.ones((settings['close_kernel'],settings['close_kernel']),np.uint8)
    closed = cv2.morphologyEx(dark,cv2.MORPH_CLOSE,kernel)
    white = np.uint8(gray >= settings['white_threshold'])*255
    # All nearby text is excluded from whiteness; the candidate's contained text must be enclosed.
    text_mask = np.zeros_like(gray,np.uint8)
    required = np.zeros_like(gray,np.uint8)
    associated = []
    for index,box in enumerate(text_boxes):
        if overlap(box,bounds)<=0: continue
        a,b,c,d = box
        pad = settings['text_padding']
        left,top = max(0,math.floor(a-bounds[0]-pad)),max(0,math.floor(b-bounds[1]-pad))
        right,bottom = min(gray.shape[1],math.ceil(c-bounds[0]+pad)),min(gray.shape[0],math.ceil(d-bounds[1]+pad))
        text_mask[top:bottom,left:right]=255
        area = max(1,(c-a)*(d-b))
        if (index in required_text_indices if required_text_indices is not None
            else overlap(box,candidate['box'])/area >= 0.8):
            # Use unexpanded text rectangles for the enclosure requirement.
            left,top=max(0,math.floor(a-bounds[0])),max(0,math.floor(b-bounds[1]))
            right,bottom=min(gray.shape[1],math.ceil(c-bounds[0])),min(gray.shape[0],math.ceil(d-bounds[1]))
            required[top:bottom,left:right]=255
            associated.append(index)
    contours = []
    # On retry, trace spaces separated by recovered faint ink as well as ink contours.
    binaries=[closed,cv2.bitwise_not(closed)] if settings.get('faint_retry') else [closed,white]
    for binary in binaries:
        found,_ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    records = []
    target = [x1-bounds[0],y1-bounds[1],x2-bounds[0],y2-bounds[1]]
    target_area = max(1,(x2-x1)*(y2-y1))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25 or area > target_area*settings['max_area_ratio']: continue
        x,y,w,h = cv2.boundingRect(contour)
        # Edge-touching contours are evaluated by the same text/interior/shape checks.
        inter = overlap([x,y,x+w,y+h],target)
        iou = inter/max(1,w*h+target_area-inter)
        if iou < settings['min_candidate_iou']: continue
        mask = np.zeros_like(gray,np.uint8)
        cv2.drawContours(mask,[contour],-1,255,cv2.FILLED)
        coverage = np.count_nonzero((mask>0)&(required>0))/max(1,np.count_nonzero(required))
        interior = cv2.erode(mask,np.ones((3,3),np.uint8),iterations=2)
        usable = (interior>0)&(text_mask==0)
        whiteness = float((gray[usable]>=settings['white_threshold']).mean()) if usable.any() else 0.
        # Sample a narrow ring around the contour on the original image, not the repaired mask.
        ring = cv2.dilate(mask,np.ones((5,5),np.uint8))-cv2.erode(mask,np.ones((5,5),np.uint8))
        # Do not let an image-border segment count as a dark drawn outline.
        valid_ring=(ring>0)&(text_mask==0)
        edge=settings['crop_edge_margin_px']+1
        valid_ring[:edge,:]=False; valid_ring[-edge:,:]=False
        valid_ring[:,:edge]=False; valid_ring[:,-edge:]=False
        outline_ink=(dark>0) if settings.get('faint_retry') else gray<settings['dark_threshold']
        border_dark=float(outline_ink[valid_ring].mean()) if valid_ring.any() else 0.
        perimeter = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.025*perimeter,True)
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0]*rect[1][1]
        fill = area/max(1,rect_area)
        solidity = area/max(1,cv2.contourArea(cv2.convexHull(contour)))
        yy,xx = np.where(required>0)
        margin = float(min(xx.min()-x,yy.min()-y,x+w-1-xx.max(),y+h-1-yy.max())) if len(xx) else 0.
        reasons = []
        if not associated: reasons.append('no contained text')
        if coverage<settings['min_text_coverage']: reasons.append('does not enclose text')
        # OCR-box margin is diagnostic only: loose rectangles can touch a valid outline.
        if whiteness<settings['min_white_fraction']: reasons.append('interior not mostly white')
        if border_dark<settings['min_border_dark_fraction']: reasons.append('weak dark outline')
        profile = boundary_profile(contour,settings)
        profile.update(drawn_boundary_evidence(profile,gray,text_mask,settings))
        if profile['crop_truncated']:
            reasons.append('crop-truncated region; crop edge is not outline evidence')
        if profile['supported_boundary_fraction'] < settings['min_supported_boundary_fraction']:
            reasons.append('insufficient real drawn boundary support')
        if profile['longest_unsupported_fraction'] > settings['max_unsupported_fraction']:
            reasons.append('long unsupported boundary gap')
        # Rounded rectangles and flat-sided curved bubbles can fill almost their entire rectangle.
        mixed = (profile['flat_fraction'] >= settings['mixed_min_flat']
                 and profile['curved_fraction'] >= settings['mixed_min_curved']
                 and solidity >= settings['min_solidity'])
        rectangular = len(approx)==4 and fill>=settings['rectangle_fill'] and not mixed
        rounded = (profile['curved_fraction'] >= settings['mixed_min_curved']
                   and len(approx)>=5 and fill<settings['rounded_max_fill']
                   and solidity>=settings['min_solidity'])
        shape = 'mixed_flat_curved' if mixed else 'rectangular' if rectangular else 'rounded' if rounded else 'ambiguous'
        status = ({'mixed_flat_curved':'mixed_flat_curved_candidate','rectangular':'rectangular_candidate',
                   'rounded':'rounded_white_candidate'}.get(shape,'uncertain')) if not reasons else 'uncertain'
        if shape=='ambiguous': reasons.append('ambiguous shape')
        # Enclosure is independent of whiteness and whether the boundary is round or rectangular.
        enclosed_boundary_found = bool(associated
            and coverage >= settings['min_text_coverage']
            and border_dark >= settings['min_border_dark_fraction']
            and not profile['crop_truncated']
            and profile['supported_boundary_fraction'] >= settings['min_supported_boundary_fraction']
            and profile['longest_unsupported_fraction'] <= settings['max_unsupported_fraction'])
        score = coverage + whiteness + border_dark + iou + profile['supported_boundary_fraction'] + (1 if not reasons else 0)
        records.append(dict(enclosed_boundary_found=enclosed_boundary_found,**profile,contour=contour,mask=mask,usable=usable,score=score,
            coverage=coverage,white_fraction=whiteness,border_dark_fraction=border_dark,
            margin_px=margin,vertices=len(approx),rectangularity=fill,solidity=solidity,
            candidate_iou=iou,shape=shape,status=status,reasons='; '.join(reasons) or 'passed provisional checks'))
    best = max(records,key=lambda r:r['score']) if records else None
    boundaries = [r for r in records if r['enclosed_boundary_found']]
    boundary_best = max(boundaries,key=lambda r:r['score']) if boundaries else None
    return dict(crop=crop,dark=dark,closed=closed,text_mask=text_mask,required=required,
                bounds=bounds,associated_text_indices=associated,contours=records,best=best,
                enclosed_boundary_found=bool(boundaries),boundary_best=boundary_best,
                boundary_kind='drawn_enclosed' if boundaries else 'crop_truncated' if best and best['crop_truncated'] else 'not_verified')

def check_with_faint_retry(candidate, page_image, text_boxes, settings, required_text_indices=None):
    """Keep first-pass successes intact; retry failures on the SAME crop with local contrast."""
    first=check_candidate(candidate,page_image,text_boxes,settings,required_text_indices)
    first['detection_pass']='first'
    first['faint_retry_attempted']=False
    if first['enclosed_boundary_found'] or not FAINT['enabled']:
        return first
    retry_settings=dict(settings, **{k:v for k,v in FAINT.items() if k!='enabled'})
    retry_settings['faint_retry']=True
    retry=check_candidate(candidate,page_image,text_boxes,retry_settings,required_text_indices)
    # Preserve the failed first result for comparison, even when the retry recovers an outline.
    first['faint_retry_attempted']=True
    first['faint_retry_result']=retry
    if not retry['enclosed_boundary_found']:
        return first
    retry['detection_pass']='faint_retry'
    retry['faint_retry_attempted']=True
    retry['first_pass_result']=first
    return retry

def overlap_merge_groups(candidate_list, individual_checks):
    """Include verified neighbors; skip groups where every member already has a boundary."""
    neighbors = {i:set() for i in range(len(candidate_list))}
    groups = set()
    for i in range(len(candidate_list)):
        for j in range(i+1,len(candidate_list)):
            # Original YOLO rectangles must overlap in area; padding alone does not create a link.
            if overlap(candidate_list[i]['box'],candidate_list[j]['box']) > 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                if not all(individual_checks[k]['enclosed_boundary_found'] for k in [i,j]):
                    groups.add((i,j))
    # Also try the full connected group, allowing a chain of three or more related crops.
    visited=set()
    for start in neighbors:
        if start in visited: continue
        stack,component=[start],set()
        while stack:
            current=stack.pop()
            if current in component: continue
            component.add(current)
            stack.extend(neighbors[current]-component)
        visited.update(component)
        if len(component)>2 and not all(individual_checks[k]['enclosed_boundary_found'] for k in component):
            groups.add(tuple(sorted(component)))
    return sorted(groups,key=lambda members:(len(members),members))
def inspect_page_groups(candidates,page_image,records,settings):
    """Only surviving Magi texts take part. No merged-boundary check is required."""
    survivors=[r for r in records if not r['delete_ocr']]
    texts=[r['box'] for r in survivors]
    kept=[]
    for original_id,candidate in enumerate(candidates):
        # Ignore a YOLO detection unless it covers a surviving Magi text box.
        matched=any(overlap(t,candidate['box'])/max(1,(t[2]-t[0])*(t[3]-t[1]))>=.8 for t in texts)
        if matched:kept.append(dict(candidate,original_candidate_id=original_id))
    checks=[check_with_faint_retry(c,page_image,texts,settings) for c in kept]
    individual=[]
    for candidate,result in zip(kept,checks):
        best=result['boundary_best'] or result['best']
        points=(best['contour'][:,0]+np.array(result['bounds'][:2])).tolist() if best else []
        individual.append(dict(candidate_id=candidate['original_candidate_id'],box=candidate['box'],
            text_ids=[survivors[t]['box_id'] for t in result['associated_text_indices']],
            boundary_found=result['enclosed_boundary_found'],detection_pass=result['detection_pass'],
            faint_retry_attempted=result['faint_retry_attempted'],contour_points=points,
            reasons=best['reasons'] if best else 'no qualifying enclosing contour'))
    # Includes verified neighbors, but never groups members that are ALL individually verified.
    proposals=overlap_merge_groups(kept,checks)
    proposals.sort(key=lambda ids:(-len(ids),ids))
    used=set();selected=[];skipped=[]
    for local_ids in proposals:
        original_ids=[kept[i]['original_candidate_id'] for i in local_ids]
        if used.intersection(local_ids):
            skipped.append(original_ids)
            continue
        text_ids=sorted({bid for i in local_ids for bid in individual[i]['text_ids']})
        if not text_ids:continue
        used.update(local_ids)
        selected.append(dict(group_id=f'M{len(selected)+1}',candidate_ids=original_ids,
            text_ids=text_ids,unresolved_candidate_ids=[kept[i]['original_candidate_id'] for i in local_ids
                                                       if not checks[i]['enclosed_boundary_found']]))
    return dict(individual=individual,merge_candidates=selected,
        overlapping_proposals_skipped=skipped,
        ignored_candidate_ids=[i for i in range(len(candidates)) if i not in {c['original_candidate_id'] for c in kept}])


def concatenate_records(records,page_groups):
    """Join selected candidate texts in original Magi order, leaving predictions intact."""
    surviving=[r for r in records if not r['delete_ocr']]
    rows=[];assignments=[]
    for page_index,analysis in enumerate(page_groups):
        page=sorted([r for r in surviving if r['page_index']==page_index],key=lambda r:r['text_index'])
        available={r['box_id'] for r in page}
        memberships={bid:[] for bid in available}
        for group in analysis['merge_candidates']:
            for bid in group['text_ids']:
                if bid in available:memberships[bid].append(group['group_id'])
        buckets={}
        for record in page:
            alternatives=memberships[record['box_id']]
            gid=alternatives[0] if alternatives else 'S_'+record['box_id']
            buckets.setdefault(gid,[]).append(record)
            assignments.append(dict(box_id=record['box_id'],page_index=page_index,
                group_id=f'PG{page_index+1}:{gid}',alternatives=alternatives[1:],
                assignment_ambiguous=len(alternatives)>1))
        for gid,members in buckets.items():
            speakers=list(dict.fromkeys(r['speaker'] for r in members))
            roles=list(dict.fromkeys(r['text_role'] for r in members))
            proposal=next((g for g in analysis['merge_candidates'] if g['group_id']==gid),None)
            rows.append(dict(page_index=page_index,group_id=f'PG{page_index+1}:{gid}',
                box_ids=[r['box_id'] for r in members],
                text=' '.join(r['text'].strip() for r in members if r['text'].strip()),
                speaker=speakers[0] if len(speakers)==1 else 'MULTIPLE',
                text_role=roles[0] if len(roles)==1 else 'MIXED',
                member_predictions=[dict(box_id=r['box_id'],speaker=r['speaker'],text_role=r['text_role']) for r in members],
                candidate_ids=proposal['candidate_ids'] if proposal else [],
                grouping_source='overlap_candidate' if proposal else 'standalone',
                reading_position=members[0]['text_index'],
                needs_review=any(r.get('needs_review',False) or len(memberships[r['box_id']])>1 for r in members)
                             or len(speakers)>1 or len(roles)>1))
    rows.sort(key=lambda r:(r['page_index'],r['reading_position']))
    expected={r['box_id'] for r in surviving}
    emitted=[bid for row in rows for bid in row['box_ids']]
    assert len(emitted)==len(set(emitted)) and set(emitted)==expected,'Missing/duplicated surviving text'
    return rows,assignments
