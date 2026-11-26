"""안전모 미착용 결과를 같은 기준으로 세기 위한 함수."""

def box_iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - w*h
    return w*h / union if union else 0.0


def evaluate_image(gt, pred, conf=0.25, iou_limit=0.5):
    # gt: cls, box / pred: cls, box, conf
    # 예측 하나와 정답 하나만 연결한다. 신뢰도가 높은 예측부터 확인한다.
    predictions = [dict(p, pred_id=i) for i, p in enumerate(pred) if p['conf'] >= conf]
    predictions.sort(key=lambda p: (-p['conf'], p['pred_id']))
    used = set()
    pred_rows = []
    for p in predictions:
        same = [(j, box_iou(p['box'], g['box'])) for j, g in enumerate(gt)
                if g['cls'] == p['cls']]
        available = [(j, v) for j, v in same if j not in used]
        best = max(available, key=lambda x: x[1], default=(None, 0.0))
        hit = best[0] is not None and best[1] >= iou_limit
        if hit:
            used.add(best[0])
        nearest = max(same, key=lambda x: x[1], default=(None, 0.0))
        other = [(j, box_iou(p['box'], g['box'])) for j, g in enumerate(gt)
                 if g['cls'] != p['cls']]
        other_best = max(other, key=lambda x: x[1], default=(None, 0.0))
        # IoU만으로 라벨 누락이나 실제 오탐의 원인을 확정하지 않는다.
        if hit:
            kind = 'TP'
        elif nearest[1] >= iou_limit and nearest[0] in used:
            kind = 'FP_duplicate_candidate'
        elif nearest[1] > 0:
            kind = 'FP_overlap_below_threshold'
        elif other_best[1] >= iou_limit:
            kind = 'FP_other_class_overlap'
        else:
            kind = 'FP_no_same_class_overlap'
        pred_rows.append(dict(p, status='TP' if hit else 'FP', kind=kind,
                              gt_id=best[0] if hit else None,
                              max_iou=nearest[1], nearest_gt_id=nearest[0],
                              other_gt_id=other_best[0], other_iou=other_best[1],
                              other_cls=gt[other_best[0]]['cls'] if other_best[0] is not None else None))
    gt_rows = []
    for j, g in enumerate(gt):
        match = next((p for p in pred_rows if p['gt_id'] == j), None)
        different = [(p, box_iou(g['box'], p['box'])) for p in predictions
                     if p['cls'] != g['cls']]
        other, overlap = max(different, key=lambda x: x[1], default=(None, 0.0))
        kind = 'TP' if match else ('FN_other_class_overlap' if overlap >= iou_limit
                                  else 'FN_no_valid_match')
        gt_rows.append(dict(g, gt_id=j, status='TP' if match else 'FN', kind=kind,
                            other_cls=other['cls'] if other and overlap >= iou_limit else None))
    return gt_rows, pred_rows
