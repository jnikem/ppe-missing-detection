"""기존 가중치로 검증한다. 학습이나 입력 크기 비교는 하지 않는다."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from ppe_eval import evaluate_image

BASE = Path(__file__).resolve().parents[1]

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_labels(path, width, height):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            rows.append(dict(cls=int(cls), box=[(cx-bw/2)*width,(cy-bh/2)*height,
                                               (cx+bw/2)*width,(cy+bh/2)*height]))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(BASE/'data/construction-ppe'))
    parser.add_argument('--weights', default=str(BASE/'inputs/best.pt'))
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--out', default=str(BASE/'results/week_2026-11-24'))
    args = parser.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    names = yaml.safe_load((root/'construction-ppe.yaml').read_text())['names'] if (root/'construction-ppe.yaml').exists() else None
    if names is None:
        config = next(root.glob('*.yaml'))
        names = yaml.safe_load(config.read_text())['names']
    if isinstance(names,list):
        names = dict(enumerate(names))
    images_by_split = {}
    counts, hashes, errors, bad_labels = [], {}, [], []
    for split in ['train','val']:
        images = sorted(p for p in (root/'images'/split).iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png'])
        images_by_split[split] = images
        boxes, image_counts = Counter(), Counter()
        label_paths = list((root/'labels'/split).glob('*.txt'))
        for ip in images:
            lp = root/'labels'/split/(ip.stem+'.txt')
            if not lp.exists():
                errors.append(str(lp))
            else:
                for line_no,line in enumerate(lp.read_text().splitlines(),1):
                    fields=line.split()
                    if len(fields)!=5:
                        bad_labels.append(dict(file=lp.name,line=line_no,reason='field_count'))
                        continue
                    c,cx,cy,bw,bh=map(float,fields)
                    if c!=int(c) or int(c) not in names or not (0<=cx<=1 and 0<=cy<=1 and 0<bw<=1 and 0<bh<=1):
                        bad_labels.append(dict(file=lp.name,line=line_no,reason='class_or_coordinate'))
            with Image.open(ip) as im:
                gs = read_labels(lp,*im.size)
            boxes.update(g['cls'] for g in gs)
            image_counts.update(set(g['cls'] for g in gs))
            h = hashlib.sha256(ip.read_bytes()).hexdigest()
            hashes.setdefault(h,[]).append((split,ip.name))
        for cls,name in names.items():
            counts.append(dict(split=split,cls=cls,name=name,images=image_counts[cls],boxes=boxes[cls]))
        print(split,len(images),'images',len(label_paths),'labels',flush=True)
    duplicates = [v for v in hashes.values() if len({x[0] for x in v})>1]
    write_csv(out/'class_counts.csv',counts)
    audit = dict(images={s:len(v) for s,v in images_by_split.items()},missing_labels=errors,
                 bad_label_rows=bad_labels,
                 exact_cross_split_duplicates=duplicates, note='파일 내용이 같은 중복만 확인. 비슷한 장면은 별도 검토 필요.')
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.audit_only:
        return
    from ultralytics import YOLO
    import ultralytics, torch
    model = YOLO(args.weights)
    if model.names != names:
        raise ValueError('데이터와 모델의 클래스 번호가 다릅니다.')
    images = images_by_split['val']
    gt_rows, pred_rows, raw = [], [], []
    for i,r in enumerate(model.predict([str(p) for p in images],imgsz=640,conf=0.25,
                                       device='cpu',stream=True,verbose=False)):
        ip = Path(r.path)
        with Image.open(ip) as im:
            width,height = im.size
        gt = read_labels(root/'labels/val'/(ip.stem+'.txt'),width,height)
        pred = [dict(cls=int(c),conf=float(s),box=b) for c,s,b in zip(
            r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.xyxy.cpu().tolist())]
        gs, ps = evaluate_image(gt,pred)
        raw.append(dict(img=ip.name,width=width,height=height,gt=gt,pred=pred))
        gt_rows.extend(dict(img=ip.name,**g) for g in gs)
        pred_rows.extend(dict(img=ip.name,**p) for p in ps)
        if (i+1)%20 == 0:
            print('predict',i+1,'/',len(images),flush=True)
    (out/'predictions.json').write_text(json.dumps(raw),encoding='utf-8')
    write_csv(out/'gt_review.csv',gt_rows)
    write_csv(out/'pred_review.csv',pred_rows)
    summary=[]
    for cls,name in names.items():
        gs=[g for g in gt_rows if g['cls']==cls]
        ps=[p for p in pred_rows if p['cls']==cls]
        tp=sum(p['status']=='TP' for p in ps)
        fp=len(ps)-tp
        fn=len(gs)-tp
        summary.append(dict(cls=cls,name=name,GT=len(gs),TP=tp,FP=fp,FN=fn,
                            precision=tp/(tp+fp) if tp+fp else None,
                            recall=tp/len(gs) if gs else None))
    write_csv(out/'fixed_threshold_metrics.csv',summary)
    (out/'run_settings.json').write_text(json.dumps(dict(
        ultralytics=ultralytics.__version__,torch=torch.__version__,device='cpu',imgsz=640,
        conf=0.25,match_iou=0.5,matching='confidence descending, best unused same-class GT',
        weights_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
