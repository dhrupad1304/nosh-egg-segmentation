import argparse, json, csv
from pathlib import Path
import cv2, numpy as np
import matplotlib.pyplot as plt

def boundary(mask):
    mask = (mask > 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8)) > 0

def boundary_fscore(gt, pred, tolerance=2):
    gb, pb = boundary(gt), boundary(pred)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1,2*tolerance+1))
    gd = cv2.dilate(gb.astype(np.uint8), kernel) > 0
    pd = cv2.dilate(pb.astype(np.uint8), kernel) > 0
    precision = (pb & gd).sum() / max(pb.sum(), 1)
    recall = (gb & pd).sum() / max(gb.sum(), 1)
    return 2*precision*recall/max(precision+recall, 1e-12)

def metrics(gt, pred):
    gt, pred = gt > 0, pred > 0
    tp = np.logical_and(gt,pred).sum()
    fp = np.logical_and(~gt,pred).sum()
    fn = np.logical_and(gt,~pred).sum()
    union = tp+fp+fn
    iou = tp/union if union else 1.0
    dice = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.0
    precision = tp/(tp+fp) if tp+fp else 1.0
    recall = tp/(tp+fn) if tp+fn else 1.0
    return iou, dice, precision, recall, boundary_fscore(gt,pred)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images",required=True); ap.add_argument("--gt",required=True)
    ap.add_argument("--pred",required=True); ap.add_argument("--split",required=True)
    ap.add_argument("--output",default="reports/evaluation"); ap.add_argument("--worst-k",type=int,default=8)
    args=ap.parse_args()
    imgdir,gtdir,predir=map(Path,[args.images,args.gt,args.pred]); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True); gallery=out/"worst_gallery"; gallery.mkdir(exist_ok=True)
    split=json.loads(Path(args.split).read_text()); rows=[]
    for name in split.get("eval",[]):
        stem=Path(name).stem
        ip=imgdir/name; gp=gtdir/f"{stem}.png"; pp=predir/f"{stem}.png"
        if not (ip.exists() and gp.exists() and pp.exists()): continue
        gt=cv2.imread(str(gp),0); pr=cv2.imread(str(pp),0); image=cv2.imread(str(ip))
        if gt is None or pr is None: continue
        vals=metrics(gt,pr); rows.append([name,*vals])
    if not rows: raise SystemExit("No matching eval image, GT mask, and prediction files.")
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["image","IoU","Dice","Precision","Recall","Boundary_F"]); w.writerows(rows)
    arr=np.array([r[1:] for r in rows],float)
    print("Eval means:", dict(zip(["IoU","Dice","Precision","Recall","Boundary_F"],arr.mean(0).round(4))))
    plt.figure(); plt.hist(arr[:,0],bins=10); plt.xlabel("IoU"); plt.ylabel("Image count")
    plt.title("Evaluation IoU distribution"); plt.tight_layout(); plt.savefig(out/"iou_histogram.png"); plt.close()

    for row in sorted(rows,key=lambda r:r[1])[:args.worst_k]:
        name=row[0]; stem=Path(name).stem
        image=cv2.imread(str(imgdir/name)); gt=cv2.imread(str(gtdir/f"{stem}.png"),0)>0
        pr=cv2.imread(str(predir/f"{stem}.png"),0)>0
        overlay=image.copy(); overlay[gt & ~pr]=(0,0,255); overlay[pr & ~gt]=(255,0,0)
        cv2.imwrite(str(gallery/f"{stem}_error.png"),overlay)
    print("Wrote metrics and qualitative outputs to",out)

if __name__=="__main__": main()
