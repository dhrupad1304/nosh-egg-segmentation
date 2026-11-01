import argparse
from pathlib import Path
import cv2, numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

def detect(frame):
    """Simple color/threshold segmentation starter; tune for the supplied video."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # Broad low-saturation light-object cue; works only as an initial heuristic.
    mask=cv2.inRange(hsv,np.array([0,0,100]),np.array([180,120,255]))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
    detections=[]
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA]
        if area < 250: continue
        detections.append({"mask":labels==i,"centroid":centroids[i],"area":area})
    return detections

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--output",default="outputs/tracked_eggs.mp4")
    ap.add_argument("--max-distance",type=float,default=90); ap.add_argument("--max-missed",int,default=12)
    args=ap.parse_args()
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Could not open video: {args.video}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outp=Path(args.output); outp.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(outp),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    tracks={}; next_id=1; frame_no=0

    while True:
        ok,frame=cap.read()
        if not ok: break
        detections=detect(frame)
        tids=list(tracks.keys())
        matched_t=set(); matched_d=set()
        if tids and detections:
            cost=np.zeros((len(tids),len(detections)),float)
            for i,tid in enumerate(tids):
                for j,d in enumerate(detections):
                    cost[i,j]=np.linalg.norm(tracks[tid]["centroid"]-d["centroid"])
            if linear_sum_assignment:
                rr,cc=linear_sum_assignment(cost)
            else:
                pairs=sorted((cost[i,j],i,j) for i in range(len(tids)) for j in range(len(detections)))
                rr=[]; cc=[]
                for _,i,j in pairs:
                    if i not in rr and j not in cc: rr.append(i); cc.append(j)
            for i,j in zip(rr,cc):
                if cost[i,j] <= args.max_distance:
                    tid=tids[i]; d=detections[j]
                    tracks[tid].update(centroid=d["centroid"],mask=d["mask"],missed=0)
                    matched_t.add(tid); matched_d.add(j)

        for j,d in enumerate(detections):
            if j not in matched_d:
                tracks[next_id]={"centroid":d["centroid"],"mask":d["mask"],"missed":0}
                next_id+=1
        for tid in list(tracks):
            if tid not in matched_t and tid not in [next_id-1]:
                tracks[tid]["missed"]+=1
            if tracks[tid]["missed"]>args.max_missed: del tracks[tid]

        vis=frame.copy()
        for tid,t in tracks.items():
            if t["missed"]>0: continue
            color=((tid*73)%255,(tid*137)%255,(tid*199)%255)
            vis[t["mask"]]=((vis[t["mask"]].astype(np.float32)*.55)+(np.array(color)*.45)).astype(np.uint8)
            cx,cy=np.round(t["centroid"]).astype(int)
            cv2.putText(vis,f"Egg {tid}",(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
        writer.write(vis); frame_no+=1
        if frame_no%100==0: print("Frames processed:",frame_no)

    cap.release(); writer.release()
    print("Saved tracked video:",outp)

if __name__=="__main__": main()
