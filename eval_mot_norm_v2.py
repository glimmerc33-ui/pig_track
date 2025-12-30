import argparse
import pandas as pd
import numpy as np
import motmetrics as mm
import hashlib
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def file_md5(path, nbytes=2_000_000):
    """快速 MD5（读前nbytes即可判断是否明显不同）"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()

def read_table_anysep(path, ncols, names):
    """
    兼容：
    - 逗号分隔: 1,1,0.2,0.3,...
    - 空格分隔: 1 1 0.2 0.3 ...
    """
    df = pd.read_csv(
        path,
        header=None,
        names=names,
        sep=r"[,\s]+",
        engine="python",
    )
    # 有些文件行尾可能多分隔符导致空列，截断
    if df.shape[1] > ncols:
        df = df.iloc[:, :ncols]
        df.columns = names[:ncols]
    return df

def load_gt(gt_path, img_w, img_h):
    # MOT GT: frame,id,x,y,w,h,valid,cls,vis,ignore (像素)
    names = ["frame","id","x","y","w","h","valid","cls","vis","ignore"]
    df = read_table_anysep(gt_path, 10, names)
    df = df[["frame","id","x","y","w","h"]].copy()

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    # 像素 -> 归一化
    df["x"] = df["x"].astype(float) / img_w
    df["y"] = df["y"].astype(float) / img_h
    df["w"] = df["w"].astype(float) / img_w
    df["h"] = df["h"].astype(float) / img_h
    return df

def load_pred(pred_path):
    # Pred: frame,id,x,y,w,h,score,cls,ignore （你现在的格式）
    names = ["frame","id","x","y","w","h","score","cls","ignore"]
    df = read_table_anysep(pred_path, 9, names)

    # 有的 pred 可能缺 score/cls/ignore，做兼容
    for col in ["score","cls","ignore"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["frame","id","x","y","w","h","score"]].copy()
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    for c in ["x","y","w","h"]:
        df[c] = df[c].astype(float)
    # score 可能为空
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df

def xywh_to_xyxy(df):
    x1 = df["x"].to_numpy(dtype=float)
    y1 = df["y"].to_numpy(dtype=float)
    x2 = x1 + df["w"].to_numpy(dtype=float)
    y2 = y1 + df["h"].to_numpy(dtype=float)
    return np.stack([x1,y1,x2,y2], axis=1)

def iou_xyxy(a, b):
    """
    a: [N,4], b:[M,4] in 0~1
    returns IoU [N,M]
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, ax2-ax1) * np.maximum(0.0, ay2-ay1)
    area_b = np.maximum(0.0, bx2-bx1) * np.maximum(0.0, by2-by1)
    union = area_a + area_b - inter + 1e-12
    return (inter / union)

def dedup_best_score(df, key=("frame","id")):
    """
    同一帧同一ID多条：优先保留 score 最大；
    若 score 全 NaN，则保留最后一条。
    """
    if df.empty:
        return df
    if df["score"].notna().any():
        df2 = df.sort_values(list(key) + ["score"], ascending=[True, True, False])
        return df2.drop_duplicates(subset=list(key), keep="first")
    else:
        return df.drop_duplicates(subset=list(key), keep="last")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--img_w", type=float, default=1280.0)
    ap.add_argument("--img_h", type=float, default=736.0)
    ap.add_argument("--iou_thr", type=float, default=0.5, help="匹配IoU阈值（建议也试0.7/0.8）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sanity_compare", default=None,
                    help="可选：再给一个pred文件路径，与--pred做差异检查（不影响评测）")
    args = ap.parse_args()

    print("加载 GT 和预测文件（归一化坐标）...")
    print("GT  :", args.gt)
    print("Pred:", args.pred)
    print("GT  md5:", file_md5(args.gt))
    print("Pred md5:", file_md5(args.pred))

    gt = load_gt(args.gt, args.img_w, args.img_h)
    pr = load_pred(args.pred)

    # 去重
    gt = gt.drop_duplicates(subset=["frame","id"], keep="last")
    pr = dedup_best_score(pr)

    if args.verbose:
        print(f"[GT]   rows={len(gt)}, frames={gt['frame'].nunique()}, ids={gt['id'].nunique()}")
        print(f"[Pred] rows={len(pr)}, frames={pr['frame'].nunique()}, ids={pr['id'].nunique()}")

    # 可选：检查两个pred是否真的不同
    if args.sanity_compare:
        other = load_pred(args.sanity_compare)
        other = dedup_best_score(other)
        # 只比前6列（frame,id,x,y,w,h）
        a = pr[["frame","id","x","y","w","h"]].to_numpy()
        b = other[["frame","id","x","y","w","h"]].to_numpy()
        same_shape = a.shape == b.shape
        allclose = same_shape and np.allclose(a, b, atol=1e-9)
        print("\n[Sanity] compare pred vs sanity_compare:")
        print("  sanity_compare:", args.sanity_compare)
        print("  same_shape:", same_shape, "allclose:", allclose)
        if not same_shape:
            print("  pred shape:", a.shape, "other shape:", b.shape)

    # motmetrics accumulator
    acc = mm.MOTAccumulator(auto_id=False)

    all_frames = sorted(set(gt["frame"].unique()) | set(pr["frame"].unique()))

    # 额外统计：平均匹配 IoU
    matched_ious = []

    for frame in all_frames:
        g = gt[gt["frame"] == frame]
        p = pr[pr["frame"] == frame]

        gt_ids = [int(x) for x in g["id"].tolist()]
        pr_ids = [int(x) for x in p["id"].tolist()]

        if len(gt_ids) == 0 and len(pr_ids) == 0:
            continue

        g_boxes = xywh_to_xyxy(g) if len(gt_ids) else np.empty((0,4), float)
        p_boxes = xywh_to_xyxy(p) if len(pr_ids) else np.empty((0,4), float)

        # 我们自己算 IoU，并把 iou < thr 的置为 NaN（不可匹配）
        ious = iou_xyxy(g_boxes, p_boxes)  # [Ng, Np]
        dists = np.full_like(ious, np.nan, dtype=float)
        ok = ious >= args.iou_thr
        dists[ok] = 1.0 - ious[ok]

        # update（让 motmetrics 自己做 Hungarian）
        acc.update(gt_ids, pr_ids, dists, frameid=int(frame))

        # 把这一帧真正被匹配上的 iou 记录下来（需要读 acc 的 events）
        # 简化：这里不从 events 里复原匹配对，直接统计“可匹配 iou 的均值”作为参考
        if ok.any():
            matched_ious.append(float(np.mean(ious[ok])))

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_misses",
            "num_false_positives",
            "mostly_tracked",
            "mostly_lost",
        ],
        name="summary",
    )

    print("\n===== MOT Evaluation Summary =====")
    print(summary)

    if len(matched_ious):
        print(f"\n[Extra] mean IoU over (IoU>=thr) pairs (rough): {np.mean(matched_ious):.4f}  (thr={args.iou_thr})")
    else:
        print(f"\n[Extra] No pairs with IoU>=thr={args.iou_thr} (check coordinate scale / paths)")

if __name__ == "__main__":
    main()
