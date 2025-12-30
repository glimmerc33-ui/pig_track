import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
import numpy as np
import motmetrics as mm

# ============ 路径 & 分辨率 ============
GT_FILE   = "D:/codepig/pig_track/track_final/gt/gt.txt"
PRED_FILE = r"D:/codepig/pig_track/no_obb/pred_norm.txt"   # ✅ 用 raw string

IMG_W = 1280.0
IMG_H = 736.0
# ======================================


def load_gt_norm(path: str) -> pd.DataFrame:
    """加载 GT（像素），并归一化到 0~1，GT通常是 tlwh(左上角)"""
    cols = ["frame", "id", "x", "y", "w", "h", "valid", "cls", "vis", "ignore"]
    df = pd.read_csv(path, header=None, names=cols)
    df = df[["frame", "id", "x", "y", "w", "h"]].copy()
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)

    df["x"] = df["x"] / IMG_W
    df["y"] = df["y"] / IMG_H
    df["w"] = df["w"] / IMG_W
    df["h"] = df["h"] / IMG_H
    return df


def load_pred_norm(path: str) -> pd.DataFrame:
    """加载预测：兼容 6列/9列。若有score则用于去重保留最大score"""
    # 你的格式一般是: frame,id,x,y,w,h,score,cls,na
    cols = ["frame", "id", "x", "y", "w", "h", "score", "cls", "na"]
    df = pd.read_csv(path, header=None, names=cols)

    # 兼容有些文件没有score的情况
    if df["score"].isna().all():
        df = df[["frame", "id", "x", "y", "w", "h"]].copy()
        df["score"] = 1.0
    else:
        df = df[["frame", "id", "x", "y", "w", "h", "score"]].copy()

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def dedup_by_frame_id_keep_best_score(df: pd.DataFrame) -> pd.DataFrame:
    """同一帧同一id只能出现一次：保留score最高"""
    if df.empty:
        return df
    df = df.sort_values(["frame", "id", "score"], ascending=[True, True, False])
    return df.drop_duplicates(subset=["frame", "id"], keep="first")


def xywh_to_xyxy_tl(df: pd.DataFrame) -> np.ndarray:
    """假设 x,y 是左上角 tlwh -> xyxy"""
    x1 = df["x"].to_numpy(dtype=float)
    y1 = df["y"].to_numpy(dtype=float)
    x2 = x1 + df["w"].to_numpy(dtype=float)
    y2 = y1 + df["h"].to_numpy(dtype=float)
    return np.stack([x1, y1, x2, y2], axis=1)


def xywh_to_xyxy_center(df: pd.DataFrame) -> np.ndarray:
    """假设 x,y 是中心点 cxcywh -> xyxy"""
    cx = df["x"].to_numpy(dtype=float)
    cy = df["y"].to_numpy(dtype=float)
    w  = df["w"].to_numpy(dtype=float)
    h  = df["h"].to_numpy(dtype=float)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def pick_pred_box_mode(gt: pd.DataFrame, pred: pd.DataFrame, sample_frames=100) -> str:
    """
    自动判断 pred 的 x,y 是 tl 还是 center：
    在若干帧上计算“平均可匹配IoU”，谁大用谁。
    """
    frames = sorted(set(gt["frame"].unique()) & set(pred["frame"].unique()))
    if not frames:
        return "tl"
    frames = frames[:min(sample_frames, len(frames))]

    def avg_best_iou(pred_xyxy_fn):
        best_ious = []
        for f in frames:
            gt_f = gt[gt["frame"] == f]
            pr_f = pred[pred["frame"] == f]
            if len(gt_f) == 0 or len(pr_f) == 0:
                continue
            gt_boxes = xywh_to_xyxy_tl(gt_f)   # GT通常就是tlwh
            pr_boxes = pred_xyxy_fn(pr_f)
            # iou_matrix 输出的是距离=1-iou（不可匹配为nan），这里反推一下iou
            dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=1.0)
            # dist: nan表示iou>max_iou?（这里max_iou=1.0基本不会），用1-dist近似iou
            iou = 1.0 - dist
            iou = np.where(np.isnan(iou), 0.0, iou)
            # 每个GT取一个最大iou
            best_ious.extend(iou.max(axis=1).tolist())
        return float(np.mean(best_ious)) if best_ious else 0.0

    s_tl = avg_best_iou(xywh_to_xyxy_tl)
    s_c  = avg_best_iou(xywh_to_xyxy_center)

    mode = "center" if s_c > s_tl else "tl"
    print(f"[AutoMode] avg_best_iou tl={s_tl:.4f}, center={s_c:.4f} -> use {mode}")
    return mode


def main():
    print("加载 GT 和预测文件（归一化坐标）...")

    gt = load_gt_norm(GT_FILE)
    pred = load_pred_norm(PRED_FILE)

    gt = dedup_by_frame_id_keep_best_score(gt.assign(score=1.0))  # GT没score也照样去重
    pred = dedup_by_frame_id_keep_best_score(pred)

    mode = pick_pred_box_mode(gt, pred, sample_frames=120)

    pred_xyxy_fn = xywh_to_xyxy_center if mode == "center" else xywh_to_xyxy_tl

    acc = mm.MOTAccumulator(auto_id=False)

    all_frames = sorted(set(gt["frame"].unique()) | set(pred["frame"].unique()))
    for frame in all_frames:
        frame = int(frame)
        gt_f = gt[gt["frame"] == frame]
        pr_f = pred[pred["frame"] == frame]

        gt_ids = [int(x) for x in gt_f["id"].tolist()]
        pr_ids = [int(x) for x in pr_f["id"].tolist()]

        if len(gt_ids) == 0 and len(pr_ids) == 0:
            continue

        gt_boxes = xywh_to_xyxy_tl(gt_f) if len(gt_ids) > 0 else np.empty((0, 4), dtype=float)
        pr_boxes = pred_xyxy_fn(pr_f)    if len(pr_ids) > 0 else np.empty((0, 4), dtype=float)

        dists = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(gt_ids, pr_ids, dists, frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames","mota","motp","idf1","idp","idr",
            "num_switches","num_misses","num_false_positives",
            "mostly_tracked","mostly_lost",
        ],
        name="summary",
    )
    print(summary)


if __name__ == "__main__":
    main()
