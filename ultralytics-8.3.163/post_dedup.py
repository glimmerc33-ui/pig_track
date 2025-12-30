# post_dedup_cls0_center.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

PRED_IN  = r"D:\codepig\pig_track\scene_final\pred_norm.txt"
PRED_OUT = r"D:\codepig\pig_track\scene_final\pred_norm_dedup.txt"

# ---------- 你要调的超参 ----------
KEEP_CLS = {0}          # 只保留猪类（按你统计结果，cls=0）
SCORE_MIN = 0.00        # 建议先 0.00；后面再试 0.05/0.1
IOU_THR = 0.75         # IoU>0.7 视为重复框（可试 0.6~0.8）
CENTER_THR = 0.06   # 原来 0.03
MAX_KEEP_PER_FRAME = 7  # 真实7只猪；只在该帧>7时截断到7（可设 None 关闭）
# ---------------------------------

cols = ["frame","id","x","y","w","h","score","cls","na"]
df = pd.read_csv(PRED_IN, header=None, names=cols)

df["frame"] = df["frame"].astype(int)
df["id"]    = df["id"].astype(int)
df["score"] = df["score"].astype(float)
df["cls"]   = df["cls"].astype(int)

# 1) 过滤类别 + 分数
df = df[df["cls"].isin(KEEP_CLS)].copy()
df = df[df["score"] >= SCORE_MIN].copy()

def xywh_to_xyxy(row):
    x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-12)

def center_dist(row_i, row_j):
    # 中心点欧氏距离（归一化坐标系）
    ci_x = float(row_i["x"]) + 0.5 * float(row_i["w"])
    ci_y = float(row_i["y"]) + 0.5 * float(row_i["h"])
    cj_x = float(row_j["x"]) + 0.5 * float(row_j["w"])
    cj_y = float(row_j["y"]) + 0.5 * float(row_j["h"])
    return float(np.hypot(ci_x - cj_x, ci_y - cj_y))

out = []
for f, g in df.groupby("frame", sort=True):
    g = g.sort_values("score", ascending=False).reset_index(drop=True).copy()

    boxes = [xywh_to_xyxy(r) for _, r in g.iterrows()]

    keep_idx = []
    suppressed = set()

    for i in range(len(g)):
        if i in suppressed:
            continue
        keep_idx.append(i)

        for j in range(i + 1, len(g)):
            if j in suppressed:
                continue

            # 2) IoU 重复
            if iou(boxes[i], boxes[j]) > IOU_THR:
                suppressed.add(j)
                continue

            # 3) 中心点近似重复（你要加的规则）
            if CENTER_THR is not None and CENTER_THR > 0:
                if (
                     center_dist(g.loc[i], g.loc[j]) < CENTER_THR
                     and iou(boxes[i], boxes[j]) > 0.30
                     ):
                    suppressed.add(j)
                    continue

    kept = g.iloc[keep_idx].copy()

    # 4) 可选：每帧最多保留 7 个（只在>7时裁剪）
    if MAX_KEEP_PER_FRAME is not None and len(kept) > MAX_KEEP_PER_FRAME:
        kept = kept.sort_values("score", ascending=False).head(MAX_KEEP_PER_FRAME)

    out.append(kept)

out_df = pd.concat(out, ignore_index=True)
out_df = out_df.sort_values(["frame","id"])
out_df.to_csv(PRED_OUT, header=False, index=False, float_format="%.6f")

print("Wrote:", PRED_OUT, "rows=", len(out_df))
