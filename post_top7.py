# post_top7.py
# -*- coding: utf-8 -*-
import pandas as pd

PRED_IN  = r"D:\codepig\pig_track\scene_final\pred_norm.txt"
PRED_OUT = r"D:\codepig\pig_track\scene_final\pred_norm_top7.txt"

TOP_N = 7
SCORE_MIN = 0.0   # 先不强杀，想更狠就改成 0.2 / 0.3 试试

cols = ["frame","id","x","y","w","h","score","cls","na"]
df = pd.read_csv(PRED_IN, header=None, names=cols)

# 确保类型正确
df["frame"] = df["frame"].astype(int)
df["id"]    = df["id"].astype(int)
df["score"] = df["score"].astype(float)

# 先做一个最低分过滤（可选）
df = df[df["score"] >= SCORE_MIN].copy()

out_rows = []
prev_ids = set()

for f, g in df.groupby("frame", sort=True):
    # 同一帧：按分数降序
    g = g.sort_values("score", ascending=False)

    # 1) 先拿“上一帧出现过的ID”（保持轨迹连续）
    g_keep_prev = g[g["id"].isin(prev_ids)]
    kept = g_keep_prev.head(TOP_N)

    # 2) 如果还没到 7 个，用本帧剩下的最高分补齐
    if len(kept) < TOP_N:
        need = TOP_N - len(kept)
        rest = g[~g.index.isin(kept.index)]
        kept = pd.concat([kept, rest.head(need)], ignore_index=False)

    # 3) 更新 prev_ids（注意：只用最终保留的那些ID）
    prev_ids = set(kept["id"].tolist())

    out_rows.append(kept)

out = pd.concat(out_rows, ignore_index=True)
out = out.sort_values(["frame","id"])

# 写回原格式（保持你 eval 脚本能读）
out.to_csv(PRED_OUT, header=False, index=False, float_format="%.6f")
print("Wrote:", PRED_OUT, "rows=", len(out))
