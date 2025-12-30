import os
import glob
import re

# ============ 路径配置 ============
# YOLO OBB 的 label 目录（这次的新结果）
LABEL_DIR = "D:/codepig/pig_track/scene_final/labels"

# 输出的 MOT 预测文件（归一化坐标）
OUT_FILE = "D:/codepig/pig_track/scene_final/pred_norm.txt"

# GT，用来拿到最大帧号，避免多出来的尾帧影响指标
GT_FILE = "D:/codepig/pig_track/scene_final/gt"
# ==================================


def get_max_gt_frame(gt_path: str) -> int:
    """读取 GT，拿到最大 frame id，用来截断多余帧"""
    if not os.path.isfile(gt_path):
        print(f"[WARN] 找不到 GT 文件：{gt_path}，将不过滤多余帧")
        return None

    max_frame = 0
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                frame_id = int(parts[0])
            except Exception:
                continue
            if frame_id > max_frame:
                max_frame = frame_id

    print(f"[INFO] GT 最大帧号 = {max_frame}")
    return max_frame or None


def natural_key(path: str) -> int:
    """
    对文件名里的数字做自然排序：
    例如：track_final_1.txt, track_final_2.txt, ..., track_final_10.txt
    """
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    txt_files = sorted(
        glob.glob(os.path.join(LABEL_DIR, "*.txt")),
        key=natural_key
    )
    print("[INFO] 发现 label txt 文件数量:", len(txt_files))

    if len(txt_files) == 0:
        print("[ERROR] 没有找到任何 txt，请检查 LABEL_DIR:", LABEL_DIR)
        return

    # 读 GT 拿最大帧号（2115）
    max_gt_frame = get_max_gt_frame(GT_FILE)

    fout = open(OUT_FILE, "w", encoding="utf-8")

    frame_id = 0  # 我们每个 txt 文件视为一帧，从 1 开始计数

    for txt_file in txt_files:
        frame_id += 1

        # 如果有 GT 最大帧限制，就不要写多出来的帧
        if max_gt_frame is not None and frame_id > max_gt_frame:
            break

        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # 期望 10 列：cls + 8个坐标 + track_id
                if len(parts) < 9:
                    print("格式异常（列数不足）:", txt_file, "->", line)
                    continue

                # 兼容两种写法：
                # A: cls x1 y1 x2 y2 x3 y3 x4 y4 id  (10列)
                # B: id  cls x1 y1 x2 y2 x3 y3 x4 y4 (10列)
                if len(parts) == 10:
                    # 判断第一个是不是类别（0~n），最后一个是不是整数 id
                    first = float(parts[0])
                    last = float(parts[-1])

                    # 这里我们假设「class 在 [0,10], id 在 [1,100]」这种范围
                    # 你的数据是 0-4 / 1-5，完全符合
                    if 0 <= first <= 10 and 1 <= last <= 100:
                        cls = int(round(first))
                        track_id = int(round(last))
                        coords = list(map(float, parts[1:-1]))  # 8 个数
                    else:
                        # 反过来：第一个是 id，第二个是 cls
                        track_id = int(round(first))
                        cls = int(round(float(parts[1])))
                        coords = list(map(float, parts[2:]))
                else:
                    print("暂不支持的列数格式:", len(parts), "->", line)
                    continue

                if len(coords) != 8:
                    print("坐标数量不是 8 个:", txt_file, "->", line)
                    continue

                xs = coords[0::2]  # x1,x2,x3,x4
                ys = coords[1::2]  # y1,y2,y3,y4

                xmin = max(0.0, min(xs))
                ymin = max(0.0, min(ys))
                xmax = max(xs)
                ymax = max(ys)

                # 有些点可能略超出 [0,1]，这里 clamp 一下
                xmax = min(1.0, xmax)
                ymax = min(1.0, ymax)

                w = xmax - xmin
                h = ymax - ymin

                if w <= 0 or h <= 0:
                    continue

                # 注意：这里直接输出「归一化 xywh」
                # MOT 格式：frame, id, x, y, w, h, score, class, -1
                fout.write(
                    f"{frame_id},{track_id},"
                    f"{xmin:.6f},{ymin:.6f},{w:.6f},{h:.6f},"
                    f"1,{cls},-1\n"
                )

    fout.close()
    print("[INFO] 转换完成，输出文件:", OUT_FILE)


if __name__ == "__main__":
    main()
