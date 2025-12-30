
import os
import glob
import re

# YOLO OBB tracking 的标签目录
LABEL_DIR = "D:/codepig/pig_track/track_detect"

# 输出的 MOT 预测文件（归一化坐标）
OUT_FILE = "D:/codepig/pig_track/no_obb/pred_norm.txt"


def natural_key(path):
    """按 track_final_1, track_final_2, ... 自然顺序排序"""
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def main():
    txt_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")),
                       key=natural_key)
    print("发现 txt 文件数量:", len(txt_files))
    if len(txt_files) == 0:
        print("!!! 没有找到任何 txt，请检查 LABEL_DIR")
        return

    fout = open(OUT_FILE, "w")
    frame_id = 1  # 与 gt.txt 一样，从 1 开始

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # 期望格式: cls x1 y1 x2 y2 x3 y3 x4 y4 track_id
                if len(parts) != 10:
                    print("格式异常:", txt_file, "->", line)
                    continue

                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:-1]))  # 8 个数
                track_id = int(float(parts[-1]))

                xs = coords[0::2]
                ys = coords[1::2]

                # 有些点略超出 0~1，做一下裁剪
                xs = [max(0.0, min(1.0, x)) for x in xs]
                ys = [max(0.0, min(1.0, y)) for y in ys]

                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)

                w = xmax - xmin
                h = ymax - ymin
                if w <= 1e-6 or h <= 1e-6:
                    continue

                # 这里 x,y,w,h 仍然是 0~1 归一化坐标
                fout.write(
                    f"{frame_id},{track_id},"
                    f"{xmin:.6f},{ymin:.6f},{w:.6f},{h:.6f},1,{cls},-1\n"
                )

        frame_id += 1

    fout.close()
    print("转换完成，输出文件:", OUT_FILE)


if __name__ == "__main__":
    main()
