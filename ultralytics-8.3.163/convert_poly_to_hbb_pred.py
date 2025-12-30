import os
import glob

LABEL_DIR = "D:/codepig/pig_track/ultralytics-8.3.163/track_final_model_labels"
OUT_FILE  = "D:/codepig/pig_track/no_obb/pred_norm_hbb.txt"

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def main():
    files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No txt found in {LABEL_DIR}")

    with open(OUT_FILE, "w", encoding="utf-8") as fw:
        for fp in files:
            base = os.path.basename(fp)
            # 兼容 000001.txt / 1.txt / frame_000001.txt 等，尽量提取数字
            digits = "".join([c for c in os.path.splitext(base)[0] if c.isdigit()])
            if digits == "":
                continue
            frame = int(digits)

            with open(fp, "r", encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()

                    # 你给的格式：cls x1 y1 x2 y2 x3 y3 x4 y4 id
                    if len(parts) < 10:
                        continue

                    cls = int(float(parts[0]))
                    xs = [safe_float(parts[1]), safe_float(parts[3]), safe_float(parts[5]), safe_float(parts[7])]
                    ys = [safe_float(parts[2]), safe_float(parts[4]), safe_float(parts[6]), safe_float(parts[8])]
                    tid = int(float(parts[9]))

                    if any(v is None for v in xs+ys):
                        continue

                    x_min = min(xs); x_max = max(xs)
                    y_min = min(ys); y_max = max(ys)

                    w = x_max - x_min
                    h = y_max - y_min

                    # 可选：裁剪到 [0,1]，避免你示例里 y>1 或 y<0 影响 IoU
                    x_min = max(0.0, min(1.0, x_min))
                    y_min = max(0.0, min(1.0, y_min))
                    x_max = max(0.0, min(1.0, x_max))
                    y_max = max(0.0, min(1.0, y_max))
                    w = max(0.0, x_max - x_min)
                    h = max(0.0, y_max - y_min)

                    score = 1.0
                    ignore = 0

                    fw.write(f"{frame},{tid},{x_min:.6f},{y_min:.6f},{w:.6f},{h:.6f},{score:.3f},{cls},{ignore}\n")

    print("Wrote:", OUT_FILE)

if __name__ == "__main__":
    main()
