import os
import re
import math
import xml.etree.ElementTree as ET

# ======== 按你的路径修改这里 ========
ANN_DIR = r"D:/codepig/pig_track/track_final/labels"        # XML目录
OUT_GT  = r"D:/codepig/pig_track/track_final/gt/gt.txt"    # 输出
# =====================================

os.makedirs(os.path.dirname(OUT_GT), exist_ok=True)


def natural_key(name: str):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def clsname_to_id(name: str) -> int:
    name = name.strip().lower()
    if name.startswith("pig"):
        num = "".join([c for c in name if c.isdigit()])
        return int(num)
    raise ValueError(f"无法识别类别名: {name}")


def rotated_box_to_hbb(cx, cy, w, h, angle_rad):
    """robndbox(弧度) -> 水平框 xmin,ymin,w,h"""
    cosa = math.cos(angle_rad)
    sina = math.sin(angle_rad)

    bw = float(w)
    bh = float(h)
    cx = float(cx)
    cy = float(cy)

    pts = []
    for dx, dy in [(-bw/2, -bh/2), (bw/2, -bh/2), (bw/2, bh/2), (-bw/2, bh/2)]:
        x = cx + dx * cosa - dy * sina
        y = cy + dx * sina + dy * cosa
        pts.append((x, y))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, (xmax - xmin), (ymax - ymin)


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objs = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            continue
        tid = clsname_to_id(name_node.text)

        rob = obj.find("robndbox")
        if rob is None:
            continue

        cx = float(rob.findtext("cx"))
        cy = float(rob.findtext("cy"))
        w  = float(rob.findtext("w"))
        h  = float(rob.findtext("h"))
        angle = float(rob.findtext("angle"))  # ✅ XML里就是弧度，别 radians()

        x, y, bw, bh = rotated_box_to_hbb(cx, cy, w, h, angle)
        objs.append((tid, x, y, bw, bh))

    return objs


def main():
    xml_files = [f for f in os.listdir(ANN_DIR) if f.lower().endswith(".xml")]
    xml_files = sorted(xml_files, key=natural_key)

    print(f"发现 {len(xml_files)} 个 XML")

    rows = []
    for frame_id, xml_name in enumerate(xml_files, start=1):  # ✅ frame从1开始，和pred_norm对齐
        xml_path = os.path.join(ANN_DIR, xml_name)
        objs = parse_xml(xml_path)

        # ✅ 同一帧同一id只保留一次（理论上你的标注不会重复，但这里强保险）
        seen = set()
        for tid, x, y, w, h in objs:
            key = (frame_id, tid)
            if key in seen:
                continue
            seen.add(key)
            rows.append((frame_id, tid, x, y, w, h, 1, -1, -1, -1))

    # 排序
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(OUT_GT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]},{r[7]},{r[8]},{r[9]}\n")

    print("✅ 写入完成:", OUT_GT)

    # ✅ 额外：快速检查重复
    # 如果这里输出>0，说明你的XML里同一帧同一猪被标了多次（或文件重复）
    from collections import Counter
    c = Counter((r[0], r[1]) for r in rows)
    dup = sum(1 for k,v in c.items() if v > 1)
    print("重复(frame,id)数量:", dup)


if __name__ == "__main__":
    main()
