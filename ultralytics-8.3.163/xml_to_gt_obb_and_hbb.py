# -*- coding: utf-8 -*-
import os
import re
import math
import glob
import argparse
import xml.etree.ElementTree as ET

def rot_rect_to_poly(cx, cy, w, h, angle_rad):
    """
    由 robndbox (cx,cy,w,h,angle) 转成 4个角点 (x1,y1,...,x4,y4)
    angle: 弧度，按你XML的值处理
    返回顺序：逆时针/顺时针都可，只要一致。这里给“先左上再右上再右下再左下”（近似）
    """
    # 矩形局部坐标（以中心为原点）
    hw = w / 2.0
    hh = h / 2.0

    # 四个角（未旋转）
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]  # 左上,右上,右下,左下（在局部坐标系）

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    pts = []
    for x, y in corners:
        xr = x * cos_a - y * sin_a + cx
        yr = x * sin_a + y * cos_a + cy
        pts.append((xr, yr))
    return pts  # [(x1,y1)...(x4,y4)]

def poly_to_hbb(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, (x_max - x_min), (y_max - y_min)

def parse_frame_from_filename(fn):
    """
    从 001451.jpg -> 1451
    """
    base = os.path.splitext(os.path.basename(fn))[0]
    m = re.search(r'(\d+)$', base)
    if not m:
        raise ValueError(f"Cannot parse frame id from filename: {fn}")
    return int(m.group(1))

def parse_pig_id(name):
    """
    pig1 -> 1
    """
    m = re.search(r'(\d+)$', name)
    if not m:
        raise ValueError(f"Cannot parse pig id from object name: {name}")
    return int(m.group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True, help="XML标注文件夹（逐帧xml）")
    ap.add_argument("--out_gt_hbb", required=True, help="输出HBB GT (MOT格式) 路径")
    ap.add_argument("--out_gt_poly", required=True, help="输出OBB poly GT 路径（自定义poly格式）")
    ap.add_argument("--img_w", type=float, default=1280.0)
    ap.add_argument("--img_h", type=float, default=736.0)
    args = ap.parse_args()

    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    if not xml_files:
        raise FileNotFoundError(f"No xml found in {args.xml_dir}")

    # 写 GT：HBB（MOT格式：frame,id,x,y,w,h,valid,cls,vis,ignore）
    # 写 GT：poly（frame,id,x1,y1,...,x4,y4）
    with open(args.out_gt_hbb, "w", encoding="utf-8") as f_hbb, \
         open(args.out_gt_poly, "w", encoding="utf-8") as f_poly:

        for xp in xml_files:
            tree = ET.parse(xp)
            root = tree.getroot()

            filename = root.findtext("filename")
            if not filename:
                # 有的XML没有filename，退化用xml名
                filename = os.path.basename(xp).replace(".xml", ".jpg")

            frame = parse_frame_from_filename(filename)

            for obj in root.findall("object"):
                name = obj.findtext("name", default="pig")
                pid = parse_pig_id(name)

                robnd = obj.find("robndbox")
                if robnd is None:
                    continue

                cx = float(robnd.findtext("cx"))
                cy = float(robnd.findtext("cy"))
                w  = float(robnd.findtext("w"))
                h  = float(robnd.findtext("h"))
                ang = float(robnd.findtext("angle"))

                pts = rot_rect_to_poly(cx, cy, w, h, ang)

                # 1) 写 poly GT（像素）
                poly_flat = []
                for (x, y) in pts:
                    poly_flat.extend([x, y])
                f_poly.write(
                    f"{frame},{pid}," + ",".join([f"{v:.4f}" for v in poly_flat]) + "\n"
                )

                # 2) 写 HBB GT（像素，MOT格式）
                x, y, bw, bh = poly_to_hbb(pts)

                valid = 1
                cls = -1
                vis = -1
                ignore = -1
                f_hbb.write(f"{frame},{pid},{x:.2f},{y:.2f},{bw:.2f},{bh:.2f},{valid},{cls},{vis},{ignore}\n")

    print("Done.")
    print("HBB GT :", args.out_gt_hbb)
    print("Poly GT:", args.out_gt_poly)

if __name__ == "__main__":
    main()
