# -*- coding: utf-8 -*-
import os, re, glob, math, argparse
import xml.etree.ElementTree as ET

def rot_rect_to_poly(cx, cy, w, h, angle):
    hw, hh = w/2.0, h/2.0
    corners = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    ca, sa = math.cos(angle), math.sin(angle)
    pts=[]
    for x,y in corners:
        xr = x*ca - y*sa + cx
        yr = x*sa + y*ca + cy
        pts.append((xr,yr))
    return pts

def poly_to_yolo_xywh_norm(pts, img_w, img_h):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    x1,x2=min(xs),max(xs); y1,y2=min(ys),max(ys)
    xc=(x1+x2)/2.0; yc=(y1+y2)/2.0; w=(x2-x1); h=(y2-y1)
    return xc/img_w, yc/img_h, w/img_w, h/img_h

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--out_label_dir", required=True, help="输出labels目录（与images同层结构一致）")
    ap.add_argument("--img_w", type=float, default=1280.0)
    ap.add_argument("--img_h", type=float, default=736.0)
    ap.add_argument("--cls_id", type=int, default=0)  # 全部猪视为同一类
    args=ap.parse_args()

    os.makedirs(args.out_label_dir, exist_ok=True)
    xmls=sorted(glob.glob(os.path.join(args.xml_dir,"*.xml")))
    for xp in xmls:
        root=ET.parse(xp).getroot()
        filename=root.findtext("filename") or os.path.basename(xp).replace(".xml",".jpg")
        stem=os.path.splitext(os.path.basename(filename))[0]
        out_txt=os.path.join(args.out_label_dir, stem + ".txt")

        lines=[]
        for obj in root.findall("object"):
            rob=obj.find("robndbox")
            if rob is None:
                continue
            cx=float(rob.findtext("cx")); cy=float(rob.findtext("cy"))
            w=float(rob.findtext("w")); h=float(rob.findtext("h"))
            ang=float(rob.findtext("angle"))
            pts=rot_rect_to_poly(cx,cy,w,h,ang)
            xc,yc,wn,hn=poly_to_yolo_xywh_norm(pts,args.img_w,args.img_h)
            lines.append(f"{args.cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        with open(out_txt,"w",encoding="utf-8") as f:
            f.write("\n".join(lines))

    print("Done:", args.out_label_dir)

if __name__=="__main__":
    main()
