import os
import cv2
import math

video_path = r"D:/codepig/pig_track/five_pigs_track_9_2.avi"
out_dir = "frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 估算时长（秒）
duration = frame_count / fps if frame_count and fps else None
if duration is None or math.isnan(duration) or duration <= 0:
    # 兜底：尝试从最后位置读一下（有些AVI拿不到frame_count）
    duration = 120  # 不知道就先给个大概，下面会读不到自动跳过

print("fps =", fps, "frame_count =", frame_count, "duration(s)~", duration)

saved = 0
t = 0
max_t = int(duration) + 1

for sec in range(0, max_t):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if not ret:
        continue
    out_path = os.path.join(out_dir, f"t_{sec:06d}.jpg")
    cv2.imwrite(out_path, frame)
    saved += 1

cap.release()
print(f"完成！导出 {saved} 张图片（按秒尝试抓取）")
