# ultralytics/trackers/utils/reid_infer.py
import onnxruntime as ort
import numpy as np
import cv2

class ReidONNX:
    def __init__(self, onnx_path, providers=('CUDAExecutionProvider','CPUExecutionProvider')):
        self.sess = ort.InferenceSession(onnx_path, providers=list(providers))
        self.iname = self.sess.get_inputs()[0].name

    def _pre(self, img, xyxy, size=(128,256)):  # (W,H) for OSNet
        x1,y1,x2,y2 = map(int, xyxy)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: crop = img
        crop = cv2.resize(crop, size)[:,:,::-1].astype(np.float32)/255.0
        mean = np.array([0.485,0.456,0.406],dtype=np.float32)
        std  = np.array([0.229,0.224,0.225],dtype=np.float32)
        crop = (crop-mean)/std
        crop = np.transpose(crop,(2,0,1))[None,...]
        return crop

    def feat(self, img, xyxy):
        inp = self._pre(img, xyxy)
        f = self.sess.run(None, {self.iname: inp})[0][0]
        n = np.linalg.norm(f) + 1e-12
        return f / n
