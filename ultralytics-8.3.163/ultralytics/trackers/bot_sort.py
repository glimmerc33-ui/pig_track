# Ultralytics 🚀 AGPL-3.0 License - [https://ultralytics.com/license](https://ultralytics.com/license)

from collections import deque
from typing import Any, List, Optional

import numpy as np
import torch

from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.plotting import save_one_box

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH

class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLO, adding object tracking features.

    ```
    This class extends the STrack class to include additional functionalities for object tracking, such as feature
    smoothing, Kalman filter prediction, and reactivation of tracks.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features: Update features vector and smooth it using exponential moving average.
        predict: Predict the mean and covariance using Kalman filter.
        re_activate: Reactivate a track with updated features and optionally new ID.
        update: Update the track with new detection and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict: Predict the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords: Convert tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh: Convert bounding box to xywh format `(center x, center y, width, height)`.

    Examples:
        Create a BOTrack instance and update its features
        >>> bo_track = BOTrack(tlwh=[100, 50, 80, 40], score=0.9, cls=1, feat=np.random.rand(128))
        >>> bo_track.predict()
        >>> new_track = BOTrack(tlwh=[110, 60, 80, 40], score=0.85, cls=1, feat=np.random.rand(128))
        >>> bo_track.update(new_track, frame_id=2)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(
        self, tlwh: np.ndarray, score: float, cls: int, feat: Optional[np.ndarray] = None, feat_history: int = 50
    ):
        """
        Initialize a BOTrack object with temporal parameters, such as feature history, alpha, and current features.

        Args:
            tlwh (np.ndarray): Bounding box coordinates in tlwh format (top left x, top left y, width, height).
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            feat (np.ndarray, optional): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.

        Examples:
            Initialize a BOTrack object with bounding box, score, class ID, and feature vector
            >>> tlwh = np.array([100, 50, 80, 120])
            >>> score = 0.9
            >>> cls = 1
            >>> feat = np.random.rand(128)
            >>> bo_track = BOTrack(tlwh, score, cls, feat)
        """
        super().__init__(tlwh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        self.strict_until = -1  # 帧号；在这个帧号前关联更严格

    def update_features(self, feat: np.ndarray) -> None:
        """Update the feature vector and apply exponential moving average smoothing."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self) -> None:
        """Predict the object's future state using the Kalman filter to update its mean and covariance."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track: "BOTrack", frame_id: int, new_id: bool = False) -> None:
        """Reactivate a track with updated features and optionally assign a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)
        # 设置严格期
        self.strict_until = frame_id + getattr(self, "strict_frames", 0)

    def update(self, new_track: "BOTrack", frame_id: int) -> None:
        """Update the track with new detection information and the current frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)
        # 设置严格期
        self.strict_until = frame_id + getattr(self, "strict_frames", 0)

    @property
    def tlwh(self) -> np.ndarray:
        """Return the current bounding box position in `(top left x, top left y, width, height)` format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: List["BOTrack"]) -> None:
        """Predict the mean and covariance for multiple object tracks using a shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert tlwh bounding box coordinates to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh (top-left-width-height) to xywh (center-x-center-y-width-height) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLO, designed for object tracking with ReID and GMC algorithm.

    ```
    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (Any): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (Any): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter: Return an instance of KalmanFilterXYWH for object tracking.
        init_track: Initialize track with detections, scores, and classes.
        get_dists: Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict: Predict and track multiple objects with a YOLO model.
        reset: Reset the BOTSORT tracker to its initial state.

    Examples:
        Initialize BOTSORT and process detections
        >>> bot_sort = BOTSORT(args, frame_rate=30)
        >>> bot_sort.init_track(dets, scores, cls, img)
        >>> bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with a YOLO object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """
        Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args (Any): Parsed command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video being processed.

        Examples:
            Initialize BOTSORT with command-line arguments and a specified frame rate:
            >>> args = parse_args()
            >>> bot_sort = BOTSORT(args, frame_rate=30)
        """
        super().__init__(args, frame_rate)
        self.gmc = GMC(method=args.gmc_method)

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = (
            (lambda feats, s: [f.cpu().numpy() for f in feats])  # native features do not require any model
            if args.with_reid and self.args.model == "auto"
            else ReID(args.model)
            if args.with_reid
            else None
        )

        # ↓↓↓ 新增：全局身份库 + 阈值/超参 ↓↓↓
        self.id_bank = {}                       # id -> {'feat': np.ndarray(d), 'age': int}
        self.tau_reid_create = getattr(args, "tau_reid_create", 0.30)   # 等价 cos > 0.70
        self.bank_momentum   = getattr(args, "bank_momentum", 0.90)     # 滑动平均
        self.bank_max_age    = getattr(args, "bank_max_age", 3000)       # 老化清理
        
        # ↓↓↓ 新增：严格期参数 ↓↓↓
        self.strict_frames = getattr(args, "strict_frames", 10)  # 5~15
        self.strict_bonus = getattr(args, "strict_bonus", 0.1)   # 在严格期把 appearance_thresh 再提高 0.1

    def _dedup_dets_with_nms_and_reid(self, img, dets_xyxy, scores, cls):
        """同帧去重：NMS + ReID合并"""
        if len(dets_xyxy) == 0:
            return dets_xyxy, scores, cls
            
        # 1) NMS 先去掉几何上高度重叠的框
        iou_thr = getattr(self.args, "det_nms_iou", 0.6)
        keep = torch.ops.torchvision.nms(
            torch.from_numpy(dets_xyxy), 
            torch.from_numpy(scores), 
            iou_thr
        ).numpy()
        dets_xyxy = dets_xyxy[keep]
        scores = scores[keep]
        cls = cls[keep]

        # 2) ReID 合并：IoU>0.7 且 (1-cos_sim)<tau_merge 才认为是同一头
        tau_merge = getattr(self.args, "reid_merge_tau", 0.25)   # 可 0.20~0.35
        merged = []
        used = set()

        # 预抽特征（避免重复裁图）
        if self.args.with_reid and self.encoder is not None and len(dets_xyxy):
            crops_xywh = xyxy2xywh(torch.from_numpy(dets_xyxy)).numpy()
            # 添加angle和idx列（如果原始dets有的话）
            if crops_xywh.shape[1] == 4:
                crops_xywh = np.concatenate([crops_xywh, np.zeros((len(crops_xywh), 2))], axis=1)
            feats = self.encoder(img, crops_xywh)
            feats = [f/np.linalg.norm(f) for f in feats]
        else:
            feats = [None] * len(dets_xyxy)

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / (area_a + area_b - inter + 1e-6)

        for i in range(len(dets_xyxy)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(dets_xyxy)):
                if j in used:
                    continue
                if cls[i] != cls[j]:  # 只同类
                    continue
                if iou(dets_xyxy[i], dets_xyxy[j]) < 0.7:
                    continue
                if feats[i] is not None and feats[j] is not None:
                    cos_sim = float(np.dot(feats[i], feats[j]))
                    if 1.0 - cos_sim > tau_merge:
                        continue
                # 合并
                group.append(j)
            # 组内保留分数最高的框
            best = max(group, key=lambda k: scores[k])
            merged.append(best)
            used.update(group)

        idx = np.array(merged, dtype=int)
        return dets_xyxy[idx], scores[idx], cls[idx]

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return an instance of KalmanFilterXYWH for predicting and updating object states in the tracking process."""
        return KalmanFilterXYWH()

    def init_track(
        self, dets: np.ndarray, scores: np.ndarray, cls: np.ndarray, img: Optional[np.ndarray] = None
    ) -> List[BOTrack]:
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features."""
        if len(dets) == 0:
            return []
            
        # dets 可能是 xywhr 或 xywh；先转 xyxy 做 NMS
        if dets.shape[1] >= 4:
            xyxy = xywh2xyxy(torch.from_numpy(dets[:, :4])).numpy()
        else:
            return []  # 容错

        # 同帧去重
        xyxy, scores, cls = self._dedup_dets_with_nms_and_reid(img, xyxy, scores, cls)
        
        # 再转回 xywh 供后续 STrack 使用
        xywh = xyxy2xywh(torch.from_numpy(xyxy)).numpy()
        if dets.shape[1] > 4:
            xywh = np.concatenate([xywh, dets[xyxy.shape[0]:, 4:]], axis=1)

        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, xywh)
            tracks = [BOTrack(x, s, c, f) for (x, s, c, f) in zip(xywh, scores, cls, features_keep)]
        else:
            tracks = [BOTrack(x, s, c) for (x, s, c) in zip(xywh, scores, cls)]
            
        # 为新轨迹设置严格期
        for track in tracks:
            track.strict_frames = self.strict_frames
            track.strict_until = self.frame_id + self.strict_frames
            
        return tracks

    def get_dists(self, tracks: List[BOTrack], detections: List[BOTrack]) -> np.ndarray:
        """Calculate distances between tracks and detections using IoU and optionally ReID embeddings."""
        d_iou = matching.iou_distance(tracks, detections)
        mask_far = d_iou > (1.0 - self.proximity_thresh)  # 原有IoU过滤

        # 中心门限：中心点距离超过阈值(相对对角线比例)直接置1
        gate = getattr(self.args, "center_gate", 0.2)  # 0.15~0.3
        if gate > 0 and len(tracks) and len(detections):
            d_center = matching.center_distance(tracks, detections)  # 归一化到[0,1]
            d_iou[d_center > gate, :] = 1.0
            d_iou[:, d_center.T > gate] = 1.0  # 两侧都 gate

        if self.args.fuse_score:
            d_iou = matching.fuse_score(d_iou, detections)

        if self.args.with_reid and self.encoder is not None:
            d_app = matching.embedding_distance(tracks, detections) / 2.0
            d_app[mask_far] = 1.0

            # 严格期处理：若任何一端处在严格期，则阈值更苛刻
            now = getattr(self, "frame_id", 0)
            for ti, tr in enumerate(tracks):
                if hasattr(tr, "strict_until") and now <= tr.strict_until:
                    d_app[ti, :] += self.strict_bonus
            
            # 外观硬阈
            app_thr = (1.0 - self.appearance_thresh)
            d_app[d_app > app_thr] = 1.0

            # 代价融合：α*几何 + (1-α)*外观
            alpha = getattr(self.args, "cost_alpha", 0.5)  # 0.4~0.6
            dists = alpha * d_iou + (1.0 - alpha) * d_app
            return dists
        else:
            return d_iou

    def multi_predict(self, tracks: List[BOTrack]) -> None:
        """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self) -> None:
        """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()


class ReID:
    """YOLO model as encoder for re-identification."""

    def __init__(self, model: str):
        """
        Initialize encoder for re-identification.

        Args:
            model (str): Path to the YOLO model for re-identification.
        """
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False, save=False)  # init

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> List[np.ndarray]:
        """Extract embeddings for detected objects."""
        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]