Long-term Multi-pig Tracking in Group-housed Pens



This repository provides the implementation of a long-term multi-object tracking system

for group-housed pigs, developed to support the experiments reported in our paper.



The method focuses on maintaining identity consistency over long video sequences under

challenging conditions such as frequent occlusion, dense interactions, appearance similarity,

and visual degradation.



&nbsp;Overview



The proposed system integrates:



A rotated-object detector (YOLOv11-OBB) for pig detection

An appearance-based Re-identification (ReID) network for identity modeling

A trajectory management strategy with temporal smoothing and identity re-linking

Joint motion–appearance data association for robust multi-object tracking



The code is designed for offline evaluation and analysis, rather than real-time deployment.



Repository Structure



text

pig\_track/

├── ultralytics-8.3.163/        # Detection and tracking framework (modified)

│   ├── trackers/               # Customized tracker implementation

│   ├── reid/                   # ReID model and feature extraction

│   ├── utils/                  # Tracking utilities and cost functions

│   └── eval\_mot\_norm.py        # MOT evaluation script

├── scripts/                    # Data processing and evaluation scripts

├── figures/                    # Visualization results used in the paper

├── README.md

Key Components

Detection: Rotated bounding box detection based on YOLOv11-OBB



Re-identification: 64-dimensional appearance embeddings for identity matching



Data Association: Joint cost based on spatial overlap and appearance similarity



Trajectory Management:



Temporal smoothing of appearance features



Short-term lost track buffer



Identity re-linking using a global identity memory



Evaluation

Multi-object tracking performance is evaluated using standard MOT metrics, including:



MOTA



MOTP



IDF1 / IDP / IDR



ID Switch (IDSW)



Evaluation scripts follow the MOTChallenge evaluation protocol.



Notes

This repository is intended to support academic research and result reproducibility.



The code may require adaptation for different datasets or deployment environments.



Some dataset paths and configurations are task-specific and should be adjusted accordingly.

