import os
import cv2
import torch
import time
import random
import math
import numpy as np

import mediapipe


class LandmarkDetection():
    def __init__(self) -> None:

        self.faceDetector_m = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True)

    def run(self, image):
        self.image = image
        face_infos_m = self.faceDetector_m.process(image)

        if face_infos_m.multi_face_landmarks == None:
            return None

        for prediction in face_infos_m.multi_face_landmarks:
            self.landmarks = np.array(
                [(pt.x * image.shape[1], pt.y * image.shape[0])
                 for pt in prediction.landmark],
                dtype=np.float64)
            break

        return self.landmarks

    def draw(self, savepath, all_idxs=[]):
        if not all_idxs:
            all_idxs = list(range(len(self.landmarks)))
        for idx in all_idxs:
            cv2.circle(
                self.image,
                (int(self.landmarks[idx][0]), int(self.landmarks[idx][1])), 2,
                (0, 0, 255), -1)

        cv2.imwrite(savepath, self.image)
        return


if __name__ == '__main__':

    ldmk_detector = LandmarkDetection()

    image = cv2.imread('../save/tmp.png')

    ldmk_detector.run(image)

    all_idxs = [33, 133, 158, 145, 362, 263, 385, 374,  # eye
                63, 107, 336, 293,  # brow
                6, 5, 1, 102, 331,  # nose
                0, 17, 61, 291, 37, 267, 181, 405,  # mouth
                ]
    ldmk_detector.draw(savepath='../save/tmp_draw.png', all_idxs=all_idxs)
