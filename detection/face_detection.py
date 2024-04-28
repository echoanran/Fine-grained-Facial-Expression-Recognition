import os
import cv2
import numpy as np

import mediapipe


class FaceDetection():
    def __init__(self) -> None:

        self.faceDetector_m = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True)

    def get_face(self, image, resize=(256, 256)):
        face_infos_m = self.faceDetector_m.process(image)

        if face_infos_m.multi_face_landmarks == None:
            return None, None, None

        for prediction in face_infos_m.multi_face_landmarks:
            self.landmarks = np.array(
                [(pt.x * image.shape[1], pt.y * image.shape[0])
                 for pt in prediction.landmark],
                dtype=np.float64)
            bbox = np.vstack(
                [self.landmarks.min(axis=0),
                 self.landmarks.max(axis=0)])
            bbox_int = np.round(bbox).astype(np.int32)
            break

        cutted_faces = [
            image[bbox_int[0][1]:bbox_int[1][1], bbox_int[0][0]:bbox_int[1][0]]
        ]
        self.faces_coordinates = [
            bbox_int[0][0], bbox_int[0][1], bbox_int[1][0], bbox_int[1][1]
        ]
        self.normalized_faces = [
            cv2.resize(face, resize) for face in cutted_faces
            if face.shape[0] != 0 and face.shape[1] != 0
        ]

        if not self.normalized_faces:
            return None, None, None
        return self.normalized_faces[0], self.faces_coordinates, self.landmarks

    def get_landmark(self, image):
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

    def draw_landmark(self, savepath, all_idxs=[]):
        if not all_idxs:
            all_idxs = list(range(len(self.landmarks)))
        for idx in all_idxs:
            cv2.circle(
                self.image,
                (int(self.landmarks[idx][0]), int(self.landmarks[idx][1])), 2,
                (0, 0, 255), -1)

        cv2.imwrite(savepath, self.image)
        return

    def draw_face(self, savepath):
        cv2.rectangle(self.image,
                      (self.face_coordinates[0], self.face_coordinates[1]),
                      (self.face_coordinates[2], self.face_coordinates[3]),
                      (0, 255, 0), 2)

        cv2.imwrite(savepath, self.image)
        return


if __name__ == '__main__':

    face_detector = FaceDetection()

    image = cv2.imread('../save/tmp.png')

    face_detector.get_landmark(image)

    all_idxs = [
        33,
        133,
        158,
        145,
        362,
        263,
        385,
        374,  # eye
        63,
        107,
        336,
        293,  # brow
        6,
        5,
        1,
        102,
        331,  # nose
        0,
        17,
        61,
        291,
        37,
        267,
        181,
        405,  # mouth
    ]
    face_detector.draw_landmark(savepath='../save/tmp_draw.png',
                                all_idxs=all_idxs)
