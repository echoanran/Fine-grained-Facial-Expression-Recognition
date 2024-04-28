import os
import cv2
import torch
import random
import math
import numpy as np

from detection.model.regression import Model
from detection.face_detection import FaceDetection
from detection.closed_eye_detection import EyeDetection


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class AUDetection(object):
    def __init__(self, resume, model_args):

        set_seed(2024)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(**model_args).to(self.device)

        checkpoint = torch.load(resume, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.face_detector = FaceDetection()
        self.eye_detector = EyeDetection()

        self.aulist = []
        self.num_class = model_args['num_class']
        if self.num_class == 12:
            self.aulist = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
        else:
            self.aulist = [6, 10, 12, 14, 17]
        self.au_names = ['AU' + str(au) for au in self.aulist]

    def transfer(self, x, y, RotationMatrix):

        x_new = int(
            round(RotationMatrix[0, 0] * x + RotationMatrix[0, 1] * y +
                  RotationMatrix[0, 2]))
        y_new = int(
            round(RotationMatrix[1, 0] * x + RotationMatrix[1, 1] * y +
                  RotationMatrix[1, 2]))

        return x_new, y_new

    def alignment(self, img, landmarks):

        Xs = landmarks[0]
        Ys = landmarks[1]

        eye_center = ((Xs[36] + Xs[45]) * 1. / 2, (Ys[36] + Ys[45]) * 1. / 2)
        dx = Xs[45] - Xs[36]
        dy = Ys[45] - Ys[36]

        angle = math.atan2(dy, dx) * 180. / math.pi

        RotationMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

        new_img = cv2.warpAffine(img, RotationMatrix,
                                 (img.shape[1], img.shape[0]))

        RotationMatrix = np.array(RotationMatrix)
        align_landmarks = []
        align_Xs = []
        align_Ys = []
        for i in range(len(Xs)):
            x, y = self.transfer(Xs[i], Ys[i], RotationMatrix)
            align_Xs.append(x)
            align_Ys.append(y)
        align_landmarks.append(align_Xs)
        align_landmarks.append(align_Ys)

        return new_img, np.array(align_landmarks), RotationMatrix, eye_center

    def cropping(self, img, landmarks, eye_center):

        h, w = img.shape[:2]
        Xs = landmarks[0]
        Ys = landmarks[1]
        size = (Ys[32] - Ys[27]) * 3 // 6 * 6

        y_start = max(0, int(eye_center[1] - size / 3))
        y_end = min(h, int(eye_center[1] + 2 * size / 3))
        x_start = max(0, int(eye_center[0] - size / 2))
        x_end = min(w, int(eye_center[0] + size / 2))
        cropimg = img[y_start:y_end, x_start:x_end]

        crop_landmarks = []
        crop_Xs = []
        crop_Ys = []
        for i in range(len(Xs)):
            x = min(max(0, Xs[i] - x_start), size - 1)
            y = min(max(0, Ys[i] - y_start), size - 1)
            crop_Xs.append(x)
            crop_Ys.append(y)
        crop_landmarks.append(crop_Xs)
        crop_landmarks.append(crop_Ys)

        return cropimg, np.array(crop_landmarks)

    def run(self, frame, face=None, frame_idx=0):
        self.is_eye_open = -1
        self.face = None
        self.img_show = frame.copy()

        if face is not None:
            pass
        else:
            face, face_coordinates, landmarks = self.face_detector.get_face(frame)
            
            if face is None:
                print("Failed to detect face in frame: {}".format(frame_idx))
                self.au_pred = [0.0] * self.num_class
                return self.au_pred + [-1], self.au_names + ['AU43']

            if landmarks is not None:
                left_eye, right_eye, ear, is_eye_open = self.eye_detector.get_eye_state(
                    landmarks)
                self.is_eye_open = float(is_eye_open)

                # for checking
                # all_idxs = self.eye_detector.chosen_left_eye_idxs + self.eye_detector.chosen_right_eye_idxs
                # for idx in all_idxs:
                #     cv2.circle(
                #         self.img_show,
                #         (int(landmarks[idx][0]), int(landmarks[idx][1])), 2,
                #         (0, 0, 255))

            # for checking
            # cv2.rectangle(self.img_show,
            #               (face_coordinates[0], face_coordinates[1]),
            #               (face_coordinates[2], face_coordinates[3]),
            #               (0, 255, 0), 2)
        
        self.face = face.copy()
        if torch.cuda.is_available():
            face = torch.tensor(face).cuda().float().permute(2, 0,
                                                             1).unsqueeze(0)
        else:
            face = torch.tensor(face).float().permute(2, 0, 1).unsqueeze(0)
        face = face / 255.0

        with torch.no_grad():
            au_pred = self.model(face)
        au_pred = au_pred.squeeze().data.cpu().numpy()
        self.au_pred = np.clip(au_pred, 0, 1).tolist()
       
        return self.au_pred + [self.is_eye_open], self.au_names + ['AU43']

    def get_img(self):
        return self.img_show

    def get_face_img(self):
        return self.face


if __name__ == "__main__":

    resume = 'checkpoints/disfa_model.pt'
    model_args = {
        'num_class': 12,
        'backbone': 'resnet34',
        'pooling': True,
        'normalize': True,
        'activation': '',
    }

    AUDet = AUDetection(resume=resume, model_args=model_args)

    # frame = cv2.imread('./resources/test_image.jpg', 1)
    # au_pred = AUDet.run(frame)
    # img_show = AUDet.get_expression_img()
    # cv2.imshow("img", img_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    sample_rate = 1
    videopath = './vid/vid_15s.mp4'

    savename = videopath.split('/')[-1].split('.')[0] + '_' + resume.split(
        '/')[-1].split('.')[0]
    vc = cv2.VideoCapture(videopath)
    fps = vc.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)

    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")

    os.makedirs(os.path.join('./save', savename), exist_ok=True)

    out = cv2.VideoWriter(os.path.join('./save', savename + '_out.avi'),
                          cv2.VideoWriter_fourcc(*'MJPG'), fps,
                          (webcam_image.shape[1], webcam_image.shape[0]))
    clean = cv2.VideoWriter(
        os.path.join('./save',
                     videopath.split('/')[-1].split('.')[0] + '_clean.avi'),
        cv2.VideoWriter_fourcc(*'MJPG'), fps,
        (webcam_image.shape[1], webcam_image.shape[0]))

    cnt = 0
    num_frames = 0
    ori_frames = 0
    all_preds = []
    while read_value:
        read_value, webcam_image = vc.read()

        ori_frames += 1
        if not read_value:
            break

        if num_frames % sample_rate != 0:
            continue

        au_pred, exp_pred, pains = AUDet.run(webcam_image,
                                             ori_count=ori_frames)
        if au_pred is None:
            continue
        num_frames += 1
        all_preds.append(au_pred)
        img_show = AUDet.get_expression_img()

        cv2.imwrite(
            os.path.join(os.path.join('./save', savename),
                         str(cnt).zfill(4) + '.png'), img_show)
        out.write(img_show)
        clean.write(webcam_image)
        cnt += 1

        if cnt >= 3000:
            break

        # cv2.imshow("face", img_show)
        # if cv2.waitKey(1) & 0xff == ord('q'):
        #     break

    print(ori_frames, "in_total, ", num_frames, " frames detetced.")
    np.save(os.path.join('./save', savename + '_all_preds.npy'), all_preds)
    cv2.destroyAllWindows()
    vc.release()
    out.release()
    clean.release()
