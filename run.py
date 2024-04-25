import os
import cv2
import torch
import random
import numpy as np
import warnings
from tqdm import tqdm
from easydict import EasyDict

from detection.au_detection import AUDetection
from detection.exp_mapping import ExpMapping
from utils.draw_fig import *

from collections import Counter

warnings.filterwarnings("ignore")

import mediapipe
faceDetector_m = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def record_video(videopath):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, img = cap.read()
    h, w = img.shape[1], img.shape[0]
    out = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'MJPG'), fps,
                          (h, w))
    total_frames = 0
    print("正在录制中.............")
    while (cap.isOpened()):
        _, img = cap.read()
        out.write(img)
        cv2.imshow("正在录制中", img)
        total_frames += 1
        k = cv2.waitKey(10)  # Esc to escape
        if k & 0xFF == ord('q'):
            print("您已按q键, 录制已结束。")
            break
    cap.release()
    out.release()

    print('record video and save to {}'.format(videopath))
    return


def main(args,
         is_reload=False,
         is_ending=False,
         renew_exp=False,
         is_draw=True):

    set_seed(2024)
    savefolder = os.path.join('./save', args.savename)
    os.makedirs(savefolder, exist_ok=True)

    if args.mode == 'record':
        videopath = os.path.join('./resources', args.savename + '.avi')
        record_video(videopath)
    elif args.mode == 'load':
        if args.videopath is None:
            print("Parameter videopath is required for load mode.")
            raise ValueError
        videopath = args.videopath
    else:
        print("Input mode {} not implemented.".format(args.mode))
        raise ValueError

    print('Load video from {}'.format(videopath))
    vc = cv2.VideoCapture(videopath)
    total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    args.max_frames = min(args.max_frames, total_frames)
    fps = vc.get(cv2.CAP_PROP_FPS)
    print('Start processing: {} frames in total, {} fps.'.format(
        total_frames, fps))

    width, height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    outpath = os.path.join(savefolder, args.savename + '.avi')

    ExpMap = ExpMapping(selected_exps=args.selected_exps)
    exp_names = ExpMap.get_expnames()

    if is_reload:
        print('Reload for drawing.')
        os.makedirs(os.path.join(savefolder, 'timing_diagram'), exist_ok=True)

        au_preds = np.load(os.path.join(savefolder, 'au_preds.npy'))
        aulist = [1, 2, 4, 5, 6, 9, 10, 12, 14, 15, 17, 20, 25, 26, 43]
        au_names = ['AU' + str(au) for au in aulist]

        if renew_exp:
            exp_preds = []
            for au_pred in au_preds:
                exp_pred, exp_names = ExpMap.forward(np.array(au_pred))
                exp_preds.append(exp_pred)
            exp_preds = np.array(exp_preds)
            neutral_thres = ExpMap.get_neutral_thres(exp_preds, savepath=os.path.join(savefolder, 'kde.png'))
            print(neutral_thres)
            exp_preds = []
            for au_pred in au_preds:
                exp_pred, exp_names = ExpMap.forward(np.array(au_pred), neutral_thres)
                exp_preds.append(exp_pred)
            exp_preds = ExpMap.global_smoothing(exp_preds)
        else:
            exp_preds = np.load(os.path.join(savefolder, 'exp_preds.npy'))

        if not is_draw:
            return exp_preds, exp_names, outpath
        else:
            ylim = [
                np.round(exp_preds.min(), 1),
                np.round(exp_preds.max(), 1) + 0.1
            ]
            T = min(args.max_frames, len(exp_preds))
            fig = plt.figure(figsize=(30, 6), dpi=330)
            for frame_idx in tqdm(range(1, T + 1)):
                if args.is_topk:
                    draw_expfig_topk(exp_preds[:frame_idx, :],
                                    fps,
                                    exp_names,
                                    T=T,
                                    topk=args.topk,
                                    transparent=True,
                                    ylim=ylim,
                                    savepath=os.path.join(
                                        savefolder, 'timing_diagram',
                                        str(frame_idx).zfill(4) + '.png'),
                                    fig=fig)
                else:
                    draw_expfig_thres(exp_preds[:frame_idx, :],
                                      fps,
                                      exp_names,
                                      T=T,
                                      thres=args.thres,
                                      transparent=True,
                                      savepath=os.path.join(
                                          savefolder, 'timing_diagram',
                                          str(frame_idx).zfill(4) + '.png'),
                                      fig=fig)
                fig.clf()
            plt.close('all')
            # pass
            
    else:
        os.makedirs(os.path.join(savefolder, 'detected_face'), exist_ok=True)

        resume = './detection/checkpoints/MCS13_repeat_balance_0.pt'  # disfa
        model_args = {
            'num_class': 12,
            'backbone': 'resnet34',
            'pooling': True,
            'normalize': True,
            'activation': '',
        }
        AUDet = AUDetection(resume=resume, model_args=model_args)

        resume_bp4d = './detection/checkpoints/MCS3_final_balance_model.pt'  # bp4d
        model_args_bp4d = {
            'num_class': 5,
            'backbone': 'resnet34',
            'pooling': True,
            'normalize': True,
            'activation': '',
        }
        AUDet_bp4d = AUDetection(resume=resume_bp4d,
                                 model_args=model_args_bp4d)
        au_preds, exp_preds = [], []

    frame_idx = 0
    exp_tops = []
    all_weights = np.array([1/2, 1/4, 1/8, 1/16, 1/32])
    weights = all_weights[:args.topk] / sum(all_weights[:args.topk])
    exp_preds_cum = []
    ending_image = None

    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MJPG'), fps,
                          (height, width))

    while True:
        if frame_idx % args.sample_rate != 0:
            continue

        read_value, webcam_image = vc.read()
        if not read_value:
            break

        face_infos_m = faceDetector_m.process(webcam_image)
        if face_infos_m.multi_face_landmarks == None:
            print("Failed to detect face in loop for frame:", frame_idx)
            continue

        ending_image = webcam_image
        frame_idx += 1

        if is_reload:
            au_pred = au_preds[frame_idx - 1]
            exp_pred = exp_preds[frame_idx - 1]
        else:
            au_pred, au_names = AUDet.run(webcam_image, frame_idx=frame_idx)
            au_pred_bp4d, au_names_bp4d = AUDet_bp4d.run(
                webcam_image, face=AUDet.get_face_img(), frame_idx=frame_idx)

            au_pred_all = au_pred[:6] + [au_pred_bp4d[1]] + [au_pred[6]] + [
                au_pred_bp4d[3]
            ] + au_pred[7:]
            au_names_all = au_names[:6] + [au_names_bp4d[1]] + [
                au_names[6]
            ] + [au_names_bp4d[3]] + au_names[7:]

            au_pred, au_names = au_pred_all, au_names_all
            exp_pred, exp_names = ExpMap.forward(np.array(au_pred))

        if is_reload:
            # get cumulated exp
            sorted_all = sorted(enumerate(exp_pred), key=lambda x: x[1], reverse=True)
            sorted_idx = [x[0] for x in sorted_all]
            topk_idxs = sorted_idx[:args.topk]
            exp_pred_new = exp_pred.copy()
            exp_pred_new[exp_pred_new < neutral_thres] = 0
            exp_pred_new[exp_pred_new >= neutral_thres] = 1
            for c in range(len(exp_names)): 
                if c in topk_idxs and c != len(exp_names) - 1:
                    exp_pred_new[c] = weights[topk_idxs.index(c)]
                else:
                    exp_pred_new[c] = 0
            exp_preds_cum.append(exp_pred_new)
            exp_final = np.argmax(np.sum(exp_preds_cum[max(0, frame_idx - 1 - 250): frame_idx], 0))

            img_show = webcam_image.copy()
            img_show = draw(img_show,
                            exp_final,
                            au_names,
                            exp_names,
                            au_pred,
                            exp_pred,
                            savefolder=savefolder,
                            frame_idx=frame_idx if is_reload else -1,
                            thres=args.thres,
                            topk=args.topk,
                            is_topk=args.is_topk)
        else:
            img_show = webcam_image.copy()
            
            # img_show = AUDet.get_img()
            face = AUDet.get_face_img()
            if face is not None:
                cv2.imwrite(
                    os.path.join(savefolder, 'detected_face',
                                 str(frame_idx).zfill(4) + '.png'), face)
            au_preds.append(au_pred)
            exp_preds.append(exp_pred)

            print('=' * 10, 'AU Intensity Results', '=' * 10)
            for name, pred in zip(au_names, au_pred):
                print(name + ': ', np.round(pred * 5, 2))
            print('=' * 10, 'Fine-grained Facial Expression Results', '=' * 10)
            for name, pred in zip(exp_names, exp_pred):
                print(name + ': ', np.round(pred, 2))

        out.write(img_show)

        if frame_idx >= args.max_frames:
            break

        i = (int)((100 * frame_idx) / min(args.max_frames, total_frames))
        print("\r", "Processing: {}%:".format(i) + "-" * i, end="", flush=True)

    if not is_reload:
        np.save(os.path.join(savefolder, 'au_preds.npy'), au_preds)
        exp_preds = ExpMap.global_smoothing(exp_preds)
        np.save(os.path.join(savefolder, 'exp_preds.npy'), exp_preds)

    if is_ending:
        text = "In the video, you are likely to be " + exp_names[np.max(
            exp_tops)].split(' ')[0] + "."
        blk = np.zeros(ending_image.shape, np.uint8)  # transparent background
        cv2.rectangle(blk, (int(height * 0.5) - 350, int(width * 0.5) - 50),
                      (int(height * 0.5) + 420, int(width * 0.5) + 20),
                      (155, 155, 155),
                      thickness=-1)
        webcam_image = cv2.addWeighted(ending_image, 1.0, blk, 0.5, 1)
        cv2.putText(webcam_image, text,
                    (int(height * 0.5) - 350, int(width * 0.5)),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (155, 155, 155), 2)
        for i in range(40):
            out.write(webcam_image)

    print('\nEnd processing: {} frames in total, {} frames processed.'.format(
        total_frames, frame_idx))
    vc.release()
    out.release()

    exp_preds = np.array(exp_preds)

    return exp_preds, exp_names, outpath


if __name__ == "__main__":

    args = EasyDict()

    args.savename = 'dragon_tmp'
    args.mode = 'load'
    args.videopath = 'resources/dragon_clean.avi'

    args.sample_rate = 1
    args.max_frames = 3000
    args.selected_exps = list(range(0, 7, 1)) + [23, 24] + list(
        range(26, 31, 1))

    args.neutral_thres = 0.3
    args.thres = 0.4
    args.topk = 3
    args.is_topk = True

    args.window_size = 150

    exp_preds, exp_names, outpath = main(args,
                                         is_reload=False,
                                         is_ending=False,
                                         renew_exp=False,
                                         is_draw=False)

    exp_preds, exp_names, outpath = main(args,
                                         is_reload=True,
                                         is_ending=False,
                                         renew_exp=True,
                                         is_draw=True)

    vc = cv2.VideoCapture(outpath)
    fps = vc.get(cv2.CAP_PROP_FPS)
    savefolder = os.path.join('./save', args.savename)

    draw_expfig_thres(exp_preds,
                      fps,
                      exp_names,
                      thres=0.0,
                      transparent=False,
                      savepath=os.path.join(savefolder,
                                            'timing_diagram_thres_raw.png'))

    ExpMap = ExpMapping(selected_exps=args.selected_exps)
    exp_names = ExpMap.get_expnames()

    draw_expfig_thres(exp_preds,
                      fps,
                      exp_names,
                      thres=0.0,
                      transparent=False,
                      savepath=os.path.join(savefolder,
                                            'timing_diagram_thres0.0.png'))

    ylim = [np.round(exp_preds.min(), 1), np.round(exp_preds.max(), 1) + 0.1]
    
    analysis_results = ExpMap.automatic_analysis(exp_preds)

    draw_expfig_topk(exp_preds,
                     fps,
                     exp_names,
                     topk=3,
                     transparent=False,
                     ylim=ylim,
                     savepath=os.path.join(savefolder,
                                           'timing_diagram_topk3.png'))

    draw_expfig_topk(exp_preds,
                     fps,
                     exp_names,
                     topk=3,
                     transparent=False,
                     ylim=ylim,
                     savepath=os.path.join(savefolder,
                                           'timing_diagram_analysis.png'),
                     analysis_results=analysis_results)

    neutral_thres = ExpMap.get_neutral_thres(exp_preds, savepath=os.path.join(savefolder, 'kde.png'))
    
    exp_preds[exp_preds <= neutral_thres] = 0
    exp_preds[exp_preds > neutral_thres] = 1

    exp_topk = []
    sorted_idxs = []
    for i in range(len(exp_preds)):
        sorted_all = sorted(enumerate(exp_preds[i]), key=lambda x: x[1], reverse=True)
        sorted_idxs.append([x[0] for x in sorted_all])

    weights = np.array([1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128])
    weights = weights[:args.topk] / sum(weights[:args.topk])

    for i in range(len(exp_preds)):
        topk_idxs = sorted_idxs[i][:args.topk]
        for c in range(len(exp_names)): 
            if c in topk_idxs and c != len(exp_names) - 1:
                exp_preds[i][c] = weights[topk_idxs.index(c)]
            else:
                exp_preds[i][c] = 0
        
    exp_preds_new = []
    for i in range(len(exp_preds)):
        exp_preds_new.append(np.sum(exp_preds[max(0, i - 250): i + 1], 0))
    exp_preds_new = np.array(exp_preds_new)

    draw_expfig_max(exp_preds_new,
                    fps,
                    exp_names,
                    savepath=os.path.join(savefolder, 'exp_tops.png'),
                    )
