import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


def draw_bg(img, y1, x1, y2, x2):
    scale = 0.25
    for i in range(x1, x2):
        for j in range(y1, y2):
            img[i][j][0] = min(255, int(img[i][j][0] * scale))
            img[i][j][1] = min(255, int(img[i][j][1] * scale))
            img[i][j][2] = min(255, int(img[i][j][2] * scale))
    return


def get_env(image):
    rgb_image = image
    thres_low = 100
    thres_high = 200
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    if image.mean() < thres_low:
        return -1
    elif image.mean() > thres_high:
        return 1
    return 0


def cv2ImgAddText(img,
                  text,
                  left,
                  top,
                  textColor=(0, 255, 0),
                  textSize=20,
                  is_bold=False):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("utils/font/simsun.ttc",
                                   textSize,
                                   encoding="utf-8")
    draw.text((top, left),
              text,
              textColor,
              font=fontStyle.bold() if is_bold else fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_expfig_max(exp_preds,
                    fps,
                    exp_names,
                    savepath='./timing_dragram_max.png',
                    fig=None,
                    exp_final='None'):
    color_sep = 0.02
    palette = np.array(sns.color_palette('YlOrRd', int(1 / color_sep) + 1))

    if not fig:
        fig = plt.figure(figsize=(12, 10), dpi=330)

    labels = []
    for exp_name in exp_names:
        labels.append(exp_name.split(' ')[0])
    if exp_final != 'None':
        exp_list = exp_final
        print(len(exp_list))
    else:
        exp_list = np.argmax(exp_preds, axis=1)
    x = list(range(len(exp_list)))
    x = [a / fps for a in x]
    y = np.max(exp_preds, axis=1)

    # colors = [palette[int(a / color_sep)] for a in y]
    colors = [palette[-1] for a in y]
    plt.scatter(x, exp_list, c=colors, s=10, label='   ')
    plt.xlabel("Time/s", fontdict={'size': 16})
    plt.yticks(range(0, len(exp_names), 1), labels=labels)
    plt.title("Fine-grained Facial Expression Recognition",
              fontdict={'size': 20})
    plt.savefig(savepath)
    plt.clf()
    print(savepath)
    return


def draw_expfig_thres(exp_preds,
                      fps,
                      exp_names,
                      T=None,
                      thres=0.4,
                      transparent=False,
                      savepath='./timing_dragram_thres.png',
                      fig=None):
    exp_preds = np.array(exp_preds)

    palette = np.array(sns.color_palette('Paired', 10))
    palette2 = sns.xkcd_palette([
        'cloudy blue', 'slate blue', 'yellow brown', 'grey brown', 'dark mauve'
    ])
    palette = np.concatenate([palette, palette2])

    if not fig:
        fig = plt.figure(figsize=(30, 6), dpi=660)

    exp_list = np.argmax(exp_preds, axis=1)
    if T is None:
        T = len(exp_list)
    x = list(range(T))
    x = [a / fps for a in x]

    for i in range(len(exp_names) - 1):
        xx = []
        yy = []
        colors = []
        for t in range(len(exp_list)):
            if exp_preds[t][i] >= thres:
                yy.append(exp_preds[t][i])
            else:
                exp_preds[t][i] = 0
                yy.append(0)
            xx.append(x[t])
            colors.append(palette[i])

        plt.plot(xx,
                 yy,
                 label=exp_names[i].split(' ')[0],
                 c=palette[i],
                 linewidth=2)

    if transparent:
        fontdict = {'size': 26, 'color': 'white', 'fontweight': 'bold'}
        legend_font_size = 16
        plt.vlines([x[len(exp_list) - 1]],
                   0,
                   1,
                   linewidth=3,
                   linestyles='dashed',
                   colors='red')
    else:
        fontdict = {'size': 26, 'color': 'black'}
        legend_font_size = 15
        plt.title("Fine-grained Facial Expression Recognition",
                  fontdict=fontdict)

    plt.xlim(0, x[-1] + x[-1] // 6)
    plt.xticks(np.arange(0, x[-1] + x[-1] // 6, 20).tolist(),
               fontsize=22,
               color=fontdict['color'])
    if thres == 0:
        plt.ylim(0.01, 1.0)
        plt.yticks(np.arange(0, 1.1, 0.2).tolist(),
                   fontsize=22,
                   color=fontdict['color'])
    else:
        plt.ylim(thres, 0.7)
        plt.yticks(np.arange(thres, 0.8, 0.1).tolist(),
                   fontsize=22,
                   color=fontdict['color'])

    plt.xlabel("Time/s", fontdict=fontdict)
    plt.ylabel("Intensity", fontdict=fontdict)
    plt.legend(loc="lower right", fontsize=legend_font_size)

    if transparent:
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['right'].set_visible(False)

        plt.gca().spines['bottom'].set_linewidth(3)
        plt.gca().spines['left'].set_linewidth(3)

        plt.gca().spines['bottom'].set_color('w')
        plt.gca().spines['left'].set_color('w')

    plt.gcf().subplots_adjust(left=0.12, bottom=0.13)
    plt.savefig(savepath, transparent=transparent)
    plt.clf()
    return


def draw_expfig_topk(exp_preds,
                     fps,
                     exp_names,
                     T=None,
                     topk=2,
                     transparent=False,
                     ylim=[0.0, 1.0],
                     savepath='./timing_dragram_topk.png',
                     fig=None,
                     analysis_results=None,
                     analysis_draw=[0, 1, 2]):
    exp_preds = np.array(exp_preds)

    palette = np.array(sns.color_palette('Paired', 10))
    palette2 = sns.xkcd_palette([
        'cloudy blue', 'slate blue', 'yellow brown', 'grey brown', 'dark mauve'
    ])
    palette = np.concatenate([palette, palette2])

    if not fig:
        fig = plt.figure(figsize=(30, 6), dpi=330)

    exp_list = np.argmax(exp_preds, axis=1)
    if T is None:
        T = len(exp_list)
    x = list(range(T))
    x = [a / fps for a in x]

    # plt.hlines(0.3, x[0], x[-1], linestyles="--", colors='black', alpha=0.5)

    if analysis_results is not None:
        result1, result2, result3 = analysis_results
        if 0 in analysis_draw:
            for t, v in enumerate(result1):
                if v > 0:
                    plt.vlines(x[t],
                            ylim[0],
                            ylim[1],
                            linestyles="-",
                            colors='blue',
                            alpha=0.05)
        if 1 in analysis_draw:
            for t, v in enumerate(result2):
                if v > 0:
                    plt.vlines(x[t],
                            ylim[0],
                            ylim[1],
                            linestyles="-",
                            colors='red',
                            alpha=0.05)
        if 2 in analysis_draw:
            for t, v in enumerate(result3):
                if v > 0:
                    plt.vlines(x[t],
                            ylim[0],
                            ylim[1],
                            linestyles="-",
                            colors='green',
                            alpha=0.05)

    exp_topk = []
    for t in range(len(exp_list)):
        exp_sorted = sorted(exp_preds[t, :-1], reverse=True)
        exp_topk.append(exp_sorted[topk - 1])

    for i in range(len(exp_names) - 1):
        xx = []
        yy = []
        flag = 0
        for t in range(len(exp_list)):
            if exp_preds[t][i] >= exp_topk[t]:
                yy.append(exp_preds[t][i])
                xx.append(x[t])
            else:
                if flag == 0:
                    plt.plot(xx,
                             yy,
                             label=exp_names[i].split(' ')[0],
                             c=palette[i],
                             linewidth=2)
                    flag = 1
                else:
                    plt.plot(xx, yy, label=None, c=palette[i], linewidth=2)
                yy = []
                xx = []
        if flag == 0:
            plt.plot(xx,
                     yy,
                     label=exp_names[i].split(' ')[0],
                     c=palette[i],
                     linewidth=2)
        else:
            plt.plot(xx, yy, label=None, c=palette[i], linewidth=2)

    if transparent:
        fontdict = {'size': 26, 'color': 'white', 'fontweight': 'bold'}
        legend_font_size = 16
        plt.vlines([x[len(exp_list) - 1]],
                   0,
                   1,
                   linewidth=3,
                   linestyles='dashed',
                   colors='red')
    else:
        fontdict = {'size': 26, 'color': 'black'}
        legend_font_size = 15
        plt.title("Fine-grained Facial Expression Recognition",
                  fontdict=fontdict, pad=15)

    plt.xlim(0, x[-1] + x[-1] // 6)
    plt.xticks(np.arange(0, x[-1] + x[-1] // 6, 20).tolist(),
               fontsize=22,
               color=fontdict['color'])

    plt.ylim(ylim[0], ylim[1])
    plt.yticks(np.arange(ylim[0], ylim[1], 0.2).tolist(),
               fontsize=22,
               color=fontdict['color'])

    plt.xlabel("Time/s", fontdict=fontdict)
    plt.ylabel("Intensity", fontdict=fontdict)
    plt.legend(loc="lower right", fontsize=legend_font_size)

    if transparent:
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['right'].set_visible(False)

        plt.gca().spines['bottom'].set_linewidth(3)
        plt.gca().spines['left'].set_linewidth(3)

        plt.gca().spines['bottom'].set_color('w')
        plt.gca().spines['left'].set_color('w')

    plt.gcf().subplots_adjust(left=0.12, bottom=0.13)
    plt.savefig(savepath, transparent=transparent)
    plt.clf()
    return


def draw_expfig_stack(exp_preds,
                      fps,
                      exp_names,
                      topk=5,
                      savepath='./timing_dragram_stack.png',
                      fig=None):
    sums = np.sum(exp_preds, axis=0)
    sorted_sums = sorted(enumerate(sums), key=lambda x: x[1], reverse=True)
    idxs = [x[0] for x in sorted_sums]

    palette = np.array(sns.color_palette('Paired', 10))
    palette2 = sns.xkcd_palette(
        ['deep blue', 'sea blue', 'pink', 'grey blue', 'sage'])
    palette = np.concatenate([palette, palette2])

    if not fig:
        fig = plt.figure(figsize=(12, 10), dpi=330)

    exp_list = np.argmax(exp_preds, axis=1)
    x = list(range(len(exp_list)))
    x = [a / fps for a in x]
    y = np.max(exp_preds, axis=1)

    labels = []
    colors = []
    yys = []
    for i in idxs[:topk]:
        xx = x
        yy = exp_preds[:, i]
        yys.append(yy)
        labels.append(exp_names[i])
        colors.append(palette[i])

    plt.stackplot(xx, yys, colors=colors, labels=labels, alpha=0.5)
    plt.legend(loc="lower right", fontsize=14)

    plt.xlabel("Time/s", fontdict={'size': 16})
    plt.ylabel("Intensity", fontdict={'size': 16})
    plt.title("Fine-grained Facial Expression Recognition",
              fontdict={'size': 20})

    plt.savefig(savepath)
    plt.clf()
    return


def draw(image,
         exp_final,
         au_names,
         exp_names,
         au_pred,
         exp_pred,
         savefolder=None,
         frame_idx=-1,
         thres=0.4,
         topk=3,
         is_topk=True):

    yd, xd, fsize = int(image.shape[1] / 28), int(image.shape[0] / 26), int(
        image.shape[1] / 45)
    l_yd, r_yd = 0, image.shape[1]
    screen_size = 6 * yd

    color_sep = 0.02
    palette = np.array(sns.color_palette('YlOrRd', int(1 / color_sep) + 1))
    palette *= 255
    palette = np.array(palette, dtype="int32")

    palette_exp = np.array(sns.color_palette('Paired', 10))
    palette_exp2 = sns.xkcd_palette([
        'cloudy blue', 'slate blue', 'yellow brown', 'grey brown', 'dark mauve'
    ])
    palette_exp = np.concatenate([palette_exp, palette_exp2])
    palette_exp *= 255
    palette_exp = np.array(palette_exp, dtype="int32")

    acti_color = (100, 100, 255)[::-1]  # red
    med_color = (255, 255, 255)[::-1]  # white
    deacti_color = (100, 255, 255)[::-1]  # yellow

    # au
    draw_bg(image, l_yd, xd, l_yd + screen_size, xd * (len(au_pred) + 1) + 10)
    str_label = ''
    for i in range(len(au_pred)):
        if au_pred[i] == -1:
            str_label = "None"
            au_pred[i] = 0
        else:
            str_label = str(int(np.round(au_pred[i] * 5, 0))) + '  ' + str(
                np.round(au_pred[i] * 5, 1))
        color = int(au_pred[i] / color_sep)
        if len(au_names[i]) == 3:
            image = cv2ImgAddText(image,
                                  au_names[i] + " : " + str_label,
                                  left=xd * (i + 1),
                                  top=l_yd,
                                  textColor=(palette[color][0],
                                             palette[color][1],
                                             palette[color][2]),
                                  textSize=fsize)
        else:
            image = cv2ImgAddText(image,
                                  au_names[i] + ": " + str_label,
                                  left=xd * (i + 1),
                                  top=l_yd,
                                  textColor=(palette[color][0],
                                             palette[color][1],
                                             palette[color][2]),
                                  textSize=fsize)

    # exp
    draw_bg(image, r_yd - screen_size, xd, r_yd, xd * len(exp_names) + 10)
    if is_topk:
        thres = sorted(exp_pred, reverse=True)[topk - 1]
    for i in range(len(exp_names) - 1):  # exclude neutral
        if frame_idx == -1:
            color = int(exp_pred[i] / color_sep)
            image = cv2ImgAddText(image,
                                  exp_names[i],
                                  left=xd * (i + 1),
                                  top=r_yd - screen_size,
                                  textColor=(palette[color][0],
                                             palette[color][1],
                                             palette[color][2]),
                                  textSize=fsize)
        else:
            if exp_pred[i] >= thres:
                color = palette_exp[i]
                is_bold = True
            else:
                color = med_color
                is_bold = False

            image = cv2ImgAddText(image,
                                  exp_names[i],
                                  left=xd * (i + 1),
                                  top=r_yd - screen_size,
                                  textColor=tuple(color),
                                  textSize=fsize)

    if frame_idx != -1:
        if not savefolder:
            print('savefolder is required for drawing timing diagram.')
            raise ValueError

        foreground = cv2.imread(
            os.path.join(savefolder, 'timing_diagram',
                         str(frame_idx).zfill(4) + '.png'),
            cv2.IMREAD_UNCHANGED)

        h, w = int(foreground.shape[0] / foreground.shape[1] *
                   image.shape[1]), image.shape[1]
        foreground = cv2.resize(foreground, (w, h))

        draw_bg(image, 0,
                xd * (len(au_pred) + 1) + 30, int(image.shape[1]),
                int(image.shape[0]))

        alpha = foreground[:, :, 3]
        alpha[alpha > 0] = 1.0
        foreground = foreground[:, :, :3]
        alpha = np.expand_dims(alpha, axis=2)
        blended = (image[-h:, :, :] * (1 - alpha) + foreground * alpha).astype(
            np.uint8)
        image[-h:, :, :] = blended

        image = cv2ImgAddText(image,
                              "给定阈值: --" if is_topk else "给定阈值: " + str(thres),
                              left=xd * (len(au_pred) + 1) + 30,
                              top=yd * 10 + 10,
                              textColor=med_color if is_topk else acti_color,
                              textSize=fsize - 6)

        image = cv2ImgAddText(image,
                              "/ 强度前" + str(topk) +
                              "位" if is_topk else "/ 强度前 -- 位",
                              left=xd * (len(au_pred) + 1) + 30,
                              top=yd * 10 + screen_size * 2 // 3,
                              textColor=acti_color if is_topk else med_color,
                              textSize=fsize - 6)

        image = cv2ImgAddText(image,
                              '累积表情: ' + exp_names[exp_final].split(' ')[-1],
                              left=xd * (len(au_pred) + 1) + 30,
                              top=r_yd - screen_size,
                              textColor=acti_color,
                              textSize=fsize - 6)
    return image
