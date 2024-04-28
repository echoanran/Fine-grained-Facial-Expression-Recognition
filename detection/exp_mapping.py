import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


class ExpMapping():
    def __init__(self, selected_exps, language='bi'):
        if language == 'bi':
            self.neutral_name = 'peaceful       中性'
            # region arguments yapf: disable
            self.exp_names = {
                0: 'happy          愉快',
                1: 'sad            难过',
                2: 'fearful        恐惧',
                3: 'angry          生气',
                4: 'surprised      惊讶',
                5: 'disgusted      厌恶',
                6: 'contempt       轻蔑',
                23:'belligerent    好战',
                24:'domineering    专横',
                26:'threatening    威胁',
                27:'whined         哀诉',
                28:'awed           敬畏',
                29:'hated          憎恶',
                30:'painful        痛苦',
                31:'peaceful       中性',
            }
            # endregion yapf: enable
        else:
            self.neutral_name = 'peaceful'
            # region arguments yapf: disable
            self.exp_names = {
                0: 'happy',
                1: 'sad',
                2: 'fearful',
                3: 'angry',
                4: 'surprised',
                5: 'disgusted',
                6: 'contempt',

                7: 'happily surprised',
                8: 'happily disgusted',
                9: 'sadly fearful',
                10: 'sadly angry',
                11: 'sadly surprised',
                12: 'sadly disgusted',
                13: 'fearfully angry',
                14: 'fearfully surprised',
                15: 'fearfully disgusted',
                16: 'angrily surprised',
                17: 'angrily disgusted',
                18: 'disgustedly surprised',

                19: 'enthusiastic',
                20: 'humorous',
                21: 'interested',
                22: 'affirmative',
                23: 'belligerent',
                24: 'domineering',
                25: 'defensive',
                26: 'threatened',
                27: 'whined',
                28: 'awed',
                29: 'hated',
                30: 'painful',
                31: 'peaceful',
            }
            # endregion yapf: enable

        # region arguments yapf: disable
        self.ausets = {
                    0: [6, 12, 25],  # happy
                    1: [1, 4, 6, 15, 17],  # sad
                    2: [1, 2, 4, 5, 20, 25, 26],  # fearful
                    3: [4, 10, 17],  # angry
                    4: [1, 2, 5, 25, 26],  # surprised
                    5: [4, 9, 10, 17],  # disgusted
                    6: [14],  # contempt

                    7: [1, 2, 5, 12, 25, 26],  # happily surprised
                    8: [4, 6, 9, 10, 12, 25],  # happily disgusted
                    9: [1, 2, 4, 5, 6, 15, 20, 25],  # sadly fearful
                    10: [4, 6, 15, 17],  # sadly angry
                    11: [1, 2, 4, 6, 25, 26],  # sadly surprised
                    12: [1, 4, 6, 9, 10, 15, 17, 25],  # sadly disgusted
                    13: [4, 5, 10, 20, 25],  # fearfully angry
                    14: [1, 2, 4, 5, 10, 20, 25, 26],  # fearfully surprised
                    15: [1, 2, 4, 5, 6, 9, 10, 15, 20, 25],  # fearfully disgusted
                    16: [4, 5, 10, 25, 26],  # angrily surprised
                    17: [4, 9, 10, 17],  # angrily disgusted
                    18: [1, 2, 4, 5, 9, 10, 17],  # disgustedly surprised

                    19: [],  # enthusiastic
                    20: [],  # humorous
                    21: [],  # interested
                    22: [],  # affirmative
                    23: [1, 2],  # belligerent
                    24: [2],  # domineering
                    25: [],  # defensive
                    26: [1, 2, 5],  # threatened
                    27: [1, 2, 15],  # whined
                    28: [1, 2, 4, 5, 20, 25, 26],  # awed
                    29: [4, 10, 9, 17],  # hated
                    30: [4, 6, 9, 43],  # painful
                    31: [],  # peaceful
        }

        self.weights = {
                    0: [0.4, 0.5, 0.1],  # happy
                    1: [0.1, 0.1, 0.1, 0.4, 0.3],  # sad
                    2: [0.05, 0.05, 0.1, 0.5, 0.3, 0.05, 0.05],  # fearful
                    3: [0.5, 0.2, 0.3],  # angry
                    4: [0.2, 0.2, 0.4, 0.1, 0.1],  # surprised
                    5: [0.3, 0.5, 0.1, 0.1],  # disgusted   
                    6: [0.5],  # contempt

                    7: [],  # happily surprised
                    8: [],  # happily disgusted
                    9: [],  # sadly fearful
                    10: [],  # sadly angry
                    11: [],  # sadly surprised
                    12: [],  # sadly disgusted
                    13: [],  # fearfully angry
                    14: [],  # fearfully surprised
                    15: [],  # fearfully disgusted
                    16: [],  # angrily surprised
                    17: [],  # angrily disgusted
                    18: [],  # disgustedly surprised

                    19: [],  # enthusiastic
                    20: [],  # humorous
                    21: [],  # interested
                    22: [],  # affirmative
                    23: [0.2, 0.2],  # belligerent
                    24: [0.5],  # domineering
                    25: [],  # defensive
                    26: [0.1, 0.1, 0.3],  # threatened
                    27: [0.1, 0.1, 0.8],  # whined
                    28: [0.1, 0.1, 0.4, 0.4],  # awed
                    29: [0.4, 0.35, 0.1, 0.1, 0.05],  # hated
                    30: [0.1, 0.1, 0.2, 0.2],  # painful
                    31: [],  # peaceful
        }
        # endregion yapf: disable

        # region arguments yapf: disable
        # [1, 2, 4, 5, 6,  9, 10, 12, 14, 15,  17, 20, 25, 26, 43]
        self.exp_codes = {
                    0: [0, 0, 0, 0, 0.51,    0, 0, 1, 0, 0,     0, 0, 1, 0, 0],  # happy
                    1: [0.60, 0, 1, 0, 0.50,    0, 0, 0, 0, 1,     0.67, 0, 0, 0, 0],  # sad
                    2: [1, 0.57, 1, 0.63, 0,    0, 0, 0, 0, 0,     0, 1, 1, 0.33, 0],  # fearful
                    3: [0, 0, 1, 0, 0,    0, 0.26, 0, 0, 0,     0.52, 0, 0, 0, 0],  # angry
                    4: [1, 1, 0, 0.66, 0,    0, 0, 0, 0, 0,     0, 0, 1, 1, 0],  # surprised
                    5: [0, 0, 0.31, 0, 0,    1, 1, 0, 0, 0,     1, 0, 0, 0, 0],  # disgusted
                    6: [0, 0, 0, 0, 0,    0, 0, 0, 1, 0,     0, 0, 0, 0, 0],  # contempt

                    7: [1, 1, 0, 0.64, 0,    0, 0, 1, 0, 0,    0, 0, 1, 0.67, 0],  # happily surprised
                    8: [0, 0, 0.32, 0, 0.61,    0.59, 1, 1, 0, 0,    0, 0, 1, 0, 0],  # happily disgusted
                    9: [1, 0.46, 1, 0.24, 0.34,    0, 0, 0, 0, 0.30,    0, 1, 1, 0, 0],  # sadly fearful
                    10: [0, 0, 1, 0, 0.26,    0, 0, 0, 0, 1,    0.50, 0, 0, 0, 0],  # sadly angry
                    11: [1, 0.27, 1, 0, 0.31,    0, 0, 0, 0, 0,    0, 0, 1, 1, 0],  # sadly surprised
                    12: [0.49, 0, 1, 0, 0.61,    0.20, 1, 0, 0, 0.54,    0.47, 0, 0.43, 0, 0],  # sadly disgusted
                    13: [0, 0, 1, 0.40, 0,    0, 0.30, 0, 0, 0,    0, 1, 1, 0, 0],  # fearfully angry
                    14: [1, 1, 0.47, 1, 0,    0, 0.35, 0, 0, 0,    0, 1, 1, 0.51, 0],  # fearfully surprised
                    15: [1, 0.64, 1, 0.50, 0.26,    0.28, 1, 0, 0, 0.33,    0, 1, 1, 0, 0],  # fearfully disgusted
                    16: [0, 0, 1, 0.35, 0,    0, 0.34, 0, 0, 0,    0, 0, 1, 1, 0],  # angrily surprised
                    17: [0, 0, 1, 0, 0,    0.57, 1, 0, 0, 0,    1, 0, 0, 0, 0],  # angrily disgusted
                    18: [1, 1, 0.45, 1, 0,    0.37, 1, 0, 0, 0,    0.66, 0, 0, 0, 0],  # disgustedly surprised

                    # 19: [1, 1, 0, 1, 1,    0, 0, 1, 0, 0,    0, 0, 1, 1, 0],  # enthusiastic
                    # 20: [1, 1, 0, 0, 1,    0, 0, 1, 0, 0,    0, 0, 1, 1, 0],  # humorous
                    # 21: [1, 1, 0, 0, 1,    0, 0, 1, 0, 0,    0, 0, 0, 0, 0],  # interested (leaning forward)
                    # 22: [1, 1, 0, 0, 1,    0, 0, 1, 0, 0,    0, 0, 0, 0, 0],  # affirmative (head nod)
                    # 23: [1, 1, 0, 0, 0,    0, 0, 0, 0, 0,    1, 0, 0, 0, 0],  # belligerent (jaw forward)
                    # 24: [0, 1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # domineering (head forward, head cocked to one side)
                    # 25: [1, 1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # defensive (arms folded across chest)
                    # 26: [0.6, 0.4, 0, 1, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # threatened (head forward, head cocked to one side)
                    # 27: [1, 0.2, 0, 0, 0,    0, 0, 0, 0, 1,    0, 0, 0, 0, 0],  # whined
                    # 28: [1, 1, 0.21, 1, 0,    0, 0, 0, 0, 0,    0, 0.62, 1, 0.56, 0],  # awed
                    # 29: [0, 0, 1, 0, 0,    0.27, 1, 0, 0, 0,    0.63, 0, 0, 0, 0],  # hated
                    # 30: [0, 0, 0.5, 0, 0.2,    1, 0, 0, 0, 0,    0, 0, 0, 0, 0.2],  # painful
                    # 31: [0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # peaceful

                    19: [1, 1, 0, 1, 1,    0, 0, 1, 0, 0,    0, 0, 1, 1, 0],  # enthusiastic   
                    20: [1, 1, 0, 0, 1,    0, 0, 1, 0, 0,    0, 0, 1, 1, 0],  # humorous
                    21: [1, 1, 0, 0, 1,    0, 0, 1, 0, 0,    0, 0, 0, 0, 0],  # interested (leaning forward)
                    22: [1, 1, 0, 0, 1,    0, 0, 1, 0, 0,    0, 0, 0, 0, 0],  # affirmative (head nod)
                    23: [0.6, 0.4, 0, 0, 0,    0, 0, 0, 0, 0,    1, 0, 0, 0, 0],  # belligerent (jaw forward)
                    24: [0, 1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # domineering (head forward, head cocked to one side)
                    25: [1, 1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # defensive (arms folded across chest)
                    26: [0.6, 0.4, 0, 1, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # threatened (head forward, head cocked to one side)
                    27: [0.6, 0.2, 0, 0, 0,    0, 0, 0, 0, 1,    0, 0, 0, 0, 0],  # whined
                    28: [1, 1, 0.21, 1, 0,    0, 0, 0, 0, 0,    0, 0.62, 1, 0.56, 0],  # awed
                    29: [0, 0, 1, 0, 0,    0.27, 1, 0, 0, 0,    0.63, 0, 0, 0, 0],  # hated 
                    30: [0, 0, 0.3, 0, 0.2,    0.3, 0, 0, 0, 0,    0, 0, 0, 0, 0.2],  # painful
                    31: [0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 0],  # peaceful
        }
        # endregion yapf: disable

        # region arguments yapf: disable
        self.tree_nodes = {
            0: [1, 1],  # happy 
            1: [-1, 0],  # sad 
            2: [-1, 1],  # fearful 
            3: [-1, 1],  # angry 
            4: [0, 1],  # surprised 
            5: [0, 1],  # disgusted 
            6: [0, 0],  # contempt 

            7: [1, 1],  # happily surprised 
            8: [1, 1],  # happily disgusted 
            9: [-1, 1],  # sadly fearful 
            10: [-1, 1],  # sadly angry 
            11: [-1, 1],  # sadly surprised 
            12: [-1, 1],  # sadly disgusted 
            13: [-1, 1],  # fearfully angry 
            14: [-1, 1],  # fearfully surprised 
            15: [-1, 1],  # fearfully disgusted 
            16: [-1, 1],  # angrily surprised 
            17: [-1, 1],  # angrily disgusted 
            18: [0, 1],  # disgustedly surprised 

            19: [1, 1],  # enthusiastic 
            20: [1, 0],  # humorous 
            21: [1, 0],  # interested 
            22: [1, 0],  # affirmative 
            23: [-1, 1],  # belligerent 
            24: [-1, 1],  # domineering 
            25: [-1, 0],  # defensive 
            26: [-1, 0],  # threatened 
            27: [-1, 0],  # whined 
            28: [0, 1],  # awed 
            29: [-1, 1],  # hated 
            30: [-1, 1],  # painful 
            31: [0, -1],  # peaceful 
        }
        # endregion yapf: disable

        self.selected_exps = selected_exps

        self.selected_ausets = []
        self.selected_weights = []
        self.selected_exp_codes = []
        self.selected_exp_names = []
        self.selected_tree_nodes = []
        for k in self.selected_exps:
            self.selected_ausets.append(self.ausets[k])
            self.selected_weights.append(self.weights[k])
            self.selected_exp_codes.append(self.exp_codes[k])
            self.selected_exp_names.append(self.exp_names[k])
            self.selected_tree_nodes.append(self.tree_nodes[k])

        self.selected_exp_codes = np.array(self.selected_exp_codes)

    def get_expnames(self):
        return self.selected_exp_names + [self.neutral_name]

    def get_hamming_distance(self, input, targets):
        distances = []
        for target in targets:
            distances.append(sum([abs(a - b) for a, b in zip(input, target)]))
        distances = np.array(distances)
        return distances

    def get_exp_prob(self, au_pred, method='decision_tree'):
        if method == 'hamming_distance':
            distances = self.get_hamming_distance(au_pred, self.selected_exp_codes)
            exp_prob = np.exp(-distances) / sum(np.exp(-distances))
        elif method == 'relation_depression':
            exp_prob = np.ones(len(self.selected_exp_names))
            for i in range(len(self.selected_exp_names)):
                if self.selected_tree_nodes[i][0] == 1:
                    exp_prob[i] *= 1 - au_pred[9] / 2
                elif self.selected_tree_nodes[i][0] == -1:
                    exp_prob[i] *= 1 - au_pred[7] / 2

                if 'domineering' in self.selected_exp_names[i]:
                    exp_prob[i] *= 1 - au_pred[0]
                if 'contempt' in self.selected_exp_names[i]:
                    exp_prob[i] *= 1 - max(au_pred)

        else:
            exp_prob = np.ones(len(self.selected_exp_names))

        return exp_prob

    def global_smoothing(self, exp_preds):
        exp_preds_global = []
        for exp_pred in exp_preds:
            if not exp_preds_global:
                exp_preds_global.append(exp_pred)
            else:
                exp_update = np.array(exp_preds_global[-1]) * 0.7 + 0.3 * exp_pred
                exp_preds_global.append(exp_update.tolist())

        return np.array(exp_preds_global)

    def get_neutral_thres(self, exp_preds, savepath='./save/kde.png'):
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 26,
        }

        palette = np.array(sns.color_palette('Paired', 10))
        palette2 = sns.xkcd_palette([
            'cloudy blue', 'slate blue', 'yellow brown', 'grey brown', 'dark mauve'
        ])
        palette = np.concatenate([palette, palette2])

        print(exp_preds[:, :-1].flatten())
        sns.kdeplot(exp_preds[:, :-1].flatten(), shade=True, color='#7BB7D8', alpha=0.6)
        mean = np.percentile(exp_preds[:, :-1].flatten(), 90)
        plt.vlines(mean, 0, 4, color='black', linestyles='--')
        plt.ylim(0, 4)
        plt.xlabel('Intensity', fontdict=font)
        plt.ylabel('Density', fontdict=font)
        plt.gcf().subplots_adjust(left=0.16, bottom=0.18)

        plt.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelsize=18,
        )
        plt.savefig(savepath)
        plt.clf()
        return mean
    
    def get_cumulated_exp(self, exp_preds, savepath='./save/kde.png', fig=None):
        
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 26,
        }

        palette = np.array(sns.color_palette('Paired', 10))
        palette2 = sns.xkcd_palette([
            'cloudy blue', 'slate blue', 'yellow brown', 'grey brown', 'dark mauve'
        ])
        palette = np.concatenate([palette, palette2])

        for c in range(exp_preds.shape[1]):
            sns.kdeplot(exp_preds[:, c].flatten(), shade=True, color=palette[c], alpha=0.4)
            mean = np.percentile(exp_preds[:, c].flatten(), 50)
            plt.vlines(mean, 0, 100, color=palette[c], linestyles='--')
            plt.ylim(0, 100)
            plt.xlabel('Intensity', fontdict=font)
            plt.ylabel('Density', fontdict=font)
            plt.gcf().subplots_adjust(left=0.16, bottom=0.18)

            plt.tick_params(
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelsize=18,
            )

            plt.savefig(savepath.replace('kde', 'kde_' + self.get_expnames()[c].split(' ')[0]))
      
        return

    def automatic_analysis(self, exp_preds):
        exp_preds = exp_preds[:, :-1]
        T = len(exp_preds)

        exp_idx_tops = []
        exp_value_tops = []
        for i in range(len(exp_preds)):
            sorted_all = sorted(enumerate(exp_preds[i]), key=lambda x: x[1], reverse=True)
            sorted_idxs = [x[0] for x in sorted_all]
            sorted_values = [x[1] for x in sorted_all]
            exp_idx_tops.append(sorted_idxs[:3])
            exp_value_tops.append(sorted_values[:3])
        exp_idx_tops = np.array(exp_idx_tops)
        exp_value_tops = np.array(exp_value_tops)

        cnt1, cnt2, cnt3 = 0, 0, 0
        # delta1, delta3= 0.14, 0.2
        # K = 5
        # T1, T2, T3 = 15, 15, 300
        # I1, I2, I3 = 0.3, 0.3, 0.25
        delta1, delta3= 0.14, 0.1
        K = 5
        T1, T2, T3 = 15, 30, 300
        I1, I2, I3 = 0.25, 0.35, 0.25
        cur_first, cur_second = -1, -1
        label1, label2, label3 = np.ones(T) * -1, np.ones((2, T)) * -1, np.ones(T) * -1
        result1, result2, result3 = np.ones(T) * -1, np.ones(T) * -1, np.ones(T) * -1
        
        pre_sums = [exp_value_tops[0, 0]]
        for i in range(1, len(exp_preds)):
            pre_sums.append(pre_sums[-1] + exp_value_tops[i, 0])

        for i in range(len(exp_preds)):
            for j in range(i + 50, min(i + T3, len(exp_preds))):
                cnt3 = len(set(exp_idx_tops[i: j, 0]))
                values = exp_value_tops[i: j, 0]
                if cnt3 >= K and max(values) - min(values) <= delta3 and np.mean(values) > I3:
                    label3[i: j] = 1

        for i in range(len(exp_preds)):
            if i == 0 or exp_value_tops[i, 0] < I1 or exp_value_tops[i, 0] - exp_value_tops[i, 1] < delta1 or exp_idx_tops[i, 0] != exp_idx_tops[i - 1, 0]:
                if cnt1 >= T1:
                    label1[i - cnt1: i] = cur_first
                cur_first = exp_idx_tops[i, 0]
                cnt1 = 1
            else:
                cnt1 += 1

            if i == 0 or exp_value_tops[i, 1] < I2 \
                or (set(exp_idx_tops[i, 0:2]) != set(exp_idx_tops[i - 1, 0:2])):
                if cnt2 >= T2:
                    label2[0, i - cnt2: i] = cur_first
                    label2[1, i - cnt2: i] = cur_second
                cur_first, cur_second = exp_idx_tops[i, 0], exp_idx_tops[i, 1]
                cnt2 = 1
            else:
                cnt2 += 1

        result1[label1 > -1] = 1
        result2[label2[0] > -1] = 1
        result3[label3 > - 1] = 1

        return [result1, result2, result3]

    def forward(self, au_pred, neutral_thres=0):

        exp_pred = []
        for exp_code in self.selected_exp_codes:
            exp_pred.append(sum(au_pred * exp_code) / sum(exp_code))

        exp_prob = self.get_exp_prob(au_pred=au_pred, method='relation_depression')
        # exp_prob = self.get_exp_prob(au_pred=au_pred, method='None')
        exp_pred = exp_pred * exp_prob

        exp_pred = np.clip(exp_pred, 0, 1).tolist()

        if np.max(exp_pred) < neutral_thres:
            exp_pred.append(neutral_thres)
        else:
            exp_pred.append(0)

        exp_pred = np.array(exp_pred)
        return exp_pred, self.get_expnames()
