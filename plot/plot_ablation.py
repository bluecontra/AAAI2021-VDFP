import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def moving_average(data_set, periods=200):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')


def cut2sameLen(arr):
    min_len = np.inf
    for each in arr:
        if len(each) < min_len:
            min_len = len(each)
    for i in range(len(arr)):
        arr[i] = moving_average(arr[i][:min_len])
    return np.array(arr)


folder = './data/ablations/'

target = 'HalfCheetah-v1'
# target = 'Walker2d-v1'

content = ['vdfp', 'vdfp_icnn', 'vdfp_neicnn']
# content = ['vdfp', 'vdfp_mlp']
# content = ['vdfp', 'vdfp_lstm']
# content = ['vdfp', 'vdfp_concat']
# content = ['vdfp', 'vdfp_lrelu02', 'vdfp_lrelu05', 'vdfp_lrelu08']

# content = ['vdfp', 'vdfp_in0', 'vdfp_in1', 'vdfp_in_inf']
# content = ['vdfp', 'vdfp_kl1e1', 'vdfp_kl1', 'vdfp_kl10', 'vdfp_kl100']


LENGTH = 1000
AVERAGE_LENGTH = 10

colors = ['b', 'r', 'orange','purple',]
# colors = ['b', 'orange','purple',]
# colors = ['b', 'g', 'r', 'orange','purple',]
# colors = ['b', 'purple', 'k', 'dimgrey']
# colors = ['b', 'g', 'r', 'orange', 'purple', 'dimgrey', 'k', 'violet', 'pink']


style = ['-', '-', '-', '-', '-', '-', '-', '-', '-']

legends = ['VDFP','VDFP_ICNN', 'VDFP_NEICNN']
# legends = ['VDFP', 'VDFP_MLP']
# legends = ['VDFP', 'VDFP_LSTM']
# legends = ['VDFP', 'VDFP_CONCAT']
# legends = ['VDFP-linear', 'VDFP-alpha=0.2', 'VDFP-alpha=0.5', 'VDFP-alpha=0.8']

# legends = ['VDFP', 'VDFP_c=0', 'VDFP_c=1', 'VDFP_c=inf.']
# legends = ['VDFP', 'VDFP_kl=1e1', 'VDFP_kl=1', 'VDFP_kl=10', 'VDFP_kl=100']


# plt.figure(figsize=[10.0, 4.8])
ax = plt.subplot(111)

count = -1
for c in content:
    data = []
    count += 1
    for i in range(1,6):
        temp_path = folder + target + '/' + c + '/run_' + c + '_' + str(i)
        # print(temp_path)
        temp = []
        with open(temp_path + '.csv', 'r') as myFile:
            lines = csv.reader(myFile)
            for line in lines:
                temp.append([line[1], line[2]])
            data.append(temp[1:])
    data = np.array(data).reshape((5, -1, 2)).astype(float)

    x = data[0, :, 0] / 1000000
    values = data[:, :, 1]

    values_smoothed = []
    for v in values:
        smooth_period = 100
        v_list = v.tolist()
        v_padded = v_list + [v_list[-1]] * smooth_period
        tmp_smoothed = moving_average(np.array(v_padded), periods=smooth_period)
        clipped_smooth = tmp_smoothed[:-smooth_period]
        values_smoothed.append(clipped_smooth)

    mean = np.mean(values_smoothed, axis=0)
    std = np.std(values_smoothed, axis=0) / 2

    ax.plot(x, mean, style[count], label=legends[count], color=colors[count], alpha=0.5, linewidth=2)
    ax.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.12, color=colors[count])

    x_list = x.tolist()
    x_index = x_list.index(1.0)

    max = np.max(mean[:x_index + 1])
    max_index = mean.tolist().index(max)
    print('Algorithm:', c, 'mean:', mean[x_index], 'half_std:', std[x_index], 'max:', max, 'half_std:', std[max_index])


plt.xlim(0, 1)
# plt.xlim(0, 2)
plt.xlabel('Time Steps (1e6)', fontsize='14')
plt.ylabel('Average Return', fontsize='14')
# plt.legend(loc='lower right', fontsize='12')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.64, box.height])
# ax.legend(loc='upper left', fontsize='14', bbox_to_anchor=(1.0, 0.5))
plt.legend(loc='upper left', fontsize='14')

# plt.savefig('Ablation-vae_vs_mlp-' + target + '.pdf')
# plt.savefig('Ablation-cnn_vs_lstm-' + target + '.pdf')
# plt.savefig('Ablation-pp_vs_concat-' + target + '.pdf')
# plt.savefig('Ablation-kl-' + target + '.pdf')
# plt.savefig('Ablation-clip_noise-' + target + '.pdf')
# plt.savefig('Ablation-linear_vs_convex-' + target + '.pdf')
plt.savefig('Ablation-linear_vs_ICNN-' + target + '.pdf')


plt.show()
