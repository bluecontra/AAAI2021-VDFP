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


folder = './data_for_vdfp_ppo/'

# target = 'HalfCheetah-v1'
# target = 'Walker2d-v1'
# target = 'Ant-v1'
target = 'Hopper-v1'

content = ['vdppo', 'ppo']

LENGTH = 1000
AVERAGE_LENGTH = 10

colors = ['b', 'r']

style = ['-', '-', '-', '-', '-', '-']

legends = ['VD-PPO', 'PPO']

ax = plt.subplot(111)

count = -1
for c in content:
    data = []
    count += 1

    num = 0
    for i in range(1,7):
        temp_path = folder + target + '/' + c + '/run_' + c + '_' + str(i)
        temp = []
        try:
            with open(temp_path + '.csv', 'r') as myFile:
                lines = csv.reader(myFile)
                for line in lines:
                    temp.append([line[1], line[2]])
                data.append(temp[1:])
                num += 1
        except:
            print(temp_path + '.csv not found.')

    data = np.array(data).reshape((num, -1, 2)).astype(float)

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

    x = x.tolist()
    x_index = None
    for idx, v in enumerate(x):
        if v >= 1.5:
            x_index = idx
            break

    max = np.max(mean[:x_index + 1])
    max_index = mean.tolist().index(max)
    print('Algorithm:', c, 'mean:', mean[x_index], 'half_std:', std[x_index], 'max:', max, 'half_std:', std[max_index])



plt.xlim(0, 1.8)
plt.xlabel('Time Steps (1e6)', fontsize='14')
plt.ylabel('Avg Episode Reward' + '-' + target, fontsize='16')
plt.legend(loc='upper left', fontsize='18')

# plt.savefig('VDPPO-Evaluation_' + target + '.pdf')


plt.show()
