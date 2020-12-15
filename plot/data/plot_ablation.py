import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def moving_average(data_set, periods=200):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='same')
    # return np.convolve(data_set, weights, mode='full')


def cut2sameLen(arr):
    min_len = np.inf
    for each in arr:
        if len(each) < min_len:
            min_len = len(each)
    for i in range(len(arr)):
        arr[i] = moving_average(arr[i][:min_len])
    return np.array(arr)


folder = './data/ablation/'

# target = 'LunarLanderContinuous-v2'
# target = 'InvertedDoublePendulum-v1'
# target = 'HalfCheetah-v1'
# target = 'Walker2d-v1'

content = ['vdfp_icnn', 'vdfp_neicnn']
# content = ['vdfp_neicnn']

LENGTH = 1000
AVERAGE_LENGTH = 10

colors = ['b', 'g', 'r', 'orange', 'purple']

style = ['-', '-', '-', '-', '-', '-']

legends = ['VDFP', 'DDPG', 'DDSR', 'PPO', 'A2C']

# plt.figure(figsize=[9, 4.8])
# plt.figure(figsize=[8, 4.8])
plt.figure(figsize=[6.4, 4.8])
ax = plt.subplot(111)

count = -1
for c in content:
    data = []
    count += 1
    for i in range(1,10):
        try:
            temp_path = folder + '/' + c + '/run_' + c + '_' + str(i)
            # print(temp_path)
            temp = []
            with open(temp_path + '.csv', 'r') as myFile:
                lines = csv.reader(myFile)
                for line in lines:
                    temp.append([line[1], line[2]])
                # if c in ['ddpg', 'ddsr']:
                #     data.append(temp[1:101])
                # else:
                #     data.append(temp[1:])
                data.append(temp[1:])
        except:
            continue
    data = np.array(data).reshape((5, -1, 2)).astype(float)

    x = data[0, :, 0] / 1000000
    values = data[:, :, 1]

    # if c in ['ddpg', 'ddsr']:
    #     # values_smoothed = values
    #     values_smoothed = []
    #     smooth_period = 10
    #     for v in values:
    #         # smooth_period = 200
    #         v_list = v.tolist()
    #         v_padded = v_list + [v_list[-1]] * smooth_period
    #         tmp_smoothed = moving_average(np.array(v_padded), periods=smooth_period)
    #         clipped_smooth = tmp_smoothed[:-smooth_period]
    #         values_smoothed.append(clipped_smooth)
    # else:
    #     values_smoothed = []
    #     for v in values:
    #         smooth_period = 100
    #         # smooth_period = 200
    #         v_list = v.tolist()
    #         v_padded = v_list + [v_list[-1]] * smooth_period
    #         tmp_smoothed = moving_average(np.array(v_padded), periods=smooth_period)
    #         clipped_smooth = tmp_smoothed[:-smooth_period]
    #         values_smoothed.append(clipped_smooth)

    values_smoothed = []
    for v in values:
        smooth_period = 100
        # smooth_period = 200
        v_list = v.tolist()
        v_padded = v_list + [v_list[-1]] * smooth_period
        tmp_smoothed = moving_average(np.array(v_padded), periods=smooth_period)
        clipped_smooth = tmp_smoothed[:-smooth_period]
        values_smoothed.append(clipped_smooth)

    mean = np.mean(values_smoothed, axis=0)
    # std = np.std(values_smoothed, axis=0)
    std = np.std(values_smoothed, axis=0) / 2

    # FIXME 20190830
    x_1k = np.where(x <= 1.0001)
    max_avg = np.max(mean[x_1k[0]])
    max_index = mean.tolist().index(max_avg)
    print('\'', c, '\':', max_avg, '+-', std[max_index])

    ax.plot(x, mean, style[count], label=legends[count], color=colors[count], alpha=0.6, linewidth=2)
    ax.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.12, color=colors[count])


# plt.ylim(-500, 6500)
# plt.ylim(-250, 2500)

plt.ylim(-500, 6200)
# plt.ylim(-200, 10000)
# plt.ylim(-300, 250)

plt.xlim(0, 1)
# plt.xlim(0, 2)
plt.xlabel('Time Steps (1e6)', fontsize='14')
plt.ylabel('Average Episode Reward', fontsize='14')
# plt.legend(loc='lower right', fontsize='12')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8/0.9, box.height])
# ax.legend(loc='upper left', fontsize='16', bbox_to_anchor=(1.0, 1.0))
# plt.legend(loc='upper left', fontsize='16')

# plt.savefig('Evaluation-' + target + '-64x48.pdf')
# plt.savefig('Evaluation-' + target + '8x48.pdf')
# plt.savefig('Evaluation-' + target + '.pdf')


plt.show()
