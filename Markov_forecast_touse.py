import pandas as pd
import numpy as np
##################
data = pd.read_csv('---.csv')
data_tarin = data.iloc[:-10]# 预留出最后十天用于检验
price_datas = ['open', 'high', 'low', 'close']
P_all = {}
states_all = {}
P_all_round4 = {}
# 创建两个字典存储状态标签与转移概率矩阵
for i in range(4):
    P_all[price_datas[i]], states_all[price_datas[i]] = Markov_forecast(data_tarin[price_datas[i]])
    P_all_round4[price_datas[i]] = P_all[price_datas[i]].round(4)
P_all_round4
# 字典存储的转移概率矩阵
# 可以看到四组数据的一步转移概率矩阵并不完全相同
### 注意：转移概率矩阵在此时为了观感舒适而保留四位小数，但后续处理必须严格沿用计算得出的直接结果
### 否则会造成马氏链达成平稳的时间极其长，结果也不再准确
###################
states_all
# 状态标签也并不一致
###################
# 因此我们得出四套数据的流动方向，状态是一致的,取其中一个研究即可
stabilizationperiod = {}
limiting_distribution = {}
for j in range(4):
    for i in range(10000):
        P_matrix = P_all[price_datas[j]]
        if np.allclose(np.linalg.matrix_power(P_matrix, i), np.linalg.matrix_power(P_matrix, i+1)) == True:
            # 保留一定的误差，完全相同则会导致则天数过多
            break
        # 用Python计算的时候，如果用scipy.linalg里面的solve，会有问题，因为这个方法要求传入的是一个方阵
        # 可是这里有五个方程，四个未知数，无法求出。
        # Return the least-squares solution to a linear matrix equation.
        # 用最小二乘法来求解，P_matrix是系数矩阵，b是结果
        P_matrix = P_matrix-np.eye(4)
        P_matrix = np.append(P_matrix,np.array([1,1,1,1])).reshape(-1,4)
        junyun = np.linalg.lstsq(P_matrix,np.array([0,0,0,0,1]))# 最终股票的分布
    limiting_distribution[price_datas[j]] = junyun[0]
    stabilizationperiod[price_datas[j]] = i
stabilizationperiod
# 达成平稳所需天数
# 如果采用截取四位小数的转移矩阵进行计算会导致达成平稳的天数超出10000天，误差也会变得很大
####################
limiting_distribution
# 平稳分布
####################
# 可以看到虽然趋于平稳的时间长度并不完全相同，但是平稳分布是一致的
# 我们下面利用10期数据检测预测精度
data_test = data.iloc[-10:]
_, data_test_states = Markov_forecast(data_test, conclude_P=False)
# 不需要计算转移概率矩阵
correct_count = {}
false_count = {}
for j in range(4):
    # 初值定义
    correct_count[price_datas[j]] = 0
    false_count[price_datas[j]] = 0
    for date in range(1, len(data_test_states)):
        states = np.array(data_test_states[price_datas[j]])
        state_list = np.array((1, 2, 3, 4))
        P_matrix = P_all[price_datas[j]]
        P_daily = np.linalg.matrix_power(P_matrix, date)
        # n步转移概率矩阵等于一步转移概率矩阵^n
        predicted_state = state_list[P_daily[:,j]==np.max(P_daily[:,j])][0]
        # 找到最大值的位置， 并且选取出对应的状态
        # 在某些状态下，该状态转向其他多个状态的概率是等价的，这里放宽要求，只要实际的状态
        # 在预测的可能状态其中之一即可
        if states[date] == predicted_state:
            # 如果成功预测，正确计数+1
            correct_count[price_datas[j]] += 1
        else:
            false_count[price_datas[j]] += 1
###################
correct_count
false_count
