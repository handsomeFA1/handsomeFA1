def Markov_forecast(price_data:pd.DataFrame, conclude_P = True):
    # 由于时间跨度过长，为了避免通货膨胀对预测结果发生不利影响
    # 我们采用对数单日收益率进行预测
    log_rate = np.log(price_data.shift(1) / price_data).dropna()
    a = log_rate.describe()
    # 以四分位点作为分界点（四个状态）
    # 根据收益率不可能大于1的特性，做出如下处理
    states = log_rate# 创立副本，防止数据标记广播到原数据
    states[(log_rate>=a.iloc[6])]=4
    states[((log_rate>=a.iloc[5]) & (log_rate<a.iloc[6]))]=3
    states[((log_rate>=a.iloc[4]) & (log_rate<a.iloc[5]))]=2
    states[(log_rate<a.iloc[4])]=1
    # 状态标签完毕
    # 转移概率矩阵计算
    P_matrix = np.zeros((4, 4))
    array =np.array(states)
    if conclude_P == True:# 有需要才会计算
        for j in range(4):
            # 四行，每行单独计算
            for i in range(len(array)-1):
                if (array[i] == j+1 and array[i+1] == 1):
                    P_matrix[j, 0] += 1
                elif (array[i] == j+1 and array[i+1] == 2):
                    P_matrix[j, 1] += 1
                elif (array[i] == j+1 and array[i+1] == 3):
                    P_matrix[j, 2] += 1
                elif (array[i] == j+1 and array[i+1] == 4):
                    P_matrix[j, 3] += 1
            P_matrix[j: ] = P_matrix[j: ]/np.sum(P_matrix[j: ])
    return P_matrix, states
