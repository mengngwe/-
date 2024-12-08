import os
import copy
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import arch
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

#==============================================================================

# 读取数据，adf检验
sp500 = pd.read_csv('SP500_19900102_20221231.csv')[['caldt', 'vwretd']].set_index('caldt')
result = adfuller(sp500['vwretd'])
# 输出ADF检验结果
print("ADF统计量:", result[0])
print("P值:", result[1])
print("滞后阶数:", result[2])
print("观测数:", result[3])
print("临界值:", result[4])
print("最大信息准则(AIC):", result[5])
if result[1] <= 0.05:
    print("P值小于或等于0.05，可以拒绝原假设，数据是平稳的。")
else:
    print("P值大于0.05，无法拒绝原假设，数据可能是非平稳的。")

#==============================================================================

# 训练与预测arima模型
def predict_arima(series):
    scale_mean = series.values.mean()
    scale_std = series.values.std()
    series_scaled = (series - scale_mean) / scale_std
    # 临时禁用FutureWarning警告
    warnings.simplefilter(action='ignore', category=FutureWarning)
    model = ARIMA(series_scaled, order=(1, 0, 1))
    # 恢复默认警告设置
    warnings.simplefilter(action='default', category=FutureWarning)
    results = model.fit()
    # 预测下一天的值
    #insamplepredict = results.get_prediction(dynamic = True, typ='levels')
    #insamplepredict = insamplepredict.predicted_mean * scale_std + scale_mean
    outsamplepredict = results.forecast(steps=1)
    outsamplepredict = outsamplepredict.values[0] * scale_std + scale_mean
    # return insamplepredict,outsamplepredict
    return outsamplepredict

#==============================================================================

# 设置滚动窗口的大小和步长
window_size = [126,500,1000,2000]
step_size = 1
save_dir = 'arima.results'
# 存储预测结果的列表
for window_size in window_size:
    outsampleprediction = []
    # 滚动窗口预测
    for i in range(window_size, len(sp500), step_size):
        # 获取当前窗口的数据
        window_data = sp500.iloc[i - window_size:i]
        # insamples,oneoutsample = predict_arima(window_data)
        oneoutsample = predict_arima(window_data)
        # 将预测结果添加到列表中
        # insampleprediction.append(insamples.to_list())
        outsampleprediction.append(oneoutsample)
        print(f"Predicting window {window_size}, Day {i}...")

        # 将预测结果添加到原始数据中
        # insampleprediction = [item for sublist in insampleprediction for item in sublist]
        # sp500['inpredictions'] = insampleprediction
       
#==============================================================================

    sp500['outpredictions'+str(window_size)] = [None] * window_size + outsampleprediction
    df_name = 'arima_'+str(window_size)+'.csv'
    sp500['outpredictions'+str(window_size)].to_csv(os.path.join(save_dir, df_name))