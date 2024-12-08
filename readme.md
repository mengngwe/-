# 金融时间序列中基于ARMA方法与GRU方法的对比研究

利用传统的 ARMA 模型与 ARMA - GARCH 模型以及基于深度学习的 GRU 模型对标普 500 指数进行日收益率的预测

### 内容与复现步骤
- 原始数据集 SP500_19900102_20221231.csv
- 三个模型的预测结果分别在对应的文件夹下：arima.results、arima-garch.results、gru_results
- 得到上述结果分别通过arima.py, arima-garch.ipynb, gru.py，注意运行arima-garch.ipynb, gru.py时每次只能得到一个训练数据长度对应结果，要手动修改要研究的训练数据长度和保存文件名称


### 运行环境

Python 3.10.0

### 安装依赖包

numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
statsmodels==0.14.1
arch==7.2.0
torch==2.2.1

### 克隆仓库
git clone https://github.com/mengngwe/-.git


