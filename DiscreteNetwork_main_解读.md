# `DiscreteNetwork_main.py` 详细代码解读

---

## 主要目的

本脚本实现了一个基于神经网络的**欧拉弹性曲线（Euler's Elastica）离散近似**方法。其核心思想是：给定弹性梁两端的边界条件（位置和切向量），通过训练一个前馈神经网络（支持残差连接），直接预测梁内部各离散节点的位置坐标和切向量，从而高效地近似离散数值解。

脚本主要完成以下任务：

1. 加载并预处理离散弹性曲线数据集；
2. 定义并实例化神经网络模型 `approximate_curve`；
3. 可选地使用 Optuna 进行超参数搜索；
4. 使用固定超参数训练模型，或加载已保存的预训练权重；
5. 在训练集、测试集和验证集上评估误差，并可视化预测与真实轨迹的对比。

---

## 目录

1. [依赖库导入（第 1–30 行）](#1-依赖库导入第-1-30-行)
2. [绘图参数配置（第 37–42 行）](#2-绘图参数配置第-37-42-行)
3. [全局浮点精度与设备设置（第 48–55 行）](#3-全局浮点精度与设备设置第-48-55-行)
4. [用户输入与随机种子（第 61–68 行）](#4-用户输入与随机种子第-61-68-行)
5. [神经网络类 `approximate_curve`（第 70–102 行）](#5-神经网络类-approximate_curve第-70-102-行)
   - [5.1 `__init__` 初始化方法](#51-__init__-初始化方法第-71-88-行)
   - [5.2 `forward` 前向传播方法](#52-forward-前向传播方法第-90-102-行)
6. [辅助函数 `define_model`（第 106–117 行）](#6-辅助函数-define_model第-106-117-行)
7. [目标函数 `objective`（第 121–185 行）](#7-目标函数-objective第-121-185-行)
8. [Optuna 超参数搜索（第 187–198 行）](#8-optuna-超参数搜索第-187-198-行)
9. [最优超参数获取（第 200–221 行）](#9-最优超参数获取第-200-221-行)
10. [最优模型定义函数 `define_best_model`（第 222–238 行）](#10-最优模型定义函数-define_best_model第-222-238-行)
11. [模型训练或加载（第 242–268 行）](#11-模型训练或加载第-242-268-行)
12. [结果可视化（第 274–277 行）](#12-结果可视化第-274-277-行)
13. [预测推理计时（第 283–291 行）](#13-预测推理计时第-283-291-行)
14. [辅助脚本解读：`Scripts/GetData.py`](#14-辅助脚本解读scriptsgetsatapy)
    - [14.1 函数 `loadData`](#141-函数-loaddata)
    - [14.2 类 `dataset`](#142-类-dataset)
    - [14.3 函数 `getDataLoaders`](#143-函数-getdataloaders)
15. [辅助脚本解读：`Scripts/Training.py`](#15-辅助脚本解读scriptstrainingpy)
    - [15.1 类 `EarlyStopper`](#151-类-earlystopper)
    - [15.2 函数 `train`](#152-函数-train)
16. [辅助脚本解读：`Scripts/PlotResults.py`](#16-辅助脚本解读scriptsplotresultspy)
    - [16.1 函数 `plotResults`](#161-函数-plotresults)
17. [辅助脚本解读：`Scripts/SavedParameters.py`](#17-辅助脚本解读scriptssavedparameterspy)
    - [17.1 函数 `hyperparams`](#171-函数-hyperparams)

---

## 1. 依赖库导入（第 1–30 行）

```python
1:  #!/usr/bin/env python
2:  # coding: utf-8
8:  get_ipython().system('pip install optuna')
9:  import optuna
10: import torch
11: import random
12: import torch.nn as nn
13: import numpy as np
14: import matplotlib.pyplot as plt
15: import matplotlib
16: import torch.nn.functional as F
17: from csv import writer
18: import seaborn as sns
19: import os
20: os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
26: from Scripts.GetData import getDataLoaders, loadData
27: from Scripts.Training import train
28: from Scripts.PlotResults import plotResults
29: from Scripts.SavedParameters import hyperparams
30: import pandas as pd
```

**解释：**

- **第 1–2 行**：Shebang 行与编码声明，指定 Python 解释器路径及文件编码为 UTF-8。
- **第 8 行**：在 Jupyter Notebook 环境中通过 shell 命令安装 `optuna` 超参数优化库。
- **第 9–19 行**：导入核心依赖库：
  - `optuna`：贝叶斯超参数优化框架；
  - `torch`、`torch.nn`、`torch.nn.functional`：PyTorch 深度学习框架及其神经网络模块；
  - `random`、`numpy`：数值计算与随机数生成；
  - `matplotlib.pyplot`、`matplotlib`、`seaborn`：绘图库；
  - `csv.writer`：CSV 文件写入工具；
  - `os`：操作系统接口。
- **第 20 行**：设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`，避免 Intel OpenMP 与 LLVM OpenMP 共存时的冲突报错（常见于 macOS 环境）。
- **第 26–30 行**：从 `Scripts/` 子目录导入本项目自定义模块，包括数据加载、训练、绘图、超参数管理函数，以及 `pandas` 数据分析库。

---

## 2. 绘图参数配置（第 37–42 行）

```python
37: sns.set_style("darkgrid")
38: sns.set(font = "Times New Roman")
39: sns.set_context("paper")
40: plt.rcParams['mathtext.fontset'] = 'cm'
41: plt.rcParams['font.family'] = 'STIXGeneral'
42: plt_kws = {"rasterized": True}
```

**解释：**

- **第 37–39 行**：使用 `seaborn` 设置全局绘图风格：深色网格背景、Times New Roman 字体、论文级别字号比例（`paper` context 会缩小字号，适合出版物插图）。
- **第 40–41 行**：设置 Matplotlib 数学公式字体为 Computer Modern（`cm`），常规字体为 STIX General（与 LaTeX 风格一致）。
- **第 42 行**：定义绘图关键字参数字典 `plt_kws`，设置 `rasterized=True` 以便导出为矢量图时对复杂散点图进行栅格化，减小文件体积。

---

## 3. 全局浮点精度与设备设置（第 48–55 行）

```python
48: torch.set_default_dtype(torch.float32)
54: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
55: print(device)
```

**解释：**

- **第 48 行**：将 PyTorch 默认张量浮点类型设为 32 位单精度浮点数（`float32`），平衡计算精度与内存、速度。
- **第 54–55 行**：自动检测是否有可用的 CUDA GPU。若有，则将计算设备设为第一块 GPU（`cuda:0`），否则退回 CPU。

---

## 4. 用户输入与随机种子（第 61–68 行）

```python
61: datacase = int(input("Which datacase do you want to work with?\n"))
62: percentage_train = float(input("Which percentage of the dataset do you want to use for training? Choose among 0.1,0.2,0.4,0.8\n"))
64: print(f"\n\n Case with percentage_train={percentage_train} and datacase={datacase}\n\n")
66: torch.manual_seed(1)
67: np.random.seed(1)
68: random.seed(1)
```

**解释：**

- **第 61–62 行**：从命令行接受用户输入：
  - `datacase`：数据集编号，`1` 表示仅使用"两端边界条件"数据集，`2` 表示同时使用"两端"与"右端"数据集；
  - `percentage_train`：训练集占总数据集的比例，可选值为 \(0.1,\,0.2,\,0.4,\,0.8\)。
- **第 66–68 行**：固定 PyTorch、NumPy 和 Python 标准库的随机种子均为 `1`，保证实验可复现性。三行分别对应：

\[
\text{torch.manual\_seed}(1), \quad \text{np.random.seed}(1), \quad \text{random.seed}(1)
\]

---

## 5. 神经网络类 `approximate_curve`（第 70–102 行）

### 5.1 `__init__` 初始化方法（第 71–88 行）

```python
70: class approximate_curve(nn.Module):
71:     def __init__(self, is_res=True, normalize=True, act_name='tanh',
72:                  nlayers=3, hidden_nodes=50, output_dim=204):
73:         super().__init__()
74:         torch.manual_seed(1)
75:         np.random.seed(1)
76:         random.seed(1)
77:         self.act_dict = {"tanh":   lambda x: torch.tanh(x),
78:                          "sigmoid":lambda x: torch.sigmoid(x),
79:                          "swish":  lambda x: x * torch.sigmoid(x),
80:                          "relu":   lambda x: torch.relu(x),
81:                          "lrelu":  lambda x: F.leaky_relu(x)}
82:         self.is_norm = normalize
83:         self.is_res  = is_res
84:         self.act     = self.act_dict[act_name]
85:         self.nlayers = nlayers
86:         self.first   = nn.Linear(8, hidden_nodes)
87:         self.linears = nn.ModuleList([nn.Linear(hidden_nodes, hidden_nodes)
88:                                       for i in range(self.nlayers)])
89:         self.last    = nn.Linear(hidden_nodes, output_dim)
```

**解释：**

`approximate_curve` 继承自 `nn.Module`，是本脚本的核心神经网络类，用于将 8 维边界条件输入映射到 `output_dim` 维的内部节点坐标输出。

- **第 73 行**：调用父类 `nn.Module` 的初始化方法，注册网络参数。
- **第 74–76 行**：在网络初始化阶段固定随机种子，确保每次创建模型的权重初始化完全相同。
- **第 77–81 行**：构建激活函数字典，支持五种激活函数：
  - `tanh`：双曲正切，\(\sigma(x) = \tanh(x)\)；
  - `sigmoid`：Sigmoid，\(\sigma(x) = \dfrac{1}{1+e^{-x}}\)；
  - `swish`：Swish，\(\sigma(x) = x \cdot \dfrac{1}{1+e^{-x}}\)；
  - `relu`：线性整流，\(\sigma(x) = \max(0,x)\)；
  - `lrelu`：Leaky ReLU，\(\sigma(x) = \max(\alpha x, x)\)，其中 \(\alpha\) 为小斜率（PyTorch 默认 0.01）。
- **第 82–85 行**：存储控制标志和超参数：`is_norm`（是否归一化输入）、`is_res`（是否使用残差连接）、`act`（激活函数对象）、`nlayers`（隐藏层数量）。
- **第 86 行**：定义**输入层**（第一层线性层），将 8 维输入映射到 `hidden_nodes` 维隐含空间：

\[
\mathbf{h}^{(0)} = \sigma\!\left(\mathbf{W}^{(0)}\,\mathbf{x} + \mathbf{b}^{(0)}\right), \quad \mathbf{W}^{(0)} \in \mathbb{R}^{H \times 8},\ \mathbf{b}^{(0)} \in \mathbb{R}^{H}
\]

其中 \(H\) 为 `hidden_nodes`，\(\mathbf{x} \in \mathbb{R}^8\) 为边界条件输入向量。

- **第 87–88 行**：定义 `nlayers` 个**隐藏层**线性变换，均为方阵 \(\mathbb{R}^{H \times H}\)，使用 `nn.ModuleList` 管理：

\[
\mathbf{W}^{(l)} \in \mathbb{R}^{H \times H},\quad \mathbf{b}^{(l)} \in \mathbb{R}^{H}, \quad l = 1, \ldots, n_{\text{layers}}
\]

- **第 89 行**：定义**输出层**线性变换，将隐含表示映射回 `output_dim` 维输出：

\[
\hat{\mathbf{y}} = \mathbf{W}^{(\text{out})}\,\mathbf{h}^{(n_{\text{layers}})} + \mathbf{b}^{(\text{out})},\quad \mathbf{W}^{(\text{out})} \in \mathbb{R}^{d_{\text{out}} \times H}
\]

其中 `output_dim` \(= 4(N-2)\)，\(N\) 为弹性曲线的总节点数，\(N-2\) 为内部节点数，每个节点有位置 \((q_x, q_y)\) 和切向量 \((q'_x, q'_y)\) 共 4 个分量。

---

### 5.2 `forward` 前向传播方法（第 90–102 行）

```python
90: def forward(self, x):
92:     if self.is_norm:
93:         x[:, 0] = (x[:, 0] - 1.5) / 1.5
94:         x[:, 4] = (x[:, 4] - 1.5) / 1.5
95:     x = self.act(self.first(x))
96:     for i in range(self.nlayers):
97:         if self.is_res:
98:             x = x + self.act(self.linears[i](x))
99:         else:
100:            x = self.act(self.linears[i](x))
102:    return self.last(x)
```

**解释：**

- **第 92–94 行**：若启用归一化（`is_norm=True`），对输入向量 \(\mathbf{x}\) 的第 0 和第 4 个分量（左端和右端 \(q_x\) 坐标）进行平移-缩放归一化：

\[
\tilde{x}_0 = \frac{x_0 - 1.5}{1.5}, \qquad \tilde{x}_4 = \frac{x_4 - 1.5}{1.5}
\]

将范围 \([0, 3]\) 的坐标映射到 \([-1, 1]\)，有助于加速梯度收敛。

- **第 95 行**：通过输入层进行第一次线性变换，并施加激活函数：

\[
\mathbf{h}^{(0)} = \sigma\!\left(\mathbf{W}^{(0)}\,\tilde{\mathbf{x}} + \mathbf{b}^{(0)}\right)
\]

- **第 96–100 行**：循环 `nlayers` 次，逐层更新隐含表示。有两种模式：

  - **残差模式**（`is_res=True`，ResNet 风格）：

\[
\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \sigma\!\left(\mathbf{W}^{(l)}\,\mathbf{h}^{(l)} + \mathbf{b}^{(l)}\right)
\]

  残差连接（skip connection）帮助梯度直接流过深层网络，缓解梯度消失问题。

  - **标准 MLP 模式**（`is_res=False`）：

\[
\mathbf{h}^{(l+1)} = \sigma\!\left(\mathbf{W}^{(l)}\,\mathbf{h}^{(l)} + \mathbf{b}^{(l)}\right)
\]

- **第 102 行**：通过输出层线性变换得到最终预测结果（无激活函数，直接输出）：

\[
\hat{\mathbf{y}} = \mathbf{W}^{(\text{out})}\,\mathbf{h}^{(n_{\text{layers}})} + \mathbf{b}^{(\text{out})} \in \mathbb{R}^{4(N-2)}
\]

---

## 6. 辅助函数 `define_model`（第 106–117 行）

```python
104: num_nodes, _, _ = loadData(datacase)
106: def define_model(trial):
107:     torch.manual_seed(1)
108:     np.random.seed(1)
109:     random.seed(1)
110:     is_res     = False
111:     normalize  = True
112:     act_name   = "tanh"
113:     nlayers      = trial.suggest_int("n_layers", 0, 10)
114:     hidden_nodes = trial.suggest_int("hidden_nodes", 10, 1000)
115:     model = approximate_curve(is_res, normalize, act_name, nlayers,
116:                                hidden_nodes, output_dim=int(4*(num_nodes-2)))
117:     return model
```

**解释：**

- **第 104 行**：调用 `loadData(datacase)` 获取节点数 `num_nodes`（对应弹性曲线的总离散节点数 \(N\)，通常为 51）及训练/测试数据矩阵（此处用 `_` 忽略数据矩阵）。
- **第 106–117 行**：`define_model(trial)` 是供 Optuna 调用的模型构造函数，每次 trial 都会构建一个新的 `approximate_curve` 实例：
  - `is_res=False`：固定不使用残差连接；
  - `normalize=True`：固定使用输入归一化；
  - `act_name="tanh"`：固定使用双曲正切激活；
  - `nlayers`：由 Optuna 在整数区间 \([0, 10]\) 内搜索；
  - `hidden_nodes`：由 Optuna 在整数区间 \([10, 1000]\) 内搜索；
  - `output_dim = 4(N-2)`：内部节点数 \(N-2\) 乘以每节点 4 个自由度（位置 \(q_x, q_y\) + 切向量 \(q'_x, q'_y\)）。

---

## 7. 目标函数 `objective`（第 121–185 行）

```python
119: from torch.utils.data import Dataset, DataLoader
121: def objective(trial):
123:     torch.manual_seed(1)
124:     np.random.seed(1)
125:     random.seed(1)
128:     model = define_model(trial)
129:     model.to(device)
131:     lr           = 1e-3
132:     weight_decay = 0
133:     gamma = trial.suggest_float("gamma", 0, 1e-2)
134:     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
136:     criterion = nn.MSELoss()
138:     batch_size = 32
139:     _, _, _, _, x_val, y_val, trainloader, _, valloader = getDataLoaders(
140:             batch_size, datacase, percentage_train)
141:     ...
146:     epochs    = 300
147:     scheduler = torch.optim.lr_scheduler.StepLR(
148:             optimizer, step_size=int(0.45*epochs), gamma=0.1)
149:     loss = train(model, gamma, criterion, scheduler, optimizer,
150:             epochs, trainloader, valloader, device)
151:     print('Loss ', loss.item())
152:     error = 1000
154:     if not torch.isnan(loss):
155:         model.eval()
156:         learned_traj = np.zeros_like(y_val)
158:         bcs_val = torch.from_numpy(x_val.astype(np.float32)).to(device)
159:         learned_traj = model(bcs_val).detach().cpu().numpy()
160:         error = np.mean((learned_traj - y_val)**2)
162:         print(f"The error on the validation trajectories is: {error}.")
165:     if trial.number == 0:
166:         labels = []
167:         for lab, _ in trial.params.items():
168:             labels.append(str(lab))
169:         labels.append("MSE")
170:         with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
171:             writer_object = writer(f_object)
172:             writer_object.writerow(labels)
175:     results = []
176:     for _, value in trial.params.items():
177:         results.append(str(value))
179:     results.append(error)
181:     with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
182:         writer_object = writer(f_object)
183:         writer_object.writerow(results)
185:     return error
```

**解释：**

`objective(trial)` 是 Optuna 的目标函数，每次调用对应一组超参数组合的完整训练+评估流程，返回验证集 MSE 误差供 Optuna 最小化。

- **第 128–129 行**：使用当前 trial 的超参数构建模型并将其移动到计算设备。
- **第 131–134 行**：设置优化器超参数：学习率 \(\eta = 10^{-3}\)，权重衰减 \(\lambda = 0\)，\(\gamma\) 在 \([0, 10^{-2}]\) 内由 Optuna 搜索。Adam 优化器更新规则为：

\[
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

其中 \(\hat{m}_t\) 和 \(\hat{v}_t\) 分别为梯度的一阶和二阶矩估计。

- **第 136 行**：定义损失函数为均方误差（MSE）：

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \left\|\hat{\mathbf{y}}_i - \mathbf{y}_i\right\|^2
\]

- **第 147–148 行**：定义分步衰减学习率调度器 `StepLR`，每隔 \(\lfloor 0.45 \times 300 \rfloor = 135\) 个 epoch 将学习率乘以系数 \(0.1\)：

\[
\eta_t = \eta_0 \times 0.1^{\lfloor t / 135 \rfloor}
\]

- **第 149–150 行**：调用 `train(...)` 函数（详见第 15.2 节）执行 300 个 epoch 的训练，返回最后一个 batch 的损失值。
- **第 154–160 行**：若训练损失不为 NaN，则在验证集上进行前向推断，计算验证集 MSE 误差：

\[
\text{error} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \left\|\hat{\mathbf{y}}_i^{\text{val}} - \mathbf{y}_i^{\text{val}}\right\|^2
\]

- **第 165–184 行**：将当前 trial 的超参数与对应 MSE 误差追加写入 CSV 文件（文件名包含训练比例，格式为 `results{pct}_Fig2.csv`）。第 0 号 trial 时额外写入列名表头。

---

## 8. Optuna 超参数搜索（第 187–198 行）

```python
187: optuna_study = input("Do you want to do hyperparameter test? Type yes or no: ")
188: params = {}
189: if optuna_study == "yes":
190:     optuna_study = True
191: else:
192:     optuna_study = False
193: if optuna_study:
194:     study = optuna.create_study(direction="minimize", study_name="Euler Elastica")
195:     study.optimize(objective, n_trials=5)
196:     print("Study statistics: ")
197:     print("Number of finished trials: ", len(study.trials))
198:     params = study.best_params
```

**解释：**

- **第 187–192 行**：从命令行读取用户选择，决定是否启动超参数搜索。
- **第 194 行**：创建一个 Optuna Study，搜索方向为 `"minimize"`（最小化目标函数），即最小化验证集 MSE。
- **第 195 行**：执行 5 次独立 trial（`n_trials=5`），每次 trial 由 Optuna 的 TPE（Tree-structured Parzen Estimator）贝叶斯优化算法根据历史结果建议新的超参数组合：

\[
\theta^* = \arg\min_{\theta \in \Theta}\ \text{MSE}_{\text{val}}\!\left(f_\theta\right)
\]

其中 \(\Theta\) 表示超参数搜索空间（\(n_{\text{layers}} \in [0,10]\)，\(h \in [10,1000]\)，\(\gamma \in [0, 0.01]\)）。

- **第 198 行**：从 study 对象中提取使验证误差最小的一组超参数 `best_params`。

---

## 9. 最优超参数获取（第 200–221 行）

```python
200: torch.manual_seed(1)
201: np.random.seed(1)
202: random.seed(1)
204: manual_input = False
205: if params == {}:
206:     if manual_input:
207:         ...
208:         params = {'n_layers': nlayers, 'hidden_nodes': hidden_nodes, 'gamma': gamma}
219:     else:
220:         params = hyperparams(datacase, percentage_train)
221: print(f'The hyperparameters yielding the best results for this case are: {params}')
```

**解释：**

- **第 200–202 行**：再次固定随机种子，确保后续模型初始化和数据加载的一致性。
- **第 204–220 行**：若 `params` 字典为空（即未执行 Optuna 搜索），则通过两种方式获取超参数：
  - 若 `manual_input=True`：交互式命令行手动输入；
  - 若 `manual_input=False`（默认）：调用 `hyperparams(datacase, percentage_train)` 从 `SavedParameters.py` 中的预存最优超参数表中查找匹配的记录（详见第 17 节）。

---

## 10. 最优模型定义函数 `define_best_model`（第 222–238 行）

```python
222: def define_best_model():
224:     torch.manual_seed(1)
225:     np.random.seed(1)
226:     random.seed(1)
228:     normalize    = True
229:     act          = "tanh"
230:     nlayers      = params["n_layers"]
231:     hidden_nodes = params["hidden_nodes"]
232:     is_res       = False
234:     print("Nodes: ", hidden_nodes)
236:     model = approximate_curve(is_res, normalize, act, nlayers,
237:                                hidden_nodes, int(4*(num_nodes-2)))
238:     return model
239: model = define_best_model()
240: model.to(device)
```

**解释：**

- **第 222–238 行**：用已确定的最优超参数实例化 `approximate_curve` 网络。激活函数固定为 `tanh`，不使用残差连接（`is_res=False`），使用输入归一化。
- **第 239–240 行**：创建模型实例并将其所有参数（权重、偏置张量）迁移到目标计算设备（GPU 或 CPU）。

---

## 11. 模型训练或加载（第 242–268 行）

```python
242: TrainMode = input("Train Mode True or False? Type 0 for False and 1 for True: ") == "1"
243: weight_decay = 0.
244: lr           = 1e-3
245: gamma        = params["gamma"]
246: nlayers      = params["n_layers"]
247: hidden_nodes = params["hidden_nodes"]
248: batch_size   = 32
249: epochs       = 300
250: optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
251: scheduler  = torch.optim.lr_scheduler.StepLR(
252:                  optimizer, step_size=int(0.45*epochs), gamma=0.1)
253: criterion  = nn.MSELoss()
254: x_train, y_train, x_test, y_test, x_val, y_val, \
255:     trainloader, testloader, valloader = getDataLoaders(
256:         batch_size, datacase, percentage_train)
257: model.to(device)
258:
259: if TrainMode:
260:     loss = train(model, gamma, criterion, scheduler, optimizer,
261:                  epochs, trainloader, valloader, device)
262:     if datacase == 1:
263:         torch.save(model.state_dict(),
264:                    f'TrainedModels/BothEnds{percentage_train}data.pt')
265:     if datacase == 2:
266:         torch.save(model.state_dict(),
267:                    f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt')
268: else:
269:     if datacase == 1:
270:         pretrained_dict = torch.load(
271:                   f'TrainedModels/BothEnds{percentage_train}data.pt',
272:                   map_location=device)
273:     if datacase == 2:
274:         pretrained_dict = torch.load(
275:                   f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt',
276:                   map_location=device)
277:     model.load_state_dict(pretrained_dict)
278: model.eval()
```

**解释：**

- **第 242 行**：获取用户选择：输入 `"1"` 表示训练新模型，输入 `"0"` 表示加载预训练模型。
- **第 243–249 行**：配置训练超参数：权重衰减 \(\lambda=0\)（无 L2 正则），学习率 \(\eta=10^{-3}\)，`gamma` 和网络结构来自最优超参数，批大小 \(B=32\)，训练轮次 \(T=300\)。
- **第 250–251 行**：实例化 Adam 优化器和 `StepLR` 学习率调度器（同第 7 节）。
- **第 253 行**：实例化 MSE 损失函数（同第 7 节）。
- **第 254–256 行**：调用 `getDataLoaders(...)` 加载数据，返回训练/测试/验证集的 numpy 数组和对应 `DataLoader` 对象（详见第 14.3 节）。
- **第 259–267 行（训练分支）**：训练模型，训练结束后根据 `datacase` 将模型参数字典 `state_dict` 保存为 `.pt` 文件。
- **第 268–277 行（加载分支）**：从磁盘加载预训练权重，使用 `map_location=device` 确保模型加载到正确的计算设备（无论保存时使用的是 GPU 还是 CPU）。
- **第 278 行**：将模型切换到评估模式（`eval()`），禁用 `Dropout` 和 `BatchNorm` 的训练行为（本模型未使用，但为规范用法）。

---

## 12. 结果可视化（第 274–277 行）

```python
274: model.eval()
277: plotResults(model, device, x_train, y_train, x_test, y_test,
278:              x_val, y_val, num_nodes, datacase, percentage_train,
279:              gamma, nlayers, hidden_nodes)
```

**解释：**

- **第 274 行**：再次确认模型处于评估模式。
- **第 277–279 行**：调用 `plotResults(...)` 函数（详见第 16.1 节），对训练集、测试集和验证集分别计算预测轨迹，输出各集合的 MSE 误差，并绘制预测与真实弹性曲线的对比图。

---

## 13. 预测推理计时（第 283–291 行）

```python
283: import time
284: test_bvs     = torch.from_numpy(x_test.astype(np.float32))
285: initial_time = time.time()
286: preds        = model(test_bvs)
287: final_time   = time.time()
288: total_time   = final_time - initial_time
289: print("Number of trajectories in the test set : ", len(test_bvs))
290: print("Total time to predict test trajectories : ", total_time)
291: print("Average time to predict test trajectories : ", total_time / len(test_bvs))
```

**解释：**

- **第 284 行**：将测试集边界条件数组转换为 `float32` 类型的 PyTorch 张量（不移动到 GPU，用于 CPU 推理计时）。
- **第 285–288 行**：记录前向推断的总耗时：

\[
t_{\text{total}} = t_{\text{final}} - t_{\text{initial}}
\]

- **第 289–291 行**：打印测试集大小 \(N_{\text{test}}\)、总推断时间和单样本平均推断时间：

\[
t_{\text{avg}} = \frac{t_{\text{total}}}{N_{\text{test}}}
\]

---

## 14. 辅助脚本解读：`Scripts/GetData.py`

### 14.1 函数 `loadData`

```python
# GetData.py 第 7–30 行
def loadData(datacase=1):
    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir + "/DataSets")

    both_ends_360_sol = open("both_ends.txt", "r")
    trajectoriesload_b_360 = np.loadtxt(both_ends_360_sol)
    right_end_360_sol = open("right_end.txt", "r")
    trajectoriesload_r_360 = np.loadtxt(right_end_360_sol)

    if datacase == 1:
        trajectories_train = trajectoriesload_b_360
        trajectories_test  = trajectories_train
    elif datacase == 2:
        trajectories_train = np.concatenate(
            (trajectoriesload_b_360, trajectoriesload_r_360), axis=0)
        trajectories_test  = trajectories_train

    num_nodes = trajectories_train.shape[1] // 4
    os.chdir(original_dir)
    return num_nodes, trajectories_train, trajectories_test
```

**解释：**

- **功能**：从 `DataSets/` 目录读取数值求解器生成的弹性曲线离散轨迹数据，返回节点数和数据矩阵。
- **数据格式**：每行表示一条完整的弹性曲线，列按节点顺序排列，每个节点依次存储 4 个分量 \((q_x^{(k)},\, q_y^{(k)},\, q'^{(k)}_x,\, q'^{(k)}_y)\)，即位置坐标和弧长方向的单位切向量。总列数为 \(4N\)，其中 \(N\) 为节点数。
- **`datacase==1`**：仅加载"两端边界条件"数据集（both-ends），弹性梁两端的位置和切向量均给定；
- **`datacase==2`**：合并"两端"与"右端"（right-end）数据集，后者仅约束右端边界。
- **节点数计算**：

\[
N = \frac{\text{数据矩阵列数}}{4}
\]

- **工作目录切换**：函数进入 `DataSets/` 子目录读取文件，结束后恢复原始工作目录，保证后续代码的路径正确。

---

### 14.2 类 `dataset`

```python
# GetData.py 第 32–42 行
class dataset(Dataset):
    def __init__(self, x, y):
        self.bcs = torch.from_numpy(x.astype(np.float32))
        self.internal_node_outputs = torch.from_numpy(y.astype(np.float32))
        self.length = x.shape[0]

    def __getitem__(self, idx):
        return self.bcs[idx], self.internal_node_outputs[idx]

    def __len__(self):
        return self.length
```

**解释：**

- **功能**：封装 NumPy 数组为 PyTorch `Dataset`，供 `DataLoader` 迭代使用。
- **`__init__`**：将边界条件数组 \(\mathbf{X} \in \mathbb{R}^{M \times 8}\) 和内部节点目标数组 \(\mathbf{Y} \in \mathbb{R}^{M \times 4(N-2)}\) 转换为 `float32` 张量，其中 \(M\) 为样本数。
- **`__getitem__`**：按索引 `idx` 返回一对 `(x_i, y_i)`，即第 \(i\) 个样本的边界条件向量和对应的内部节点坐标向量。
- **`__len__`**：返回数据集大小，供 `DataLoader` 确定 epoch 长度。

---

### 14.3 函数 `getDataLoaders`

```python
# GetData.py 第 45–95 行
def getDataLoaders(batch_size, datacase, percentage_train):
    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    _, data_train, data_test = loadData(datacase)
    x_full_train = np.concatenate((data_train[:, :4], data_train[:, -4:]), axis=1)
    y_full_train = data_train[:, 4:-4]
    N     = len(x_full_train)
    NTrain = int(percentage_train * N)
    idx_shuffle_train = np.arange(N)
    random.shuffle(idx_shuffle_train)
    x_full_train = x_full_train[idx_shuffle_train]
    y_full_train = y_full_train[idx_shuffle_train]
    ...
    x_train, y_train = x_full_train[:NTrain], y_full_train[:NTrain]
    Number_Test_Points = int(fact * N)
    x_test, y_test = x_full_test[NTrain:NTrain+Number_Test_Points], ...
    x_val,  y_val  = x_full_test[NTrain+Number_Test_Points:NTrain+2*Number_Test_Points], ...
    ...
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=len(x_test), shuffle=True)
    valloader   = DataLoader(valset,   batch_size=len(x_val),  shuffle=True)
    return x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader
```

**解释：**

- **功能**：从原始轨迹数据中提取输入-输出对，随机打乱后按比例划分训练集、测试集和验证集，并封装为 `DataLoader`。
- **输入特征提取**：每条轨迹的前 4 列和后 4 列分别对应左端节点 \((q_x^{(0)}, q_y^{(0)}, q'^{(0)}_x, q'^{(0)}_y)\) 和右端节点 \((q_x^{(N-1)}, q_y^{(N-1)}, q'^{(N-1)}_x, q'^{(N-1)}_y)\)，拼接成 8 维边界条件向量：

\[
\mathbf{x}_i = \left[q_x^{(0)},\, q_y^{(0)},\, q'^{(0)}_x,\, q'^{(0)}_y,\, q_x^{(N-1)},\, q_y^{(N-1)},\, q'^{(N-1)}_x,\, q'^{(N-1)}_y\right]^\top \in \mathbb{R}^8
\]

- **目标标签提取**：去除首尾节点（边界条件），保留中间 \(N-2\) 个内部节点的所有分量：

\[
\mathbf{y}_i = \left[q_x^{(1)},\, q_y^{(1)},\, q'^{(1)}_x,\, q'^{(1)}_y,\, \ldots,\, q_x^{(N-2)},\, q_y^{(N-2)},\, q'^{(N-2)}_x,\, q'^{(N-2)}_y\right]^\top \in \mathbb{R}^{4(N-2)}
\]

- **数据集划分**：总样本数 \(M\) 按以下规则划分：

\[
M_{\text{train}} = \lfloor p \cdot M \rfloor, \quad M_{\text{test}} = M_{\text{val}} = \lfloor f \cdot M \rfloor
\]

其中训练比例 \(p = \texttt{percentage\_train}\)，测试/验证比例 \(f\) 根据 \(p\) 取值：当 \(p=0.8\) 时 \(f=0.1\)；当 \(p=0.7\) 时 \(f=0.15\)；其余情况 \(f=0.2\)。

- **`DataLoader` 封装**：训练集按批大小 \(B=32\) 随机采样；测试集和验证集一次性加载全部样本（`batch_size=len(x_test)`）。

---

## 15. 辅助脚本解读：`Scripts/Training.py`

### 15.1 类 `EarlyStopper`

```python
# Training.py 第 6–21 行
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience            = patience
        self.min_delta           = min_delta
        self.counter             = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
```

**解释：**

- **功能**：实现**早停（Early Stopping）**机制，防止过拟合。当验证损失连续 `patience` 个 epoch 未改善超过 `min_delta` 时，触发早停。
- **`__init__`**：初始化耐心计数器 `counter=0`，历史最低验证损失为 \(+\infty\)。
- **`early_stop(validation_loss)`**：
  - 若当前验证损失 \(\mathcal{L}_{\text{val}}^{(t)}\) 低于历史最低，更新记录并重置计数器；
  - 否则若 \(\mathcal{L}_{\text{val}}^{(t)} > \mathcal{L}_{\text{val}}^{\min} + \delta\)，递增计数器，当 `counter >= patience` 时返回 `True` 触发停止。

> **注意**：在当前代码中，`EarlyStopper` 虽被实例化（`Training.py` 第 29 行），但实际调用 `early_stop()` 的代码段被注释掉（第 81–84 行），因此训练过程**不使用**早停机制，始终运行完 `epochs` 个 epoch。

---

### 15.2 函数 `train`

```python
# Training.py 第 23–97 行
def train(model, gamma, criterion, scheduler, optimizer,
          epochs, trainloader, valloader, device):
    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    early_stopper = EarlyStopper(patience=100, min_delta=0)
    losses = []; losses_val = []
    for epoch in range(epochs):
        train_loss = 0.; counter = 0.
        for _, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            predicted = model(inputs)

            loss = criterion(predicted, labels)

            predicted = torch.cat((inputs[:, :4], predicted, inputs[:, 4:]), dim=1)
            labels    = torch.cat((inputs[:, :4], labels,    inputs[:, 4:]), dim=1)

            predicted_first   = predicted[:, :-4]
            predicted_forward = predicted[:,  4:]
            labels_first   = labels[:, :-4]
            labels_forward = labels[:,  4:]

            diff_predicted = predicted_forward - predicted_first
            diff_labels    = labels_forward    - labels_first

            loss += gamma * criterion(diff_predicted, diff_labels)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            counter += 1

        avg_train_loss = train_loss / counter
        ...
        model.eval()
        with torch.no_grad():
            data = next(iter(valloader))
            inputs, labels = data[0].to(device), data[1].to(device)
            predicted = model(inputs)
            val_loss  = criterion(predicted, labels)
        model.train()
        ...
        scheduler.step()
    return loss
```

**解释：**

`train(...)` 函数执行完整的监督学习训练循环，包含自定义的**差分正则化**损失项。

**损失函数设计（核心）：**

对每个 batch，总损失由两部分组成：

1. **主损失**（内部节点 MSE）：预测的内部节点坐标与真实值之间的均方误差：

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^{B} \left\|\hat{\mathbf{y}}_i - \mathbf{y}_i\right\|^2
\]

2. **差分正则化损失**（几何连续性约束）：将预测结果与边界条件拼接为完整轨迹后，计算相邻节点差分向量之间的 MSE：

将内部节点预测 \(\hat{\mathbf{y}}_i\) 与输入边界条件拼接，得到完整轨迹：

\[
\hat{\mathbf{z}}_i = \left[\mathbf{x}_i^{(0:4)},\, \hat{\mathbf{y}}_i,\, \mathbf{x}_i^{(4:8)}\right] \in \mathbb{R}^{4N}
\]

定义前向差分（相邻节点状态向量之差）：

\[
\Delta\hat{\mathbf{z}}_i = \hat{\mathbf{z}}_i^{[k+1:]} - \hat{\mathbf{z}}_i^{[:k]}
\]

即 `predicted_forward - predicted_first`（每个节点 4 个分量的连续差分）。差分正则化损失为：

\[
\mathcal{L}_{\text{diff}} = \frac{1}{B} \sum_{i=1}^{B} \left\|\Delta\hat{\mathbf{z}}_i - \Delta\mathbf{z}_i\right\|^2
\]

**总损失**为：

\[
\mathcal{L} = \mathcal{L}_{\text{MSE}} + \gamma \cdot \mathcal{L}_{\text{diff}}
\]

其中 \(\gamma \geq 0\) 为差分正则化系数，控制几何光滑性约束的强度。

**反向传播与参数更新：**

\[
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta_t \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}
\]

其中 \(\eta_t\) 为当前 epoch 的学习率（由 `StepLR` 调度器管理）。

**验证损失计算：**

每个 epoch 结束后，在 `torch.no_grad()` 上下文中计算验证集损失（仅使用主 MSE 损失，不含差分项）：

\[
\mathcal{L}_{\text{val}} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \left\|\hat{\mathbf{y}}_i^{\text{val}} - \mathbf{y}_i^{\text{val}}\right\|^2
\]

每隔 \(10\%\) 的总 epoch 数打印一次当前平均训练损失。训练结束后返回最后一个 batch 的损失值。

---

## 16. 辅助脚本解读：`Scripts/PlotResults.py`

### 16.1 函数 `plotResults`

```python
# PlotResults.py 第 18–160 行
def plotResults(model, device, x_train, y_train, x_test, y_test,
                x_val, y_val, num_nodes, datacase, percentage_train,
                gamma, number_layers, hidden_nodes):
    train_bcs = torch.from_numpy(x_train.astype(np.float32)).to(device)
    test_bcs  = torch.from_numpy(x_test.astype(np.float32)).to(device)
    val_bcs   = torch.from_numpy(x_val.astype(np.float32)).to(device)

    pred_train = np.concatenate((x_train[:, :4], model(train_bcs).detach().cpu().numpy(),
                                  x_train[:, -4:]), axis=1)
    pred_test  = np.concatenate((x_test[:, :4],  model(test_bcs).detach().cpu().numpy(),
                                  x_test[:, -4:]),  axis=1)
    pred_val   = np.concatenate((x_val[:, :4],   model(val_bcs).detach().cpu().numpy(),
                                  x_val[:, -4:]),   axis=1)

    true_train = np.concatenate((x_train[:, :4], y_train, x_train[:, -4:]), axis=1)
    true_test  = np.concatenate((x_test[:, :4],  y_test,  x_test[:, -4:]),  axis=1)
    true_val   = np.concatenate((x_val[:, :4],   y_val,   x_val[:, -4:]),   axis=1)

    error_training   = np.mean((pred_train[:, 4:-4] - true_train[:, 4:-4])**2)
    error_testing    = np.mean((pred_test[:,  4:-4] - true_test[:,  4:-4])**2)
    error_validation = np.mean((pred_val[:,   4:-4] - true_val[:,   4:-4])**2)
    error_all = np.mean((
        np.concatenate((pred_train[:,4:-4], pred_test[:,4:-4], pred_val[:,4:-4]), 0)
        - np.concatenate((true_train[:,4:-4], true_test[:,4:-4], true_val[:,4:-4]), 0)
    )**2)
    ...
    for i in range(len(pred_test)):
        for j in range(num_nodes):
            norms_q[i, j]  = np.linalg.norm(
                pred_test[i, 4*j:4*j+2] - true_test[i, 4*j:4*j+2])
            norms_qp[i, j] = np.linalg.norm(
                pred_test[i, 4*j+2:4*j+4] - true_test[i, 4*j+2:4*j+4])
    mean_q  = np.mean(norms_q,  axis=0)
    mean_qp = np.mean(norms_qp, axis=0)
    ...
```

**解释：**

- **功能**：对训练、测试和验证集执行批量推断，计算多种误差指标，并绘制三类可视化图形。
- **完整轨迹重建**：对每个数据集，将边界节点（边界条件）与网络预测的内部节点拼接，还原完整 \(N\) 节点轨迹：

\[
\hat{\mathbf{z}}_i = \left[\mathbf{x}_i^{(0:4)},\; \hat{\mathbf{y}}_i,\; \mathbf{x}_i^{(4:8)}\right] \in \mathbb{R}^{4N}
\]

- **MSE 误差计算**：各集合的均方误差（仅计算内部节点）：

\[
\text{error} = \frac{1}{M \cdot 4(N-2)} \sum_{i=1}^{M} \left\|\hat{\mathbf{z}}_i^{[4:-4]} - \mathbf{z}_i^{[4:-4]}\right\|^2
\]

- **逐节点平均误差**：对测试集中每个节点 \(k\)，分别计算位置和切向量的平均预测误差范数：

\[
\bar{e}_q^{(k)} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left\| \hat{\mathbf{q}}_i^{(k)} - \mathbf{q}_i^{(k)} \right\|_2, \quad k = 0, 1, \ldots, N-1
\]

\[
\bar{e}_{q'}^{(k)} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left\| \hat{\mathbf{q}}'^{(k)}_i - \mathbf{q}'^{(k)}_i \right\|_2
\]

其中 \(\hat{\mathbf{q}}_i^{(k)} = (\hat{z}_{i,4k},\, \hat{z}_{i,4k+1})\) 为节点 \(k\) 的预测位置，\(\hat{\mathbf{q}}'^{(k)}_i = (\hat{z}_{i,4k+2},\, \hat{z}_{i,4k+3})\) 为预测切向量。

- **绘图内容**：
  - **图 1**：对比测试集预测与真实弹性曲线的 \((q_x, q_y)\) 坐标折线图（黑线为真实，红虚线为预测）；
  - **图 2**：对比测试集预测与真实切向量 \((q'_x, q'_y)\) 的散点图，叠加单位圆（弧长参数化要求 \(\|(q'_x, q'_y)\|=1\)）；
  - **图 3**：各节点位置和切向量平均误差范数随节点序号 \(k\) 的变化曲线。

---

## 17. 辅助脚本解读：`Scripts/SavedParameters.py`

### 17.1 函数 `hyperparams`

```python
# SavedParameters.py 第 4–23 行
def hyperparams(case, pctg):
    best_vals = pd.DataFrame()
    best_vals["percentage_train"] = np.array([0.1, 0.2, 0.4, 0.8, 0.8])
    best_vals["datacase"]         = np.array([1,   1,   1,   1,   2  ])
    best_vals["gamma"]            = np.array([0.007044..., 0.006335...,
                                               0.009004..., 0.003853...,
                                               0.007322...])
    best_vals["n_layers"]         = np.array([4, 4, 4, 4, 3])
    best_vals["hidden_nodes"]     = np.array([950, 978, 997, 985, 616])

    vals = best_vals[
        (best_vals['datacase'] == case) & (best_vals['percentage_train'] == pctg)]

    params = {'n_layers':     int(vals.iloc[0]["n_layers"]),
              'hidden_nodes': int(vals.iloc[0]["hidden_nodes"]),
              'gamma':            vals.iloc[0]["gamma"]}
    return params
```

**解释：**

- **功能**：维护一张预先通过 Optuna 搜索得到的最优超参数查找表，根据 `(datacase, percentage_train)` 组合精确匹配并返回对应的超参数字典。
- **表格结构**：共 5 条记录，覆盖 `datacase ∈ {1, 2}` 与 `percentage_train ∈ {0.1, 0.2, 0.4, 0.8}` 的典型组合。
- **超参数含义**：
  - `n_layers`：网络隐藏层数量 \(n_{\text{layers}} \in \{3, 4\}\)；
  - `hidden_nodes`：每层隐含神经元数量 \(H \in [616, 997]\)；
  - `gamma`：差分正则化系数 \(\gamma \in [0.003, 0.009]\)。
- **查找逻辑**：对 DataFrame 使用布尔掩码筛选同时满足 `datacase==case` 和 `percentage_train==pctg` 的行，取第一条记录的三个超参数值，封装为字典返回：

\[
\text{params} = \left\{\ n_{\text{layers}},\ H,\ \gamma\ \right\}
\]

---

*文档结束*
