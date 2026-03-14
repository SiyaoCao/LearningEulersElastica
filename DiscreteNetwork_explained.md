# DiscreteNetmork_main.py 代码逐行解释

## 代码目的概述

本脚本实现了一套基于深度神经网络（MLP 与残差网络 ResNet）的**离散弹性曲线轨迹预测**框架。其核心目标是：给定初始几何参数与边界条件，训练神经网络代替传统数值求解器，快速预测离散弹性杆在约束下的空间形态（节点位置序列）。主要流程包括：①用 `optuna` 进行贝叶斯超参数搜索，自动选取最优网络结构与训练参数；②用选定超参数训练最终模型，或直接加载预训练权重；③对测试集进行预测并可视化结果；④测量推理延迟，验证实时性能。该框架可服务于柔性机器人、弹性力学等领域的正/逆运动学快速求解。

---

## 目录

- [一、环境准备与依赖导入（第 1–30 行）](#一环境准备与依赖导入第-130-行)
- [二、绘图参数配置（第 37–42 行）](#二绘图参数配置第-3742-行)
- [三、全局精度与设备配置（第 48–55 行）](#三全局精度与设备配置第-4855-行)
- [四、用户输入与随机种子固定（第 61–68 行）](#四用户输入与随机种子固定第-6168-行)
- [五、`approximate_curve` 神经网络类（第 70–102 行）](#五approximate_curve-神经网络类第-70102-行)
- [六、加载数据节点信息（第 104 行）](#六加载数据节点信息第-104-行)
- [七、`define_model` 超参数搜索模型构建函数（第 106–117 行）](#七define_model-超参数搜索模型构建函数第-106117-行)
- [八、`objective` Optuna 目标函数（第 121–185 行）](#八objective-optuna-目标函数第-121185-行)
- [九、Optuna 超参数搜索（第 187–198 行）](#九optuna-超参数搜索第-187198-行)
- [十、超参数选取逻辑（第 200–221 行）](#十超参数选取逻辑第-200221-行)
- [十一、`define_best_model` 最优模型构建函数（第 222–239 行）](#十一define_best_model-最优模型构建函数第-222239-行)
- [十二、训练配置参数（第 242–254 行）](#十二训练配置参数第-242254-行)
- [十三、训练或加载预训练模型（第 256–268 行）](#十三训练或加载预训练模型第-256268-行)
- [十四、结果评估与可视化（第 270–277 行）](#十四结果评估与可视化第-270277-行)
- [十五、推理耗时测量（第 283–291 行）](#十五推理耗时测量第-283291-行)
- [附录 A：网络结构总览](#附录-a网络结构总览)
- [附录 B：`Scripts/GetData.py` 数据加载模块](#附录-bscriptsgetdatapy-数据加载模块)
  - [B.1 `loadData` 函数](#b1-loaddata-函数)
  - [B.2 `dataset` 数据集类](#b2-dataset-数据集类)
  - [B.3 `getDataLoaders` 函数](#b3-getdataloaders-函数)
- [附录 C：`Scripts/Training.py` 训练模块](#附录-cscriptstrainingpy-训练模块)
  - [C.1 `EarlyStopper` 早停类](#c1-earlystopper-早停类)
  - [C.2 `train` 训练函数](#c2-train-训练函数)
- [附录 D：`Scripts/PlotResults.py` 可视化模块](#附录-dscriptsplotresultspy-可视化模块)
- [附录 E：`Scripts/SavedParameters.py` 超参数存储模块](#附录-escriptssavedparameterspy-超参数存储模块)

---

> 本文档按主函数的运行顺序，逐段解释 `DiscreteNetmork_main.py` 中的每一行代码。涉及计算的部分均附有对应的数学公式，行号在前，公式在后。

---

## 一、环境准备与依赖导入（第 1–30 行）

```python
1.  #!/usr/bin/env python
2.  # coding: utf-8
```

- **第 1 行**：Shebang 行，声明使用系统 Python 解释器运行脚本。
- **第 2 行**：声明源文件编码为 UTF-8，保证中文等非 ASCII 字符可正常处理。

```python
8.  get_ipython().system('pip install optuna')
9.  import optuna
10. import torch
11. import random
12. import torch.nn as nn
13. import numpy as np
14. import matplotlib.pyplot as plt
15. import matplotlib
16. import torch.nn.functional as F
17. from csv import writer
18. import seaborn as sns
19. import os
20. os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

- **第 8 行**：在 Jupyter Notebook 环境中执行 shell 命令，安装 `optuna` 超参数优化库。
- **第 9–19 行**：导入所有必要的第三方库：
  - `optuna`：贝叶斯超参数优化框架。
  - `torch`：PyTorch 深度学习框架。
  - `random`：Python 标准随机数库（用于可重复性）。
  - `torch.nn`：PyTorch 神经网络模块。
  - `numpy`：数值计算库。
  - `matplotlib.pyplot`、`matplotlib`：绘图库。
  - `torch.nn.functional`：PyTorch 函数式神经网络接口（含激活函数）。
  - `csv.writer`：CSV 文件写入工具。
  - `seaborn`：基于 matplotlib 的统计可视化库。
  - `os`：操作系统接口。
- **第 20 行**：设置环境变量，解决 Intel MKL 与其他数学库并存时可能出现的动态链接库冲突。

```python
26. from Scripts.GetData import getDataLoaders, loadData
27. from Scripts.Training import train
28. from Scripts.PlotResults import plotResults
29. from Scripts.SavedParameters import hyperparams
30. import pandas as pd
```

- **第 26–29 行**：从本地 `Scripts` 包导入自定义模块：
  - `getDataLoaders`：构建训练/验证/测试数据加载器。
  - `loadData`：读取原始数据并返回节点信息。
  - `train`：执行神经网络训练循环。
  - `plotResults`：绘制预测结果与真实值的对比图。
  - `hyperparams`：根据数据案例和训练比例返回预存的最优超参数字典。
- **第 30 行**：导入 `pandas`，用于数据处理（间接依赖）。

---

## 二、绘图参数配置（第 37–42 行）

```python
37. sns.set_style("darkgrid")
38. sns.set(font = "Times New Roman")
39. sns.set_context("paper")
40. plt.rcParams['mathtext.fontset'] = 'cm'
41. plt.rcParams['font.family'] = 'STIXGeneral'
42. plt_kws = {"rasterized": True}
```

- **第 37 行**：将 seaborn 风格设为深色网格背景。
- **第 38 行**：全局字体设为 Times New Roman（学术论文常用衬线字体）。
- **第 39 行**：将绘图上下文设为 `"paper"`，缩小元素尺寸以适配期刊图幅。
- **第 40–41 行**：配置 matplotlib 数学文本使用 Computer Modern 字体集，正文字体使用 STIXGeneral，与 LaTeX 风格保持一致。
- **第 42 行**：定义关键字参数字典 `plt_kws`，在生成大量散点/曲线时开启栅格化以减小文件体积。

---

## 三、全局精度与设备配置（第 48–55 行）

```python
48. torch.set_default_dtype(torch.float32)
```

- **第 48 行**：将 PyTorch 全局默认张量类型设为单精度浮点数（float32），在精度与计算效率之间取得平衡。

```python
54. device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
55. print(device)
```

- **第 54–55 行**：自动检测是否有可用 GPU。若有，则使用编号为 0 的 CUDA 设备以加速训练；否则退而使用 CPU。

---

## 四、用户输入与随机种子固定（第 61–68 行）

```python
61. datacase = int(input("Which datacase do you want to work with?\n"))
62. percentage_train = float(input("Which percentage of the dataset do you want to use for training? Choose among 0.1,0.2,0.4,0.8\n"))
64. print(f"\n\n Case with percentage_train={percentage_train} and datacase={datacase}\n\n")
66. torch.manual_seed(1)
67. np.random.seed(1)
68. random.seed(1)
```

- **第 61 行**：读取用户输入的数据案例编号（`datacase`），决定使用哪组边界条件数据（如 `1` 表示两端均给定边界条件，`2` 表示仅右端给定边界条件）。
- **第 62 行**：读取用于训练的数据集比例（`percentage_train`），可选 0.1、0.2、0.4、0.8。
- **第 64 行**：打印当前运行配置，便于日志追踪。
- **第 66–68 行**：分别为 PyTorch、NumPy 和 Python 内置 `random` 模块固定随机种子为 1，保证实验结果的可重复性。

---

## 五、`approximate_curve` 神经网络类（第 70–102 行）

### 5.1 类定义与 `__init__` 方法（第 70–88 行）

```python
70. class approximate_curve(nn.Module):
71.     def __init__(self, is_res=True, normalize=True, act_name='tanh',
                    nlayers=3, hidden_nodes=50, output_dim=204):
72.         super().__init__()
74.         torch.manual_seed(1)
75.         np.random.seed(1)
76.         random.seed(1)
77.         self.act_dict = {"tanh":   lambda x: torch.tanh(x),
78.                          "sigmoid": lambda x: torch.sigmoid(x),
79.                          "swish":   lambda x: x * torch.sigmoid(x),
80.                          "relu":    lambda x: torch.relu(x),
81.                          "lrelu":   lambda x: F.leaky_relu(x)}
82.         self.is_norm = normalize
83.         self.is_res  = is_res
84.         self.act     = self.act_dict[act_name]
85.         self.nlayers = nlayers
86.         self.first   = nn.Linear(8, hidden_nodes)
87.         self.linears = nn.ModuleList([nn.Linear(hidden_nodes, hidden_nodes)
                                         for i in range(self.nlayers)])
88.         self.last    = nn.Linear(hidden_nodes, output_dim)
```

- **第 70 行**：定义 `approximate_curve` 类，继承自 `nn.Module`，是 PyTorch 所有神经网络模型的基类。
- **第 71 行**：构造函数，参数含义：
  - `is_res`：是否使用残差连接（ResNet 风格）。
  - `normalize`：是否对输入进行归一化。
  - `act_name`：激活函数名称。
  - `nlayers`：隐藏层数量。
  - `hidden_nodes`：每个隐藏层的神经元数。
  - `output_dim`：输出维度，等于 $4(N-2)$，其中 $N$ 为网格节点总数。
- **第 72 行**：调用父类初始化，注册参数与子模块。
- **第 74–76 行**：在模型初始化时再次固定随机种子，确保每次构建模型时权重初始化一致。
- **第 77–81 行**：构建激活函数字典，可选：
  - `tanh`：双曲正切，$\sigma(x) = \tanh(x)$；
  - `sigmoid`：$\sigma(x) = \dfrac{1}{1+e^{-x}}$；
  - `swish`：$\sigma(x) = x \cdot \dfrac{1}{1+e^{-x}}$；
  - `relu`：$\sigma(x) = \max(0, x)$；
  - `lrelu`（Leaky ReLU）：$\sigma(x) = \max(\alpha x, x)$，$\alpha$ 为小正数（默认 0.01）。
- **第 82–85 行**：保存归一化开关、残差开关、激活函数选择和层数到实例属性。
- **第 86 行**：定义第一层全连接层，将 8 维输入映射到 `hidden_nodes` 维隐空间：

$$W^{(1)} \in \mathbb{R}^{H \times 8}, \quad b^{(1)} \in \mathbb{R}^{H}$$

- **第 87 行**：构建 `nlayers` 个隐藏层组成的 `ModuleList`，每层均为 $H \to H$ 的全连接层：

$$W^{(l)} \in \mathbb{R}^{H \times H}, \quad b^{(l)} \in \mathbb{R}^{H}, \quad l = 1, \ldots, L$$

- **第 88 行**：定义输出层，将隐空间映射到 $4(N-2)$ 维输出：

$$W^{(\text{out})} \in \mathbb{R}^{4(N-2) \times H}, \quad b^{(\text{out})} \in \mathbb{R}^{4(N-2)}$$

---

### 5.2 `forward` 前向传播方法（第 90–102 行）

```python
90.  def forward(self, x):
92.      if self.is_norm:
93.          x[:,0] = (x[:,0] - 1.5) / 1.5
94.          x[:,4] = (x[:,4] - 1.5) / 1.5
95.      x = self.act(self.first(x))
96.      for i in range(self.nlayers):
97.          if self.is_res:
98.              x = x + self.act(self.linears[i](x))
99.          else:
100.             x = self.act(self.linears[i](x))
102.     return self.last(x)
```

- **第 92–94 行**：若 `is_norm=True`，对输入 $\mathbf{x}$ 的第 0 列和第 4 列（对应两端边界处的某特定物理量）做 Min-Max 归一化，使其缩放到 $[-1, 1]$ 区间：

$$\tilde{x}_i = \frac{x_i - 1.5}{1.5}, \quad i \in \{0, 4\}$$

- **第 95 行**：通过第一层全连接加激活函数，将归一化后的 8 维输入 $\tilde{\mathbf{x}}$ 映射到 $H$ 维隐向量 $\mathbf{h}^{(0)}$：

$$\mathbf{h}^{(0)} = \sigma\!\left(W^{(1)} \tilde{\mathbf{x}} + \mathbf{b}^{(1)}\right)$$

- **第 96–100 行**：逐层通过 $L$ 个隐藏层。有两种模式：

  **普通 MLP（`is_res=False`）**（第 100 行）：

  $$\mathbf{h}^{(l+1)} = \sigma\!\left(W^{(l+1)} \mathbf{h}^{(l)} + \mathbf{b}^{(l+1)}\right), \quad l = 0, 1, \ldots, L-1$$

  **残差网络（`is_res=True`，第 98 行）**：在普通前向计算的基础上加上恒等跳跃连接，缓解深层网络梯度消失问题：

  $$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \sigma\!\left(W^{(l+1)} \mathbf{h}^{(l)} + \mathbf{b}^{(l+1)}\right), \quad l = 0, 1, \ldots, L-1$$

- **第 102 行**：通过输出层（线性，无激活函数）将最终隐向量 $\mathbf{h}^{(L)}$ 映射为预测的离散曲线坐标向量：

$$\hat{\mathbf{y}} = W^{(\text{out})} \mathbf{h}^{(L)} + \mathbf{b}^{(\text{out})}, \quad \hat{\mathbf{y}} \in \mathbb{R}^{4(N-2)}$$

  其中 $\hat{\mathbf{y}}$ 包含网格内部 $N-2$ 个节点处的 4 个物理量（例如位置 $x$、$y$ 及两个切向分量）的预测值。整个网络的作用可概括为：给定 8 维边界条件 $\boldsymbol{\chi}$，输出对离散曲线内节点状态的近似：

$$\hat{\mathbf{u}}_b \approx \mathcal{N}_\theta(\boldsymbol{\chi})$$

---

## 六、加载数据节点信息（第 104 行）

```python
104. num_nodes, _, _ = loadData(datacase)
```

- **第 104 行**：调用 `loadData` 函数，传入数据案例编号，返回网格节点数 `num_nodes`（以及其他被忽略的返回值）。`num_nodes` 决定了网络输出维度 $4(N-2)$，即内部节点数乘以每节点的状态变量数。

---

## 七、`define_model` 超参数搜索模型构建函数（第 106–117 行）

```python
106. def define_model(trial):
107.     torch.manual_seed(1)
108.     np.random.seed(1)
109.     random.seed(1)
110.     is_res = False
111.     normalize = True
112.     act_name = "tanh"
113.     nlayers = trial.suggest_int("n_layers", 0, 10)
114.     hidden_nodes = trial.suggest_int("hidden_nodes", 10, 1000)
115.     model = approximate_curve(is_res, normalize, act_name, nlayers, hidden_nodes,
                                   output_dim=int(4*(num_nodes-2)))
117.     return model
```

- **第 106 行**：定义模型构建函数，接受 Optuna `trial` 对象，用于超参数搜索时按 trial 建立不同配置的模型。
- **第 107–109 行**：每次构建新模型前固定随机种子，确保不同 trial 的唯一差异来自超参数而非随机初始化。
- **第 110–112 行**：固定超参数：不使用残差连接、开启输入归一化、激活函数为 `tanh`。
- **第 113 行**：由 Optuna 在整数范围 $[0, 10]$ 内搜索隐藏层数量 `n_layers`：

$$L \in \{0, 1, 2, \ldots, 10\}$$

- **第 114 行**：由 Optuna 在整数范围 $[10, 1000]$ 内搜索每层神经元数 `hidden_nodes`：

$$H \in \{10, 11, \ldots, 1000\}$$

- **第 115–116 行**：用搜索到的超参数实例化 `approximate_curve` 模型，输出维度固定为 $4(N-2)$，并返回该模型。

---

## 八、`objective` Optuna 目标函数（第 121–185 行）

```python
121. def objective(trial):
123.     torch.manual_seed(1)
124.     np.random.seed(1)
125.     random.seed(1)
128.     model = define_model(trial)
129.     model.to(device)
131.     lr = 1e-3
132.     weight_decay = 0
133.     gamma = trial.suggest_float("gamma", 0, 1e-2)
134.     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
136.     criterion = nn.MSELoss()
138.     batch_size = 32
139.     _, _, _, _, x_val, y_val, trainloader, _, valloader = getDataLoaders(
                batch_size, datacase, percentage_train)
```

- **第 121 行**：定义 Optuna 目标函数，Optuna 将最小化此函数的返回值（验证集 MSE）。
- **第 123–125 行**：固定随机种子。
- **第 128–129 行**：构建当前 trial 对应的模型并迁移到目标设备。
- **第 131–132 行**：固定学习率 $\eta = 10^{-3}$ 和权重衰减系数 $\lambda = 0$。
- **第 133 行**：由 Optuna 在连续区间 $[0, 10^{-2}]$ 内搜索学习率调度衰减系数 `gamma`：

$$\gamma \in [0,\, 10^{-2}]$$

- **第 134 行**：实例化 Adam 优化器，其参数更新规则为：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

  其中 $g_t$ 为当前梯度，$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$（PyTorch 默认值），$\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$ 为偏差修正项。

- **第 136 行**：实例化均方误差损失函数（MSELoss）：

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left\| \hat{\mathbf{y}}_i - \mathbf{y}_i \right\|^2$$

  其中 $\hat{\mathbf{y}}_i = \mathcal{N}_\theta(\mathbf{x}_i)$ 为模型预测，$\mathbf{y}_i$ 为真实离散曲线状态向量。

- **第 138–139 行**：设置批大小为 32，调用 `getDataLoaders` 获取验证集输入 `x_val`、验证集标签 `y_val`、训练数据加载器 `trainloader` 和验证数据加载器 `valloader`。

```python
141.     print("Current test with :\n\n")
142.     for key, value in trial.params.items():
143.         print("    {}: {}".format(key, value))
146.     epochs = 300
147.     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=int(0.45*epochs), gamma=0.1)
149.     loss = train(model, gamma, criterion, scheduler, optimizer, epochs,
                     trainloader, valloader, device)
150.     print('Loss ', loss.item())
151.     error = 1000
```

- **第 141–143 行**：打印当前 trial 的超参数组合，便于监控。
- **第 146 行**：设置最大训练轮数 $E = 300$。
- **第 147 行**：实例化步进学习率调度器（StepLR）。每 $\lfloor 0.45 \times 300 \rfloor = 135$ 个 epoch 将学习率乘以 0.1：

$$\eta_t = \eta_0 \cdot \gamma_{\text{step}}^{\lfloor t / T_{\text{step}} \rfloor}, \quad T_{\text{step}} = 135, \quad \gamma_{\text{step}} = 0.1$$

- **第 149 行**：调用 `train` 函数，执行完整的训练循环（前向传播、损失计算、反向传播、参数更新），并在每轮结束后调用调度器更新学习率。返回最终训练损失。
- **第 151 行**：初始化验证误差为 1000（代表未评估时的极大值，若训练发散则保持此默认值）。

```python
153.     if not torch.isnan(loss):
154.         model.eval()
156.         learned_traj = np.zeros_like(y_val)
158.         bcs_val = torch.from_numpy(x_val.astype(np.float32)).to(device)
159.         learned_traj = model(bcs_val).detach().cpu().numpy()
160.         error = np.mean((learned_traj - y_val)**2)
162.         print(f"The error on the validation trajectories is: {error}.")
```

- **第 153 行**：检查训练损失是否为 NaN（非数），若训练发散则跳过评估，保留 `error=1000`。
- **第 154 行**：将模型切换到推理模式，关闭 Dropout/BatchNorm 的训练行为。
- **第 156–159 行**：将验证集输入转为 float32 张量并迁移到 device，通过模型前向传播得到预测的轨迹；`.detach()` 断开计算图，`.cpu().numpy()` 转回 NumPy 数组。
- **第 160 行**：计算验证集均方误差（MSE）：

$$\text{MSE}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \left\| \hat{\mathbf{y}}_i^{(\text{val})} - \mathbf{y}_i^{(\text{val})} \right\|^2$$

```python
165.     if trial.number == 0:
166.         labels = []
167.         for lab, _ in trial.params.items():
168.             labels.append(str(lab))
169.         labels.append("MSE")
170.         with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
171.             writer_object = writer(f_object)
172.             writer_object.writerow(labels)
173.             f_object.close()
175.     results = []
176.     for _, value in trial.params.items():
177.         results.append(str(value))
179.     results.append(error)
181.     with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
182.         writer_object = writer(f_object)
183.         writer_object.writerow(results)
184.         f_object.close()
185.     return error
```

- **第 165–173 行**：仅在第一个 trial（`trial.number == 0`）时写入 CSV 表头（各超参数名称 + `"MSE"`），避免重复写头行。
- **第 175–184 行**：将当前 trial 的超参数值和对应的验证 MSE 追加写入 CSV 文件，文件名包含训练比例百分比（如 `results10_Fig2.csv`）。
- **第 185 行**：返回验证集 MSE 作为 Optuna 的优化目标（越小越好）。

---

## 九、Optuna 超参数搜索（第 187–198 行）

```python
187. optuna_study = input("Do you want to do hyperparameter test? Type yes or no: ")
188. params = {}
189. if optuna_study == "yes":
190.     optuna_study = True
191. else:
192.     optuna_study = False
193. if optuna_study:
194.     study = optuna.create_study(direction="minimize", study_name="Euler Elastica")
195.     study.optimize(objective, n_trials=5)
196.     print("Study statistics: ")
197.     print("Number of finished trials: ", len(study.trials))
198.     params = study.best_params
```

- **第 187–192 行**：询问用户是否启动超参数搜索，将字符串回答转换为布尔值。
- **第 194 行**：创建 Optuna study 对象，目标方向为最小化（`direction="minimize"`），研究名称为 `"Euler Elastica"`（欧拉弹性曲线，揭示了该网络所学习的物理问题背景）。
- **第 195 行**：执行 5 次 trial（`n_trials=5`），每次 trial 由 Optuna 的 TPE（树形 Parzen 估计）贝叶斯采样算法提议新的超参数组合，以近似最小化：

$$\theta^* = \arg\min_{\theta \in \Theta} \text{MSE}_{\text{val}}(\theta)$$

  其中 $\Theta = \{L \in [0,10]\} \times \{H \in [10,1000]\} \times \{\gamma \in [0, 10^{-2}]\}$。

- **第 196–198 行**：打印完成的 trial 数量，并将最佳超参数字典赋给 `params`。

---

## 十、超参数选取逻辑（第 200–221 行）

```python
200. torch.manual_seed(1)
201. np.random.seed(1)
202. random.seed(1)
204. manual_input = False
205. if params == {}:
207.     if manual_input:
208.         print("No parameters have been specified. Let's input them:\n\n")
209.         nlayers      = int(input("How many layers do you want the network to have? "))
210.         hidden_nodes = int(input("How many hidden nodes do you want the network to have? "))
211.         weight_decay = float(input("What weight decay do you want to use? "))
212.         gamma        = float(input("What value do you want for gamma? "))
213.         batch_size   = int(input("What batch size do you want? "))
215.         params = {'n_layers': nlayers, 'hidden_nodes': hidden_nodes, 'gamma': gamma}
218.     else:
220.         params = hyperparams(datacase, percentage_train)
221. print(f'The hyperparameters yelding the best results for this case are: {params}')
```

- **第 200–202 行**：再次固定随机种子，保持后续模型构建的一致性。
- **第 204 行**：`manual_input = False`，默认不启用手动输入模式（此开关供开发者调试用）。
- **第 205 行**：若 `params` 为空字典（即未进行 Optuna 搜索），则需要指定超参数。
- **第 207–215 行**：`manual_input=True` 时，通过交互式输入手动指定 `nlayers`、`hidden_nodes`、`weight_decay`、`gamma`、`batch_size` 并构建 `params` 字典。
- **第 220 行**：`manual_input=False`（默认）时，调用 `hyperparams(datacase, percentage_train)` 从预存最优参数表中查表，返回该数据案例和训练比例下已知最优的超参数字典。
- **第 221 行**：打印最终使用的超参数。

---

## 十一、`define_best_model` 最优模型构建函数（第 222–239 行）

```python
222. def define_best_model():
224.     torch.manual_seed(1)
225.     np.random.seed(1)
226.     random.seed(1)
228.     normalize    = True
229.     act          = "tanh"
230.     nlayers      = params["n_layers"]
231.     hidden_nodes = params["hidden_nodes"]
232.     is_res       = False
234.     print("Nodes: ", hidden_nodes)
236.     model = approximate_curve(is_res, normalize, act, nlayers, hidden_nodes,
                                   int(4*(num_nodes-2)))
238.     return model
239. model = define_best_model()
240. model.to(device)
```

- **第 222 行**：定义最优模型构建函数，使用前面确定的最优超参数字典 `params`（来自 Optuna、手动输入或预存表）。
- **第 224–226 行**：固定随机种子。
- **第 228–232 行**：从 `params` 中提取层数和每层神经元数，固定使用 MLP（`is_res=False`）、输入归一化和 `tanh` 激活函数。
- **第 236–237 行**：实例化最优配置的 `approximate_curve` 模型，输出维度为 $4(N-2)$。
- **第 239–240 行**：调用 `define_best_model()` 构建模型实例，并迁移到目标设备（GPU 或 CPU）。

---

## 十二、训练配置参数（第 242–254 行）

```python
242. TrainMode   = input("Train Mode True or False? Type 0 for False and 1 for True: ") == "1"
243. weight_decay = 0.
244. lr           = 1e-3
245. gamma        = params["gamma"]
246. nlayers      = params["n_layers"]
247. hidden_nodes = params["hidden_nodes"]
248. batch_size   = 32
249. epochs       = 300
250. optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
251. scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs),
                                                    gamma=0.1)
252. criterion    = nn.MSELoss()
253. x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader = \
         getDataLoaders(batch_size, datacase, percentage_train)
254. model.to(device)
```

- **第 242 行**：询问用户是否重新训练模型（`1` = 训练，`0` = 加载预训练权重）。
- **第 243–248 行**：设置完整训练配置：
  - 权重衰减 $\lambda = 0$（不使用 L2 正则化）；
  - 初始学习率 $\eta_0 = 10^{-3}$；
  - 从 `params` 中读取调度衰减系数 $\gamma$、层数和节点数；
  - 批大小 $B = 32$，最大训练轮数 $E = 300$。
- **第 250 行**：实例化 Adam 优化器（更新规则见第 134 行注释）。
- **第 251 行**：实例化 StepLR 调度器：每 $\lfloor 0.45 \times 300 \rfloor = 135$ 轮将学习率乘以固定步进衰减系数 $\gamma_{\text{step}} = 0.1$（注意：此处的 `gamma=0.1` 是**固定的步进衰减系数**，与 Optuna 搜索得到的超参数 `gamma`（第 133/245 行）含义完全不同，后者作为 `train` 函数的正则化或权重缩放系数传入）：

$$\eta_t = \eta_0 \times \gamma_{\text{step}}^{\lfloor t / T_{\text{step}} \rfloor}, \quad \gamma_{\text{step}} = 0.1, \quad T_{\text{step}} = 135$$

- **第 252 行**：实例化 MSELoss：

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left\| \mathcal{N}_\theta(\mathbf{x}_i) - \mathbf{y}_i \right\|^2$$

- **第 253 行**：调用 `getDataLoaders` 获取完整数据集分割：训练集 $(\mathbf{x}_{\text{train}}, \mathbf{y}_{\text{train}})$、测试集 $(\mathbf{x}_{\text{test}}, \mathbf{y}_{\text{test}})$、验证集 $(\mathbf{x}_{\text{val}}, \mathbf{y}_{\text{val}})$ 及对应的 DataLoader。

---

## 十三、训练或加载预训练模型（第 256–268 行）

```python
256. if TrainMode:
257.     loss = train(model, gamma, criterion, scheduler, optimizer, epochs,
                     trainloader, valloader, device)
258.     if datacase == 1:
259.         torch.save(model.state_dict(),
                        f'TrainedModels/BothEnds{percentage_train}data.pt')
260.     if datacase == 2:
261.         torch.save(model.state_dict(),
                        f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt')
262. else:
263.     if datacase == 1:
264.         pretrained_dict = torch.load(
                        f'TrainedModels/BothEnds{percentage_train}data.pt',
                        map_location=device)
265.     if datacase == 2:
266.         pretrained_dict = torch.load(
                        f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt',
                        map_location=device)
267.     model.load_state_dict(pretrained_dict)
268. model.eval()
```

- **第 256–257 行**：若 `TrainMode=True`，调用 `train` 函数执行正式训练，其核心计算流程为：

  对于每个批次 $\mathcal{B} \subset \{1, \ldots, n\}$，执行：

  1. **前向传播**：$\hat{\mathbf{Y}}_\mathcal{B} = \mathcal{N}_\theta(\mathbf{X}_\mathcal{B})$
  2. **损失计算**：$\mathcal{L}_\mathcal{B} = \dfrac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|^2$（批 MSE）
  3. **反向传播**：$g_t = \nabla_\theta \mathcal{L}_\mathcal{B}$
  4. **参数更新**：Adam 步（见第 134 行公式）
  5. **学习率调度**：每 135 个 epoch 触发一次 $\eta \leftarrow 0.1\eta$

- **第 258–261 行**：训练完成后，根据数据案例将模型权重字典（`.state_dict()`）保存到对应的 `.pt` 文件：
  - `datacase=1`（两端均给定边界条件）→ `BothEnds{percentage_train}data.pt`
  - `datacase=2`（仅右端给定边界条件）→ `BothEndsRightEnd{percentage_train}data.pt`

- **第 262–267 行**：若 `TrainMode=False`，从磁盘加载对应的预训练权重字典，并通过 `model.load_state_dict` 将其赋给当前模型，实现迁移推理。`map_location=device` 保证在 CPU 环境下也能加载 GPU 训练的模型。

- **第 268 行**：无论训练还是加载，均切换到推理模式（关闭 Dropout 和 BatchNorm 的随机性）。

---

## 十四、结果评估与可视化（第 270–277 行）

```python
270. torch.manual_seed(1)
271. np.random.seed(1)
272. random.seed(1)
274. model.eval()
276. # printing the accuracies and plotting the results
277. plotResults(model, device, x_train, y_train, x_test, y_test,
                x_val, y_val, num_nodes, datacase, percentage_train,
                gamma, nlayers, hidden_nodes)
```

- **第 270–272 行**：固定随机种子，确保评估过程（如有随机采样操作）可重复。
- **第 274 行**：再次确认模型处于推理模式。
- **第 277 行**：调用 `plotResults` 函数，在训练集、测试集和验证集上评估并可视化模型性能。该函数内部通常会：
  1. 对各数据子集进行前向推理，得到预测曲线 $\hat{\mathbf{Y}}$；
  2. 计算相对误差或均方误差：$\text{err}_{\text{rel}} = \dfrac{\|\hat{\mathbf{Y}} - \mathbf{Y}\|_F}{\|\mathbf{Y}\|_F}$；
  3. 绘制预测轨迹与真实轨迹的对比图，并标注误差统计。

---

## 十五、推理耗时测量（第 283–291 行）

```python
283. import time
284. test_bvs    = torch.from_numpy(x_test.astype(np.float32))
285. initial_time = time.time()
286. preds        = model(test_bvs)
287. final_time   = time.time()
288. total_time   = final_time - initial_time
289. print("Number of trajectories in the test set : ", len(test_bvs))
290. print("Total time to predict test trajectories : ", total_time)
291. print("Average time to predict test trajectories : ", total_time / len(test_bvs))
```

- **第 283 行**：导入 `time` 模块，用于计时。
- **第 284 行**：将测试集输入数组 `x_test` 转换为 float32 PyTorch 张量（未迁移到 GPU，在 CPU 上测量推理时延）。
- **第 285–287 行**：记录推理前后的 Unix 时间戳，计算批量推理总耗时：

$$t_{\text{total}} = t_{\text{final}} - t_{\text{initial}}$$

- **第 288 行**：总耗时 $t_{\text{total}}$（单位：秒）。
- **第 289–291 行**：打印测试集样本数量 $N_{\text{test}}$、总推理时间和平均单次推理时间：

$$\bar{t} = \frac{t_{\text{total}}}{N_{\text{test}}}$$

  该指标量化了模型在替代传统数值求解（如有限元）时的计算加速比：神经网络的 $\bar{t}$ 通常远小于一次有限元迭代的耗时，体现了数据驱动方法在实时预测场景中的优势。

---

## 附录 A：网络结构总览

| 层名称 | 类型 | 输入维度 | 输出维度 | 激活函数 |
|--------|------|----------|----------|---------|
| `first` | Linear | 8 | $H$ | $\tanh$ |
| `linears[0]` ~ `linears[L-1]` | Linear (×$L$) | $H$ | $H$ | $\tanh$（+残差可选）|
| `last` | Linear | $H$ | $4(N-2)$ | 无 |

**输入**（8 维边界条件向量 $\boldsymbol{\chi}$）描述离散弹性曲线两端的几何与力学约束（位置、切向角、曲率等）。**输出**（$4(N-2)$ 维）为内部节点的完整状态向量预测 $\hat{\mathbf{u}}_b$，满足：

$$\hat{\mathbf{u}}_b = \mathcal{N}_\theta(\boldsymbol{\chi}) \approx \mathbf{u}_b^*(\boldsymbol{\chi})$$

其中 $\mathbf{u}_b^*(\boldsymbol{\chi})$ 为对应边界条件下离散欧拉弹性曲线方程的真实解。

---

## 附录 B：`Scripts/GetData.py` 数据加载模块

本模块负责从磁盘读取原始数据集、构建 PyTorch `Dataset` 对象以及将数据划分为训练/验证/测试三个子集并封装为 `DataLoader`。

---

### B.1 `loadData` 函数

```python
 1. import numpy as np
 2. import os
 3. import torch
 4. from torch.utils.data import Dataset, DataLoader
 5. import random
 7. def loadData(datacase = 1):
 9.     original_dir = os.getcwd()
10.     root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
11.     os.chdir(root_dir+"/DataSets")
13.     both_ends_360_sol = open("both_ends.txt", "r")
14.     trajectoriesload_b_360 = np.loadtxt(both_ends_360_sol)
16.     right_end_360_sol = open("right_end.txt", "r")
17.     trajectoriesload_r_360 = np.loadtxt(right_end_360_sol)
19.     if datacase == 1:
20.         trajectories_train = trajectoriesload_b_360
21.         trajectories_test = trajectories_train
22.     elif datacase == 2:
23.         trajectories_train = np.concatenate((trajectoriesload_b_360, trajectoriesload_r_360), axis = 0)
24.         trajectories_test = trajectories_train
26.         print("Warning! Must be an integer between 1 and 3")
28.     num_nodes = trajectories_train.shape[1]//4
29.     os.chdir(original_dir)
30.     return num_nodes, trajectories_train, trajectories_test
```

- **第 1–5 行**：导入依赖库：`numpy` 用于矩阵运算，`os` 用于路径切换，`torch` 及 `Dataset`/`DataLoader` 用于后续数据封装，`random` 用于随机打乱。
- **第 9–11 行**：记录当前工作目录，向上一级找到项目根目录，然后切换到 `DataSets` 子目录以读取数据文件。
- **第 13–17 行**：分别读取 `both_ends.txt`（两端均指定边界条件的解集）和 `right_end.txt`（仅右端指定边界条件的解集），以 `numpy` 二维数组形式加载。每行代表一条离散弹性曲线的完整状态向量，每条曲线有 $N$ 个节点，每个节点存储 4 个量 $(q_x, q_y, q_x', q_y')$，因此每行共 $4N$ 列。
- **第 19–24 行**：根据 `datacase` 选择数据集：
  - `datacase=1`：仅使用两端数据集；
  - `datacase=2`：将两端数据集与右端数据集按行拼接：

$$\mathbf{D}_{\text{train}} = \begin{bmatrix} \mathbf{D}_{\text{both}} \\ \mathbf{D}_{\text{right}} \end{bmatrix} \in \mathbb{R}^{(n_1 + n_2) \times 4N}$$

  其余值触发警告打印（注：源码警告文本写作 "between 1 and 3"，实际有效值为 1 与 2）。

- **第 28 行**：从数据矩阵的列数推导节点总数：

$$N = \left\lfloor \frac{\text{列数}}{4} \right\rfloor$$

- **第 29–30 行**：恢复原始工作目录，返回节点数 $N$、训练数据矩阵和测试数据矩阵（在此函数中两者相同，后续由 `getDataLoaders` 负责划分）。

---

### B.2 `dataset` 数据集类

```python
32. class dataset(Dataset):
33.   def __init__(self, x, y):
35.     self.bcs = torch.from_numpy(x.astype(np.float32))
36.     self.internal_node_outputs = torch.from_numpy(y.astype(np.float32))
37.     self.length = x.shape[0]
39.   def __getitem__(self, idx):
40.     return self.bcs[idx], self.internal_node_outputs[idx]
41.   def __len__(self):
42.     return self.length
```

- **第 32 行**：定义 `dataset` 类，继承自 PyTorch `Dataset`，实现标准的数据集接口以支持 `DataLoader` 的批采样。
- **第 33–37 行**（`__init__` 方法）：
  - 将输入边界条件数组 $\mathbf{X} \in \mathbb{R}^{n \times 8}$ 和标签数组 $\mathbf{Y} \in \mathbb{R}^{n \times 4(N-2)}$ 分别转换为 float32 类型的 PyTorch 张量，存储为实例属性；
  - `self.length` 保存样本总数 $n$。
- **第 39–40 行**（`__getitem__` 方法）：根据索引 $i$ 返回第 $i$ 个样本的输入-输出对 $(\mathbf{x}_i, \mathbf{y}_i)$，供 `DataLoader` 拼装批数据。
- **第 41–42 行**（`__len__` 方法）：返回数据集总样本数 $n$，供 `DataLoader` 计算批次数量使用：

$$n_{\text{batch}} = \left\lceil \frac{n}{B} \right\rceil$$

其中 $B$ 为批大小。

---

### B.3 `getDataLoaders` 函数

```python
45. def getDataLoaders(batch_size, datacase, percentage_train):
47.     torch.manual_seed(1)
48.     np.random.seed(1)
49.     random.seed(1)
51.     _, data_train, data_test = loadData(datacase)
52.     x_full_train = np.concatenate((data_train[:,:4], data_train[:,-4:]), axis=1)
53.     y_full_train = data_train[:,4:-4]
54.     N = len(x_full_train)
55.     NTrain = int(percentage_train*N)
57.     idx_shuffle_train = np.arange(N)
58.     random.shuffle(idx_shuffle_train)
60.     x_full_train = x_full_train[idx_shuffle_train]
61.     y_full_train = y_full_train[idx_shuffle_train]
63.     x_full_test = np.concatenate((data_test[:,:4], data_test[:,-4:]), axis=1)
64.     y_full_test = data_test[:,4:-4]
66.     x_full_test = x_full_test[idx_shuffle_train]
67.     y_full_test = y_full_test[idx_shuffle_train]
69.     fact = 0.1
70.     if percentage_train==0.8:
71.         fact = 0.1
72.     elif percentage_train==0.7:
73.         fact = 0.15
74.     else:
75.         fact = 0.2
77.     x_train, y_train = x_full_train[:NTrain], y_full_train[:NTrain]
79.     Number_Test_Points = int(fact*N)
80.     x_test, y_test = x_full_test[NTrain:NTrain+Number_Test_Points], y_full_test[NTrain:NTrain+Number_Test_Points]
81.     x_val, y_val   = x_full_test[NTrain+Number_Test_Points:NTrain+2*Number_Test_Points], y_full_test[NTrain+Number_Test_Points:NTrain+2*Number_Test_Points]
87.     trainset = dataset(x_train, y_train)
88.     testset  = dataset(x_test, y_test)
89.     valset   = dataset(x_val, y_val)
91.     trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
92.     testloader  = DataLoader(testset,  batch_size=len(x_test), shuffle=True)
93.     valloader   = DataLoader(valset,   batch_size=len(x_val),  shuffle=True)
95.     return x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader
```

- **第 47–49 行**：固定随机种子，保证每次调用得到相同的随机打乱顺序。
- **第 51–53 行**：调用 `loadData` 获取完整数据矩阵，然后从中分离输入特征 $\mathbf{X}$ 和标签 $\mathbf{Y}$：
  - 输入特征：取每行的前 4 列（左端节点状态）和后 4 列（右端节点状态）拼接，形成 8 维边界条件向量：

$$\mathbf{x}_i = \left[q_x^{(0)},\, q_y^{(0)},\, q_x^{\prime(0)},\, q_y^{\prime(0)},\, q_x^{(N-1)},\, q_y^{(N-1)},\, q_x^{\prime(N-1)},\, q_y^{\prime(N-1)}\right] \in \mathbb{R}^8$$

  - 标签：取去掉首尾各 4 列后的中间 $4(N-2)$ 列，对应内部节点状态：

$$\mathbf{y}_i = \left[q_x^{(1)},\, q_y^{(1)},\, q_x^{\prime(1)},\, q_y^{\prime(1)},\, \ldots,\, q_x^{(N-2)},\, q_y^{(N-2)},\, q_x^{\prime(N-2)},\, q_y^{\prime(N-2)}\right] \in \mathbb{R}^{4(N-2)}$$

- **第 54–55 行**：计算总样本数 $n$ 和训练集大小：

$$N_{\text{train}} = \lfloor p_{\text{train}} \times n \rfloor$$

其中 $p_{\text{train}} \in \{0.1, 0.2, 0.4, 0.7, 0.8\}$ 为训练集比例（主函数通常使用 0.1、0.2、0.4、0.8 四个选项；代码内部亦处理 0.7 的情况）。

- **第 57–61 行**：生成 $[0, n)$ 的整数索引，随机打乱后同步应用于输入和标签，保证 $\mathbf{x}_i$ 与 $\mathbf{y}_i$ 的对应关系不变。
- **第 63–67 行**：对测试数据矩阵做与训练数据相同的特征/标签分离和同步打乱，保证训练集与测试集的样本顺序一致。
- **第 69–75 行**：根据训练比例自适应设置测试/验证集比例系数 $f$：
  - $p_{\text{train}} = 0.8$ 时，$f = 0.1$；
  - $p_{\text{train}} = 0.7$ 时，$f = 0.15$；
  - 其余情况，$f = 0.2$。
- **第 77–81 行**：按索引切片划分三个子集：

$$N_{\text{test}} = N_{\text{val}} = \lfloor f \times n \rfloor$$

$$\mathbf{X}_{\text{train}} = \mathbf{X}_{[0: N_{\text{train}}]}, \quad \mathbf{X}_{\text{test}} = \mathbf{X}_{[N_{\text{train}}: N_{\text{train}}+N_{\text{test}}]}, \quad \mathbf{X}_{\text{val}} = \mathbf{X}_{[N_{\text{train}}+N_{\text{test}}: N_{\text{train}}+2N_{\text{test}}]}$$

- **第 87–93 行**：将各子集封装为 `dataset` 对象，再包装为 `DataLoader`。训练集使用调用方传入的 `batch_size` 参数（主函数中为 32）并启用随机打乱（`shuffle=True`）；测试集和验证集使用全集大小的单批以方便整体评估。

---

## 附录 C：`Scripts/Training.py` 训练模块

本模块定义了 `EarlyStopper` 早停工具类和核心 `train` 训练函数。

---

### C.1 `EarlyStopper` 早停类

```python
 6. class EarlyStopper:
 7.     def __init__(self, patience=1, min_delta=0):
 8.         self.patience = patience
 9.         self.min_delta = min_delta
10.         self.counter = 0
11.         self.min_validation_loss = float('inf')
13.     def early_stop(self, validation_loss):
14.         if validation_loss < self.min_validation_loss:
15.             self.min_validation_loss = validation_loss
16.             self.counter = 0
17.         elif validation_loss > (self.min_validation_loss + self.min_delta):
18.             self.counter += 1
19.             if self.counter >= self.patience:
20.                 return True
21.         return False
```

- **第 6 行**：定义早停工具类，用于在验证损失不再改善时提前终止训练，防止过拟合。
- **第 7–11 行**（`__init__` 方法）：
  - `patience`：允许验证损失连续不改善的最大轮数；
  - `min_delta`：判断"改善"所要求的最小下降量；
  - `counter`：记录当前连续未改善的轮数；
  - `min_validation_loss`：历史最优验证损失，初始化为 $+\infty$。
- **第 13–21 行**（`early_stop` 方法）：
  - 若当前验证损失 $\mathcal{L}_{\text{val}}^{(t)} < \mathcal{L}_{\text{val}}^{*}$（历史最优），则更新最优记录并重置计数器；
  - 若当前验证损失超过历史最优加容差阈值：

$$\mathcal{L}_{\text{val}}^{(t)} > \mathcal{L}_{\text{val}}^{*} + \delta_{\min}$$

则计数器加一；当计数器达到耐心值 $P$ 时，返回 `True` 触发早停：

$$\text{stop} = \mathbf{1}\!\left[\text{counter} \geq P\right]$$

> **注**：在本脚本中早停逻辑被注释掉（`train` 函数第 81–84 行），实际训练始终运行完整的 $E=300$ 个 epoch。

---

### C.2 `train` 训练函数

```python
23. def train(model, gamma, criterion, scheduler, optimizer, epochs, trainloader, valloader, device):
25.     torch.manual_seed(1)
26.     np.random.seed(1)
27.     random.seed(1)
29.     early_stopper = EarlyStopper(patience=100, min_delta=0)
30.     losses = []
31.     losses_val = []
32.     for epoch in range(epochs):
34.         train_loss = 0.
35.         counter = 0.
37.         for _, data in enumerate(trainloader):
38.             inputs, labels = data[0].to(device), data[1].to(device)
39.             optimizer.zero_grad()
40.             predicted = model(inputs)
42.             loss = criterion(predicted, labels)
44.             predicted = torch.cat((inputs[:,:4], predicted, inputs[:,4:]), dim=1)
45.             labels    = torch.cat((inputs[:,:4], labels,    inputs[:,4:]), dim=1)
47.             predicted_first   = predicted[:,:-4]
48.             predicted_forward = predicted[:,4:]
50.             labels_first   = labels[:,:-4]
51.             labels_forward = labels[:,4:]
53.             diff_predicted = predicted_forward - predicted_first
54.             diff_labels    = labels_forward    - labels_first
56.             loss += gamma * criterion(diff_predicted, diff_labels)
58.             train_loss += loss.item()
60.             loss.backward()
61.             optimizer.step()
62.             counter += 1
64.         avg_train_loss = train_loss / counter
65.         losses.append(avg_train_loss)
67.         model.eval()
68.         with torch.no_grad():
69.             data = next(iter(valloader))
70.             inputs, labels = data[0].to(device), data[1].to(device)
71.             predicted = model(inputs)
74.             val_loss = criterion(predicted, labels)
75.         model.train()
76.         losses_val.append(val_loss.item())
78.         if epoch % int(0.1*epochs) == 0 and epoch > 1:
79.             print(f"The average loss in epoch {epoch+1} is ", avg_train_loss)
87.         scheduler.step()
88.     print('Training Done')
98.     return loss
```

- **第 25–27 行**：固定随机种子，保证批次采样顺序可重复。
- **第 29–31 行**：实例化早停器（实际未启用），初始化训练损失和验证损失历史列表。
- **第 32 行**：外层循环，遍历 $E$ 个训练轮次（epoch）。
- **第 34–35 行**：初始化本轮次的累积批损失和批次计数器。
- **第 37–62 行**：内层批次循环，对每个 mini-batch $\mathcal{B}$ 执行完整的训练步骤：
  1. **数据迁移**（第 38 行）：将批次输入 $\mathbf{X}_\mathcal{B} \in \mathbb{R}^{B \times 8}$ 和标签 $\mathbf{Y}_\mathcal{B} \in \mathbb{R}^{B \times 4(N-2)}$ 迁移到目标设备。
  2. **梯度清零**（第 39 行）：$\nabla_\theta \leftarrow \mathbf{0}$，防止梯度累积。
  3. **前向传播**（第 40 行）：$\hat{\mathbf{Y}}_\mathcal{B} = \mathcal{N}_\theta(\mathbf{X}_\mathcal{B})$。
  4. **基础 MSE 损失**（第 42 行）：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i \in \mathcal{B}} \left\| \hat{\mathbf{y}}_i - \mathbf{y}_i \right\|^2$$

  5. **完整轨迹拼接**（第 44–45 行）：将预测和真实标签的内部节点状态与边界条件拼接，还原为长度 $4N$ 的完整离散曲线状态向量：

$$\tilde{\mathbf{y}}_i = \left[\mathbf{x}_i^{(L)},\; \hat{\mathbf{y}}_i,\; \mathbf{x}_i^{(R)}\right] \in \mathbb{R}^{4N}, \quad i \in \mathcal{B}$$

其中 $\mathbf{x}_i^{(L)} = \mathbf{x}_i[:4]$（左端边界），$\mathbf{x}_i^{(R)} = \mathbf{x}_i[4:]$（右端边界）。

  6. **差分正则化损失**（第 47–56 行）：计算相邻节点的离散差分并对比真实差分，以惩罚不光滑的预测：

$$\mathbf{d}_i^{\text{pred}} = \tilde{\mathbf{y}}_i[4:] - \tilde{\mathbf{y}}_i[:-4], \quad \mathbf{d}_i^{\text{true}} = \tilde{\mathbf{y}}_i^*[4:] - \tilde{\mathbf{y}}_i^*[:-4]$$

$$\mathcal{L}_{\text{diff}} = \frac{1}{B} \sum_{i \in \mathcal{B}} \left\| \mathbf{d}_i^{\text{pred}} - \mathbf{d}_i^{\text{true}} \right\|^2$$

总损失为 MSE 损失与差分正则化损失之和：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot \mathcal{L}_{\text{diff}}$$

其中 $\gamma \in [0, 10^{-2}]$ 为超参数，控制差分正则化的强度。

  7. **反向传播**（第 60 行）：$\nabla_\theta \mathcal{L}_{\text{total}} = \frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta}$（自动微分）。
  8. **参数更新**（第 61 行）：Adam 步骤（参数更新公式见主函数第 134 行）。

- **第 64–65 行**：计算并记录本轮次的平均训练损失：

$$\bar{\mathcal{L}}_{\text{train}}^{(e)} = \frac{1}{n_{\text{batch}}} \sum_{\mathcal{B}} \mathcal{L}_{\text{total}}(\mathcal{B})$$

- **第 67–76 行**：切换推理模式，对验证集整批进行前向推理，计算验证 MSE 并记录；然后切回训练模式：

$$\mathcal{L}_{\text{val}}^{(e)} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \left\| \mathcal{N}_\theta(\mathbf{x}_i^{\text{val}}) - \mathbf{y}_i^{\text{val}} \right\|^2$$

- **第 78–79 行**：每隔 $\lfloor 0.1 \times E \rfloor$ 个 epoch 打印一次当前平均训练损失（在主函数 $E=300$ 时即每 30 个 epoch 打印一次）。
- **第 87 行**：调用调度器更新学习率（每 135 个 epoch 将 $\eta$ 乘以 0.1）。
- **第 98 行**：返回最后一个 mini-batch 的损失标量，供 `objective` 函数判断是否发生 NaN 异常。

---

## 附录 D：`Scripts/PlotResults.py` 可视化模块

```python
18. def plotResults(model, device, x_train, y_train, x_test, y_test, x_val, y_val,
                   num_nodes, datacase, percentage_train, gamma, number_layers, hidden_nodes):
20.     train_bcs = torch.from_numpy(x_train.astype(np.float32)).to(device)
21.     test_bcs  = torch.from_numpy(x_test.astype(np.float32)).to(device)
22.     val_bcs   = torch.from_numpy(x_val.astype(np.float32)).to(device)
24.     pred_train = np.concatenate((x_train[:, :4], model(train_bcs).detach().cpu().numpy(), x_train[:, -4:]), axis=1)
25.     pred_test  = np.concatenate((x_test[:, :4],  model(test_bcs).detach().cpu().numpy(),  x_test[:, -4:]),  axis=1)
26.     pred_val   = np.concatenate((x_val[:, :4],   model(val_bcs).detach().cpu().numpy(),   x_val[:, -4:]),   axis=1)
28.     true_train = np.concatenate((x_train[:, :4], y_train, x_train[:, -4:]), axis=1)
29.     true_test  = np.concatenate((x_test[:, :4],  y_test,  x_test[:, -4:]),  axis=1)
30.     true_val   = np.concatenate((x_val[:, :4],   y_val,   x_val[:, -4:]),   axis=1)
32.     pred = np.concatenate((pred_train[:,4:-4], pred_test[:,4:-4], pred_val[:,4:-4]), axis=0)
33.     true = np.concatenate((true_train[:,4:-4], true_test[:,4:-4], true_val[:,4:-4]), axis=0)
34.     error_all        = np.mean((pred - true)**2)
35.     error_training   = np.mean((pred_train[:,4:-4] - true_train[:,4:-4])**2)
36.     error_testing    = np.mean((pred_test[:,4:-4]  - true_test[:,4:-4])**2)
37.     error_validation = np.mean((pred_val[:,4:-4]   - true_val[:,4:-4])**2)
39.     test_bvs = torch.from_numpy(x_test.astype(np.float32))
40.     initial_time = time.time()
41.     _ = model(test_bvs)
42.     final_time = time.time()
43.     total_time = final_time - initial_time
62.     norms_q  = np.zeros((len(pred_test), num_nodes))
63.     mean_q   = np.zeros(num_nodes)
64.     norms_qp = np.zeros((len(pred_test), num_nodes))
65.     mean_qp  = np.zeros(num_nodes)
66.     for i in range(len(pred_test)):
67.         for j in range(num_nodes):
68.             norms_q[i, j]  = np.linalg.norm(pred_test[i, 4*j:4*j+2]   - true_test[i, 4*j:4*j+2])
69.             mean_q[j]      = np.mean(norms_q[:, j])
70.             norms_qp[i, j] = np.linalg.norm(pred_test[i, 4*j+2:4*j+4] - true_test[i, 4*j+2:4*j+4])
71.             mean_qp[j]     = np.mean(norms_qp[:, j])
```

- **第 20–22 行**：将三个数据子集的输入特征转换为 float32 张量并迁移到目标设备，以备前向推理。
- **第 24–26 行**：对训练集、测试集、验证集各自执行模型前向推理，将预测的内部节点状态与边界条件拼接，还原为完整的 $4N$ 维曲线状态向量：

$$\hat{\mathbf{Y}}^{(\cdot)} = \left[\mathbf{X}^{(\cdot)}_{:,\;:4},\;\; \mathcal{N}_\theta(\mathbf{X}^{(\cdot)}),\;\; \mathbf{X}^{(\cdot)}_{:,\;-4:}\right] \in \mathbb{R}^{n_{(\cdot)} \times 4N}$$

- **第 28–30 行**：类似地，将真实内部节点标签与边界条件拼接，构建完整真实曲线矩阵 $\mathbf{Y}^{(\cdot)}_{\text{true}}$。
- **第 32–37 行**：合并三个子集的内部节点预测与真实值，计算各子集及总体的均方误差：

$$\text{MSE}_{(\cdot)} = \frac{1}{n_{(\cdot)} \cdot 4(N-2)} \sum_{i,j} \left(\hat{Y}^{(\cdot)}_{ij} - Y^{(\cdot)}_{ij}\right)^2$$

- **第 39–43 行**：在 CPU 上对测试集进行一次推理计时，测量批量预测总时间 $t_{\text{total}}$ 和平均单次时间 $\bar{t} = t_{\text{total}} / n_{\text{test}}$（与主函数第 283–291 行逻辑相同，此处用于写入结果文件）。
- **第 62–71 行**：逐节点计算测试集上位置误差 $\|\mathbf{q}\|$ 和切向误差 $\|\mathbf{q}'\|$ 的平均范数：

对第 $j$ 个节点，定义位置误差范数：
$$e_q^{(i,j)} = \left\| \hat{\mathbf{q}}^{(i,j)} - \mathbf{q}^{(i,j)} \right\|_2, \quad \hat{\mathbf{q}}^{(i,j)} = \left(\hat{q}_x^{(j)}, \hat{q}_y^{(j)}\right)_i$$

切向误差范数：
$$e_{q'}^{(i,j)} = \left\| \hat{\mathbf{q}}^{\prime(i,j)} - \mathbf{q}^{\prime(i,j)} \right\|_2$$

各节点上测试集平均误差：
$$\bar{e}_q^{(j)} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} e_q^{(i,j)}, \quad \bar{e}_{q'}^{(j)} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} e_{q'}^{(i,j)}$$

```python
73.     if datacase == 1:
74.         fig1 = plt.figure(figsize=((20, 15)))
76.         plt.plot(true_test[i, np.arange(0, d, 4)], true_test[i, np.arange(1, d, 4)], '-', ...)
77.         plt.plot(pred_test[i, np.arange(0, d, 4)], pred_test[i, np.arange(1, d, 4)], '--d', ...)
81.         plt.xlabel(r"$q_x$", fontsize="45")
82.         plt.ylabel(r"$q_y$", fontsize="45")
88.         fig2 = plt.figure(figsize=((20, 15)))
89.         circle = plt.Circle((0, 0), 1, color='k', alpha=0.5, fill=False)
107.        fig3 = plt.figure(figsize=((20, 15)))
108.        plt.plot(np.linspace(0, 50, 51), mean_q,  '-d', ...)
109.        plt.plot(np.linspace(0, 50, 51), mean_qp, '-d', ...)
```

- **图 1（第 73–86 行）**：对比真实与预测的离散曲线在位置空间 $(q_x, q_y)$ 中的形态，绘制测试集中选取的若干样本曲线。索引步长 11（`datacase=1`）和 22（`datacase=2`）用于控制绘图密度。
- **图 2（第 88–104 行）**：在单位圆参考系下，对比真实与预测的切向分量 $(q_x', q_y')$ 的分布，以散点图展示。单位圆（$\|\mathbf{q}'\|_2 = 1$）代表不可伸长弧长参数化约束：

$$\left\| \mathbf{q}'(s) \right\|_2 = 1, \quad \forall s$$

- **图 3（第 106–114 行）**：按节点编号 $k = 0, 1, \ldots, N-1$ 绘制沿曲线的平均位置误差 $\bar{e}_q^{(k)}$ 和切向误差 $\bar{e}_{q'}^{(k)}$，直观展示预测误差在曲线各节点上的分布规律。

---

## 附录 E：`Scripts/SavedParameters.py` 超参数存储模块

```python
 1. import pandas as pd
 2. import numpy as np
 4. def hyperparams(case, pctg):
 6.     best_vals = pd.DataFrame()
 7.     best_vals["percentage_train"] = np.array([0.1, 0.2, 0.4, 0.8, 0.8])
 8.     best_vals["datacase"]         = np.array([1, 1, 1, 1, 2])
 9.     best_vals["gamma"]            = np.array([0.007044405451814177, 0.006335851468590373,
                                                  0.009004175808977003, 0.003853035138801786,
                                                  0.0073229668983443436])
12.     best_vals["n_layers"]         = np.array([4, 4, 4, 4, 3])
13.     best_vals["hidden_nodes"]     = np.array([950, 978, 997, 985, 616])
15.     vals = best_vals[(best_vals['datacase'] == case) & (best_vals['percentage_train'] == pctg)]
17.     nlayers      = vals.iloc[0]["n_layers"]
18.     hidden_nodes = vals.iloc[0]["hidden_nodes"]
19.     gamma        = vals.iloc[0]["gamma"]
21.     params = {'n_layers':     int(nlayers),
22.               'hidden_nodes': int(hidden_nodes),
23.               'gamma':        gamma}
24.     return params
```

- **第 1–2 行**：导入 `pandas` 用于表格查找，`numpy` 用于数组构建。
- **第 4 行**：定义 `hyperparams` 函数，接收数据案例 `case` 和训练比例 `pctg`，返回对应的最优超参数字典。
- **第 6–13 行**：以硬编码方式在 pandas DataFrame 中存储由 Optuna 搜索得到的最优超参数组合，共 5 组记录，覆盖如下配置：

| `datacase` | `percentage_train` | $\gamma$ | `n_layers` ($L$) | `hidden_nodes` ($H$) |
|:-----------:|:------------------:|:--------:|:----------------:|:-------------------:|
| 1 | 0.1 | 0.00704 | 4 | 950 |
| 1 | 0.2 | 0.00634 | 4 | 978 |
| 1 | 0.4 | 0.00900 | 4 | 997 |
| 1 | 0.8 | 0.00385 | 4 | 985 |
| 2 | 0.8 | 0.00732 | 3 | 616 |

- **第 15 行**：按 `datacase` 和 `percentage_train` 两个条件过滤 DataFrame，获取匹配的行：

$$\text{row} = \{ \mathbf{v} \in \mathbf{D} \mid v_{\text{case}} = \texttt{case} \;\wedge\; v_{\text{pctg}} = \texttt{pctg} \}$$

- **第 17–19 行**：从过滤结果的第一行提取 $L$（层数）、$H$（节点数）和 $\gamma$（差分正则化系数）三个超参数。
- **第 21–24 行**：将提取到的超参数组装为字典并返回，其中 $L$ 和 $H$ 转换为 Python `int` 类型以兼容 PyTorch 的 `nn.Linear` 构造参数，$\gamma$ 保持 `float` 类型：

$$\texttt{params} = \left\{ \texttt{n\_layers}: L,\; \texttt{hidden\_nodes}: H,\; \texttt{gamma}: \gamma \right\}$$
