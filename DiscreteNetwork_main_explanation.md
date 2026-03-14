# DiscreteNetwork_main.py 逐行代码解释

本文档按主函数的运行顺序，对 `DiscreteNetwork_main.py` 及其调用的所有辅助模块（`GetData.py`、`Training.py`、`PlotResults.py`、`SavedParameters.py`）进行详细解释。所有数学公式均使用 LaTeX 语法渲染。

---

## 目录

1. [包导入](#1-包导入)
2. [本地模块导入](#2-本地模块导入)
3. [绘图参数设置](#3-绘图参数设置)
4. [全局张量类型与设备设置](#4-全局张量类型与设备设置)
5. [用户输入与随机种子](#5-用户输入与随机种子)
6. [approximate_curve 神经网络类](#6-approximate_curve-神经网络类)
7. [loadData 调用与 define_model 函数](#7-loaddata-调用与-define_model-函数)
8. [objective 目标函数（Optuna 超参数搜索）](#8-objective-目标函数optuna-超参数搜索)
9. [Optuna 超参数搜索流程](#9-optuna-超参数搜索流程)
10. [最优超参数加载](#10-最优超参数加载)
11. [define_best_model 与模型初始化](#11-define_best_model-与模型初始化)
12. [训练或加载预训练模型](#12-训练或加载预训练模型)
13. [结果绘制与误差评估](#13-结果绘制与误差评估)
14. [推理时间测量](#14-推理时间测量)
15. [辅助模块详解](#15-辅助模块详解)
    - [GetData.py](#151-getdatapy)
    - [Training.py](#152-trainingpy)
    - [PlotResults.py](#153-plotresultspy)
    - [SavedParameters.py](#154-savedparameterspy)

---

## 1. 包导入

```python
# 行 1-2
#!/usr/bin/env python
# coding: utf-8
```

Shebang 行与编码声明，指定脚本使用 Python 解释器，文件编码为 UTF-8。

```python
# 行 8-20
get_ipython().system('pip install optuna')
import optuna
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from csv import writer
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

- **行 8**：在 Jupyter/IPython 环境中通过 `get_ipython().system(...)` 调用 shell 命令安装 `optuna`（超参数优化框架）。该写法仅在 Jupyter Notebook 或 IPython 内核中有效；在普通 Python 脚本中等价于命令行执行 `pip install optuna`。
- **行 9**：导入 `optuna`，用于自动化超参数搜索。
- **行 10**：导入 `torch`（PyTorch），用于神经网络的构建、训练和推理。
- **行 11**：导入 `random`，用于 Python 原生随机数生成（配合固定随机种子保证可重复性）。
- **行 12**：导入 `torch.nn`，包含神经网络层、损失函数等模块。
- **行 13**：导入 `numpy`（数值计算）。
- **行 14-15**：导入 `matplotlib`，用于图形绘制。
- **行 16**：导入 `torch.nn.functional`，提供激活函数等无状态操作。
- **行 17**：从 `csv` 模块导入 `writer`，用于将结果写入 CSV 文件。
- **行 18**：导入 `seaborn`，用于美化图表样式。
- **行 19-20**：导入 `os` 并设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`，解决 Intel OpenMP 库重复加载导致的崩溃问题。

---

## 2. 本地模块导入

```python
# 行 26-30
from Scripts.GetData import getDataLoaders, loadData
from Scripts.Training import train
from Scripts.PlotResults import plotResults
from Scripts.SavedParameters import hyperparams
import pandas as pd
```

从 `DiscreteNetwork/Scripts/` 目录导入各功能模块：

| 导入项 | 所在模块 | 功能 |
|--------|---------|------|
| `getDataLoaders` | `GetData.py` | 构建训练/验证/测试 DataLoader |
| `loadData` | `GetData.py` | 从文本文件加载弹性曲线数据 |
| `train` | `Training.py` | 执行神经网络训练循环 |
| `plotResults` | `PlotResults.py` | 绘制并保存预测结果图 |
| `hyperparams` | `SavedParameters.py` | 返回预设的最优超参数组合 |
| `pd` | `pandas` | 数据框操作，用于超参数查找 |

---

## 3. 绘图参数设置

```python
# 行 37-42
sns.set_style("darkgrid")
sns.set(font = "Times New Roman")
sns.set_context("paper")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt_kws = {"rasterized": True}
```

- **行 37**：设置 seaborn 样式为深色网格背景。
- **行 38**：将图表字体设置为 Times New Roman（论文风格）。
- **行 39**：设置绘图上下文为 "paper"，缩小图元尺寸以适合学术排版。
- **行 40**：将 matplotlib 的数学字体设置为 Computer Modern（LaTeX 默认字体）。
- **行 41**：将正文字体族设置为 STIXGeneral（与 LaTeX 风格接近）。
- **行 42**：定义绘图关键字，启用栅格化渲染（减小矢量图文件体积）。

---

## 4. 全局张量类型与设备设置

```python
# 行 48
torch.set_default_dtype(torch.float32)

# 行 54-55
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

- **行 48**：将 PyTorch 全局默认浮点类型设为 32 位单精度，节省显存并加速计算。
- **行 54**：检测是否有可用的 NVIDIA GPU（CUDA 加速），若有则使用 `cuda:0`（第一块 GPU），否则使用 CPU。
- **行 55**：打印当前使用的计算设备。

---

## 5. 用户输入与随机种子

```python
# 行 61-68
datacase = int(input("Which datacase do you want to work with?\n"))
percentage_train = float(input("Which percentage of the dataset do you want to use for training? Choose among 0.1,0.2,0.4,0.8\n"))

print(f"\n\n Case with percentage_train={percentage_train} and datacase={datacase}\n\n")

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
```

- **行 61**：从用户处读取数据集编号（`datacase`），`1` 表示仅使用双端数据集，`2` 表示使用双端+右端数据集的合并集。
- **行 62**：读取训练集比例，可选值为 $0.1, 0.2, 0.4, 0.8$。
- **行 64**：打印当前实验配置。
- **行 66-68**：分别固定 PyTorch、NumPy、Python 原生随机数生成器的随机种子为 `1`，保证整个实验的可重复性。

---

## 6. `approximate_curve` 神经网络类

```python
# 行 70-102
class approximate_curve(nn.Module):
    def __init__(self, is_res=True, normalize=True, act_name='tanh',
                 nlayers=3, hidden_nodes=50, output_dim=204):
        super().__init__()
        ...
    def forward(self, x):
        ...
```

### 6.1 `__init__` 构造方法（行 71–88）

```python
# 行 74-76
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
```

在构造函数内再次固定随机种子，确保每次创建模型时权重初始化一致。

```python
# 行 77-81
self.act_dict = {"tanh":   lambda x: torch.tanh(x),
                 "sigmoid": lambda x: torch.sigmoid(x),
                 "swish":   lambda x: x * torch.sigmoid(x),
                 "relu":    lambda x: torch.relu(x),
                 "lrelu":   lambda x: F.leaky_relu(x)}
```

定义激活函数字典，支持五种激活函数：

$$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

$$\text{swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \geq 0 \\ \alpha x & \text{if } x < 0 \end{cases}, \quad \alpha = 0.01$$

```python
# 行 82-88
self.is_norm = normalize
self.is_res = is_res
self.act = self.act_dict[act_name]
self.nlayers = nlayers
self.first = nn.Linear(8, hidden_nodes)
self.linears = nn.ModuleList([nn.Linear(hidden_nodes, hidden_nodes) for i in range(self.nlayers)])
self.last = nn.Linear(hidden_nodes, output_dim)
```

- **`self.is_norm`**：是否对输入进行归一化。
- **`self.is_res`**：是否使用残差连接（ResNet 模式）。
- **`self.act`**：选定的激活函数。
- **`self.first`**：第一个全连接层，将 8 维输入映射到 $d_h$ 维隐藏层。

  $$\mathbf{h}_0 = \sigma(W_0 \mathbf{x} + \mathbf{b}_0), \quad W_0 \in \mathbb{R}^{d_h \times 8}$$

- **`self.linears`**：由 `nlayers` 个隐藏层组成的模块列表，每层形状均为 $d_h \to d_h$。

  $$\mathbf{h}_i = \sigma(W_i \mathbf{h}_{i-1} + \mathbf{b}_i), \quad W_i \in \mathbb{R}^{d_h \times d_h}$$

- **`self.last`**：输出层，将隐藏层映射到 `output_dim` 维输出。

  $$\hat{\mathbf{y}} = W_{\text{out}} \mathbf{h}_{\text{last}} + \mathbf{b}_{\text{out}}, \quad W_{\text{out}} \in \mathbb{R}^{d_{\text{out}} \times d_h}$$

  其中 $d_{\text{out}} = 4(N-2)$，$N$ 为链中节点总数，每个内部节点有 4 个状态量 $(q_x, q_y, q'_x, q'_y)$。

### 6.2 `forward` 前向传播方法（行 90–102）

```python
# 行 92-94
if self.is_norm:
    x[:, 0] = (x[:, 0] - 1.5) / 1.5
    x[:, 4] = (x[:, 4] - 1.5) / 1.5
```

对输入特征中的第 0 列和第 4 列（分别对应左端点和右端点的某一分量）进行归一化：

$$\tilde{x}_i = \frac{x_i - 1.5}{1.5}, \quad i \in \{0, 4\}$$

```python
# 行 95
x = self.act(self.first(x))
```

通过第一个全连接层并施加激活函数：

$$\mathbf{h}_0 = \sigma(W_0 \mathbf{x} + \mathbf{b}_0)$$

```python
# 行 96-100
for i in range(self.nlayers):
    if self.is_res:  # ResNet
        x = x + self.act(self.linears[i](x))
    else:            # MLP
        x = self.act(self.linears[i](x))
```

循环通过所有隐藏层：

- **残差（ResNet）模式**：在每个隐藏层引入跳跃连接：

  $$\mathbf{h}_{i+1} = \mathbf{h}_i + \sigma(W_i \mathbf{h}_i + \mathbf{b}_i)$$

- **普通多层感知机（MLP）模式**：

  $$\mathbf{h}_{i+1} = \sigma(W_i \mathbf{h}_i + \mathbf{b}_i)$$

```python
# 行 102
return self.last(x)
```

通过输出层产生最终预测，无激活函数（线性输出）：

$$\hat{\mathbf{y}} = W_{\text{out}} \mathbf{h}_{\text{last}} + \mathbf{b}_{\text{out}}$$

输出向量 $\hat{\mathbf{y}} \in \mathbb{R}^{4(N-2)}$ 包含所有内部节点的位置和切向量预测。

---

## 7. `loadData` 调用与 `define_model` 函数

```python
# 行 104
num_nodes, _, _ = loadData(datacase)
```

调用 `GetData.py` 中的 `loadData` 函数，获取弹性链的节点总数 $N$（包含边界节点），详见 [§15.1](#151-getdatapy)。

```python
# 行 106-117
def define_model(trial):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    is_res = False
    normalize = True
    act_name = "tanh"
    nlayers = trial.suggest_int("n_layers", 0, 10)
    hidden_nodes = trial.suggest_int("hidden_nodes", 10, 1000)

    model = approximate_curve(is_res, normalize, act_name, nlayers, hidden_nodes,
                               output_dim=int(4 * (num_nodes - 2)))
    return model
```

`define_model` 是供 Optuna 调用的模型构造函数：

- **行 113**：Optuna 从整数区间 $[0, 10]$ 中建议隐藏层数 `n_layers`。
- **行 114**：Optuna 从整数区间 $[10, 1000]$ 中建议每层隐藏节点数 `hidden_nodes`。
- **行 116**：构造神经网络，输出维度为 $4(N-2)$（$N-2$ 个内部节点，每个节点预测 4 个分量）。

---

## 8. `objective` 目标函数（Optuna 超参数搜索）

```python
# 行 121-185
def objective(trial):
    ...
```

`objective` 是 Optuna 超参数优化的目标函数，每次 trial 中完整执行一次训练并返回验证集误差作为优化目标。

### 8.1 模型与优化器设置（行 128–136）

```python
# 行 128-136
model = define_model(trial)
model.to(device)

lr = 1e-3
weight_decay = 0
gamma = trial.suggest_float("gamma", 0, 1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

criterion = nn.MSELoss()
```

- **`lr = 1e-3`**：Adam 优化器学习率 $\eta = 10^{-3}$。
- **`weight_decay = 0`**：L2 正则化系数为 0（不使用权重衰减）。
- **`gamma`**：差分损失的权重系数，由 Optuna 在 $[0, 10^{-2}]$ 中搜索。
- **Adam 优化器**更新规则：

  $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}$$
  $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2$$
  $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
  $$\theta_{t+1} = \theta_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

  默认参数：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$。

- **`criterion = nn.MSELoss()`**：均方误差损失函数：

  $$\mathcal{L}_{\text{MSE}}(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

### 8.2 数据加载与训练（行 138–150）

```python
# 行 138-150
batch_size = 32
_, _, _, _, x_val, y_val, trainloader, _, valloader = getDataLoaders(batch_size, datacase, percentage_train)

epochs = 300
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs), gamma=0.1)

loss = train(model, gamma, criterion, scheduler, optimizer, epochs, trainloader, valloader, device)
```

- **`batch_size = 32`**：每个小批量包含 32 条样本。
- **StepLR 学习率调度器**：每 $\lfloor 0.45 \times 300 \rfloor = 135$ 个 epoch，将学习率乘以 $0.1$：

  $$\eta_{\text{new}} = \eta_{\text{old}} \times 0.1 \quad \text{（每 135 个 epoch）}$$

### 8.3 验证集误差计算（行 153–162）

```python
# 行 153-162
if not torch.isnan(loss):
    model.eval()
    bcs_val = torch.from_numpy(x_val.astype(np.float32)).to(device)
    learned_traj = model(bcs_val).detach().cpu().numpy()
    error = np.mean((learned_traj - y_val) ** 2)
```

若训练损失未出现 NaN，则在验证集上计算均方误差：

$$\text{MSE}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

其中 $\hat{\mathbf{y}}_i$ 为模型预测的内部节点状态，$\mathbf{y}_i$ 为真实值。

### 8.4 结果保存（行 165–185）

```python
# 行 165-184
if trial.number == 0:
    labels = [...]
    labels.append("MSE")
    with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(labels)
        ...

results = [...]
results.append(error)
with open(f"results{int(percentage_train*100)}_Fig2.csv", "a") as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(results)
    ...
return error
```

- 第一次 trial（`trial.number == 0`）时写入 CSV 表头（超参数名称 + MSE）。
- 每次 trial 结束后将当前超参数组合及其验证误差追加写入 CSV。
- 最终返回 `error` 作为 Optuna 的最小化目标。

---

## 9. Optuna 超参数搜索流程

```python
# 行 187-198
optuna_study = input("Do you want to do hyperparameter test? Type yes or no: ")
params = {}
if optuna_study == "yes":
    optuna_study = True
else:
    optuna_study = False
if optuna_study:
    study = optuna.create_study(direction="minimize", study_name="Euler Elastica")
    study.optimize(objective, n_trials=5)
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    params = study.best_params
```

- **行 187**：询问用户是否执行超参数搜索。
- **行 194**：创建 Optuna study，优化方向为最小化（最小验证集 MSE）。
- **行 195**：执行 5 次独立 trial，每次 trial 对应一组超参数配置 $(n\_\text{layers}, d_h, \gamma)$。
- **行 198**：提取使验证误差最小的最优超参数字典 `params`。

---

## 10. 最优超参数加载

```python
# 行 200-221
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

manual_input = False
if params == {}:
    if manual_input:
        # 手动输入超参数（当前被跳过）
        ...
    else:
        params = hyperparams(datacase, percentage_train)
print(f'The hyperparameters yelding the best results for this case are: {params}')
```

- **行 200-202**：重置随机种子以确保后续操作的可重复性。
- **行 205**：若未进行 Optuna 搜索（`params == {}`），进入超参数加载分支。
- **行 220**：调用 `hyperparams(datacase, percentage_train)` 从 `SavedParameters.py` 中查找预存的最优超参数，详见 [§15.4](#154-savedparameterspy)。

---

## 11. `define_best_model` 与模型初始化

```python
# 行 222-254
def define_best_model():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    normalize = True
    act = "tanh"
    nlayers = params["n_layers"]
    hidden_nodes = params["hidden_nodes"]
    is_res = False

    print("Nodes: ", hidden_nodes)

    model = approximate_curve(is_res, normalize, act, nlayers, hidden_nodes, int(4*(num_nodes-2)))
    return model

model = define_best_model()
model.to(device)
```

使用最优超参数重新构造最终模型：

- `is_res = False`：使用普通 MLP 而非 ResNet（由 Optuna 搜索后确定为更优配置）。
- `normalize = True`：对边界条件输入进行归一化。
- `act = "tanh"`：使用双曲正切激活函数。
- 输出维度为 $4(N-2)$，对应所有内部节点的 $(q_x, q_y, q'_x, q'_y)$。

```python
# 行 243-254
TrainMode = input("Train Mode True or False? Type 0 for False and 1 for True: ") == "1"
weight_decay = 0.
lr = 1e-3
gamma = params["gamma"]
nlayers = params["n_layers"]
hidden_nodes = params["hidden_nodes"]
batch_size = 32
epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.45*epochs), gamma=0.1)
criterion = nn.MSELoss()
x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader = \
    getDataLoaders(batch_size, datacase, percentage_train)
model.to(device)
```

从用户输入决定是否进行训练（`TrainMode`），并配置全部训练相关超参数和数据加载器。学习率调度与 [§8.2](#82-数据加载与训练行138150) 相同：

$$\eta_t = \begin{cases} 10^{-3} & t < 135 \\ 10^{-4} & 135 \leq t < 270 \\ 10^{-5} & t \geq 270 \end{cases}$$

---

## 12. 训练或加载预训练模型

```python
# 行 256-267
if TrainMode:
    loss = train(model, gamma, criterion, scheduler, optimizer, epochs, trainloader, valloader, device)
    if datacase == 1:
        torch.save(model.state_dict(), f'TrainedModels/BothEnds{percentage_train}data.pt')
    if datacase == 2:
        torch.save(model.state_dict(), f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt')
else:
    if datacase == 1:
        pretrained_dict = torch.load(f'TrainedModels/BothEnds{percentage_train}data.pt', map_location=device)
    if datacase == 2:
        pretrained_dict = torch.load(f'TrainedModels/BothEndsRightEnd{percentage_train}data.pt', map_location=device)
    model.load_state_dict(pretrained_dict)
model.eval()
```

- **训练模式**：调用 `train(...)` 函数（详见 [§15.2](#152-trainingpy)）执行 `epochs=300` 轮训练，并将训练完成的模型参数以 `.pt` 文件保存到磁盘。
- **推理模式**：从磁盘加载预训练模型参数，通过 `map_location=device` 确保在目标设备（GPU 或 CPU）上正确加载。
- **行 268**：将模型切换到评估模式，关闭 Dropout 和 BatchNorm 的训练行为（本模型未使用，但为标准做法）。

---

## 13. 结果绘制与误差评估

```python
# 行 270-277
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

model.eval()

plotResults(model, device, x_train, y_train, x_test, y_test, x_val, y_val,
            num_nodes, datacase, percentage_train, gamma, nlayers, hidden_nodes)
```

- 再次重置随机种子，确保评估和绘图结果可重复。
- 调用 `plotResults(...)` 函数（详见 [§15.3](#153-plotresultspy)），计算并打印训练/验证/测试误差，绘制预测曲线对比图。

---

## 14. 推理时间测量

```python
# 行 283-291
import time
test_bvs = torch.from_numpy(x_test.astype(np.float32))
initial_time = time.time()
preds = model(test_bvs)
final_time = time.time()
total_time = final_time - initial_time
print("Number of trajectories in the test set : ", len(test_bvs))
print("Total time to predict test trajectories : ", total_time)
print("Average time to predict test trajectories : ", total_time / len(test_bvs))
```

- **行 284**：将测试集边界条件 `x_test` 转换为 PyTorch float32 张量（注意此处**未**将张量移至 GPU，在 CPU 上进行推理测速）。
- **行 285-288**：使用 `time.time()` 记录前向推理的墙时间：

  $$t_{\text{total}} = t_{\text{end}} - t_{\text{start}}$$

- **行 291**：计算并打印每条弹性曲线轨迹的平均推理时间：

  $$t_{\text{avg}} = \frac{t_{\text{total}}}{n_{\text{test}}}$$

---

## 15. 辅助模块详解

### 15.1 `GetData.py`

#### `loadData(datacase=1)` 函数（行 7–30）

```python
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
        trajectories_test = trajectories_train
    elif datacase == 2:
        trajectories_train = np.concatenate((trajectoriesload_b_360, trajectoriesload_r_360), axis=0)
        trajectories_test = trajectories_train
    else:
        print("Warning! Must be an integer between 1 and 3")

    num_nodes = trajectories_train.shape[1] // 4
    os.chdir(original_dir)
    return num_nodes, trajectories_train, trajectories_test
```

- **行 9-11**：临时切换工作目录至 `DataSets/` 文件夹以读取数据文件。
- **行 13-17**：分别加载两种边界条件下的弹性曲线数据集：`both_ends.txt`（双端固定）和 `right_end.txt`（右端固定）。
- **数据格式**：每行为一条弹性曲线的全局状态，每个节点 $k$ 对应 4 个分量 $(q_x^{(k)}, q_y^{(k)}, q'^{(k)}_x, q'^{(k)}_y)$，其中 $\mathbf{q}^{(k)} \in \mathbb{R}^2$ 为位置，$\mathbf{q}'^{(k)} \in \mathbb{R}^2$ 为单位切向量（满足 $\|\mathbf{q}'^{(k)}\| = 1$）。
- **`datacase == 2`**：将两个数据集在样本维度（`axis=0`）拼接：

  $$\mathbf{D}_{\text{train}} = \begin{bmatrix} \mathbf{D}_{\text{both}} \\ \mathbf{D}_{\text{right}} \end{bmatrix}$$

- **行 28**：节点总数 $N = \lfloor \text{columns} / 4 \rfloor$，每条曲线共有 $N$ 个节点。
- **行 30**：还原工作目录并返回 $(N, \mathbf{D}_{\text{train}}, \mathbf{D}_{\text{test}})$。

#### `dataset` 类（行 32–41）

```python
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

继承 PyTorch `Dataset` 基类，封装弹性曲线数据集：

- **`__init__`**：将 NumPy 数组 `x`（边界条件，8 维）和 `y`（内部节点真实值，$4(N-2)$ 维）转换为 float32 张量。
- **`__getitem__(idx)`**：返回第 `idx` 条样本的 $(\mathbf{x}_{\text{bc}}, \mathbf{y}_{\text{internal}})$ 对，支持 DataLoader 的批量采样。
- **`__len__`**：返回数据集的样本总数，供 DataLoader 内部使用。

#### `getDataLoaders(batch_size, datacase, percentage_train)` 函数（行 45–95）

```python
def getDataLoaders(batch_size, datacase, percentage_train):
    ...
    _, data_train, data_test = loadData(datacase)
    x_full_train = np.concatenate((data_train[:, :4], data_train[:, -4:]), axis=1)
    y_full_train = data_train[:, 4:-4]
    N = len(x_full_train)
    NTrain = int(percentage_train * N)
    ...
```

该函数负责构建完整的数据流水线：

- **输入特征提取**：对每条曲线，将前 4 列（左端节点 $(q_x^{(0)}, q_y^{(0)}, q'^{(0)}_x, q'^{(0)}_y)$）和后 4 列（右端节点 $(q_x^{(N-1)}, q_y^{(N-1)}, q'^{(N-1)}_x, q'^{(N-1)}_y)$）拼接作为输入：

  $$\mathbf{x}_{\text{bc}} = \bigl(q_x^{(0)},\, q_y^{(0)},\, q'^{(0)}_x,\, q'^{(0)}_y,\, q_x^{(N-1)},\, q_y^{(N-1)},\, q'^{(N-1)}_x,\, q'^{(N-1)}_y\bigr) \in \mathbb{R}^8$$

- **输出标签提取**：中间 $N-2$ 个内部节点的全部状态：

  $$\mathbf{y}_{\text{internal}} = \bigl(q_x^{(1)}, q_y^{(1)}, q'^{(1)}_x, q'^{(1)}_y, \ldots, q_x^{(N-2)}, q_y^{(N-2)}, q'^{(N-2)}_x, q'^{(N-2)}_y\bigr) \in \mathbb{R}^{4(N-2)}$$

```python
# 行 57-61
idx_shuffle_train = np.arange(N)
random.shuffle(idx_shuffle_train)
x_full_train = x_full_train[idx_shuffle_train]
y_full_train = y_full_train[idx_shuffle_train]
```

对全部 $N$ 条样本进行随机打乱。

```python
# 行 69-81
fact = 0.1
if percentage_train == 0.8:
    fact = 0.1
elif percentage_train == 0.7:
    fact = 0.15
else:
    fact = 0.2

x_train, y_train = x_full_train[:NTrain], y_full_train[:NTrain]
Number_Test_Points = int(fact * N)
x_test, y_test = x_full_test[NTrain:NTrain+Number_Test_Points], ...
x_val, y_val = x_full_test[NTrain+Number_Test_Points:NTrain+2*Number_Test_Points], ...
```

数据集划分策略：

- 训练集：前 $N_{\text{train}} = \lfloor p \cdot N \rfloor$ 条样本，其中 $p$ = `percentage_train`。
- 测试集与验证集：各取 $N_{\text{test}} = \lfloor f \cdot N \rfloor$ 条样本，其中因子 $f$ 由 `percentage_train` 决定（$p = 0.8$ 时 $f=0.1$，否则 $f=0.2$）。

  $$N_{\text{train}} + N_{\text{test}} + N_{\text{val}} \leq N$$

```python
# 行 87-93
trainset = dataset(x_train, y_train)
testset  = dataset(x_test,  y_test)
valset   = dataset(x_val,   y_val)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader  = DataLoader(testset,  batch_size=len(x_test), shuffle=True)
valloader   = DataLoader(valset,   batch_size=len(x_val),  shuffle=True)
```

将数据封装成 `dataset` 对象后通过 `DataLoader` 管理批量数据：
- `trainloader`：每个批次随机抽取 `batch_size=32` 条样本，每个 epoch 打乱顺序。
- `testloader` 和 `valloader`：每批次加载所有数据（整批评估）。

---

### 15.2 `Training.py`

#### `EarlyStopper` 类（行 6–21）

```python
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
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

实现早停机制（当前被注释掉，未在实际训练中启用）：

- **`patience`**：允许验证集损失连续不改善的最大轮数。
- **`min_delta`**：视为"改善"的最小阈值 $\delta_{\min}$。
- **`early_stop(validation_loss)`** 逻辑：
  - 若当前验证损失 $\mathcal{L}_{\text{val}} < \mathcal{L}_{\min}$，更新最小值并重置计数器。
  - 若 $\mathcal{L}_{\text{val}} > \mathcal{L}_{\min} + \delta_{\min}$，计数器加一；若计数器 $\geq$ `patience`，返回 `True` 触发早停。

#### `train` 函数（行 23–98）

```python
def train(model, gamma, criterion, scheduler, optimizer, epochs,
          trainloader, valloader, device):
    ...
```

##### 训练循环（行 32–87）

```python
# 行 32-65
for epoch in range(epochs):
    train_loss = 0.
    counter = 0.

    for _, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        predicted = model(inputs)

        loss = criterion(predicted, labels)

        predicted = torch.cat((inputs[:, :4], predicted, inputs[:, 4:]), dim=1)
        labels    = torch.cat((inputs[:, :4], labels,    inputs[:, 4:]), dim=1)

        predicted_first   = predicted[:, :-4]
        predicted_forward = predicted[:, 4:]
        labels_first      = labels[:, :-4]
        labels_forward    = labels[:, 4:]

        diff_predicted = predicted_forward - predicted_first
        diff_labels    = labels_forward    - labels_first

        loss += gamma * criterion(diff_predicted, diff_labels)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        counter += 1
```

训练的总损失由两项组成：

**第一项：内部节点均方误差（MSE 损失）**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^{B} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

其中 $B$ 为批量大小，$\hat{\mathbf{y}}_i$ 为网络对第 $i$ 条曲线内部节点的预测，$\mathbf{y}_i$ 为真实值。

**第二项：相邻节点差分损失（正则化项）**

将预测值与真实值重新拼合边界节点后得到完整链 $\hat{\mathbf{z}}_i, \mathbf{z}_i \in \mathbb{R}^{4N}$，然后计算相邻节点差分：

$$\Delta \hat{\mathbf{z}}_i^{(k)} = \hat{\mathbf{z}}_i^{(k+1)} - \hat{\mathbf{z}}_i^{(k)}, \quad k = 0, 1, \ldots, N-2$$

差分损失为：

$$\mathcal{L}_{\text{diff}} = \frac{1}{B} \sum_{i=1}^{B} \|\Delta\hat{\mathbf{z}}_i - \Delta\mathbf{z}_i\|_2^2$$

**总损失**：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \gamma \cdot \mathcal{L}_{\text{diff}}$$

其中 $\gamma \geq 0$ 为差分损失权重，由超参数搜索确定。

```python
# 行 67-76
model.eval()
with torch.no_grad():
    data = next(iter(valloader))
    inputs, labels = data[0].to(device), data[1].to(device)
    predicted = model(inputs)
    val_loss = criterion(predicted, labels)
model.train()
losses_val.append(val_loss.item())
```

每个 epoch 结束后，在验证集上计算纯 MSE 损失（不含差分项）：

$$\mathcal{L}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

```python
# 行 87
scheduler.step()
```

更新学习率调度器，按 StepLR 规则每 `step_size` 个 epoch 将学习率乘以 $\gamma_{\text{lr}} = 0.1$。

---

### 15.3 `PlotResults.py`

#### `plotResults` 函数（行 18–160）

```python
def plotResults(model, device, x_train, y_train, x_test, y_test, x_val, y_val,
                num_nodes, datacase, percentage_train, gamma, number_layers, hidden_nodes):
```

##### 预测结果生成（行 20–30）

```python
# 行 20-30
train_bcs = torch.from_numpy(x_train.astype(np.float32)).to(device)
test_bcs  = torch.from_numpy(x_test.astype(np.float32)).to(device)
val_bcs   = torch.from_numpy(x_val.astype(np.float32)).to(device)

pred_train = np.concatenate((x_train[:, :4], model(train_bcs).detach().cpu().numpy(), x_train[:, -4:]), axis=1)
pred_test  = np.concatenate((x_test[:, :4],  model(test_bcs).detach().cpu().numpy(),  x_test[:, -4:]),  axis=1)
pred_val   = np.concatenate((x_val[:, :4],   model(val_bcs).detach().cpu().numpy(),   x_val[:, -4:]),   axis=1)

true_train = np.concatenate((x_train[:, :4], y_train, x_train[:, -4:]), axis=1)
true_test  = np.concatenate((x_test[:, :4],  y_test,  x_test[:, -4:]),  axis=1)
true_val   = np.concatenate((x_val[:, :4],   y_val,   x_val[:, -4:]),   axis=1)
```

将边界节点重新拼合到预测/真实内部节点两侧，得到完整的 $4N$ 维曲线表示 $\hat{\mathbf{z}}, \mathbf{z} \in \mathbb{R}^{4N}$。

##### 误差计算（行 32–37）

```python
# 行 32-37
pred = np.concatenate((pred_train[:,4:-4], pred_test[:,4:-4], pred_val[:,4:-4]), axis=0)
true = np.concatenate((true_train[:,4:-4], true_test[:,4:-4], true_val[:,4:-4]), axis=0)
error_all        = np.mean((pred - true)**2)
error_training   = np.mean((pred_train[:,4:-4] - true_train[:,4:-4])**2)
error_testing    = np.mean((pred_test[:,4:-4]  - true_test[:,4:-4])**2)
error_validation = np.mean((pred_val[:,4:-4]   - true_val[:,4:-4])**2)
```

各集合上的 MSE（均只计算内部节点 $k=1,\ldots,N-2$）：

$$\text{MSE}_{\text{train}} = \frac{1}{n_{\text{train}}} \sum_{i} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

$$\text{MSE}_{\text{test}} = \frac{1}{n_{\text{test}}} \sum_{i} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

$$\text{MSE}_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

$$\text{MSE}_{\text{all}} = \frac{1}{n_{\text{train}}+n_{\text{test}}+n_{\text{val}}} \sum_{\text{all}} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$

##### 逐节点误差计算（行 62–71）

```python
# 行 62-71
norms_q  = np.zeros((len(pred_test), num_nodes))
mean_q   = np.zeros(num_nodes)
norms_qp = np.zeros((len(pred_test), num_nodes))
mean_qp  = np.zeros(num_nodes)
for i in range(len(pred_test)):
    for j in range(num_nodes):
        norms_q[i, j]  = np.linalg.norm(pred_test[i, 4*j:4*j+2]   - true_test[i, 4*j:4*j+2])
        mean_q[j]       = np.mean(norms_q[:, j])
        norms_qp[i, j] = np.linalg.norm(pred_test[i, 4*j+2:4*j+4] - true_test[i, 4*j+2:4*j+4])
        mean_qp[j]      = np.mean(norms_qp[:, j])
```

对测试集中每条曲线的每个节点 $k$，分别计算位置分量和切向量分量的预测误差 $\ell^2$ 范数：

$$e_q^{(i,k)} = \left\| \hat{\mathbf{q}}^{(k)}_i - \mathbf{q}^{(k)}_i \right\|_2, \quad \mathbf{q}^{(k)}_i = \bigl(q_x^{(k)}, q_y^{(k)}\bigr)$$

$$e_{q'}^{(i,k)} = \left\| \hat{\mathbf{q}}'^{(k)}_i - \mathbf{q}'^{(k)}_i \right\|_2, \quad \mathbf{q}'^{(k)}_i = \bigl(q'^{(k)}_x, q'^{(k)}_y\bigr)$$

在测试集上对节点 $k$ 取均值：

$$\bar{e}_q^{(k)} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} e_q^{(i,k)}, \quad \bar{e}_{q'}^{(k)} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} e_{q'}^{(i,k)}$$

##### 绘图（行 73–160）

根据 `datacase` 分别绘制三张图：

1. **图 1**：测试集上预测曲线 $(q_x, q_y)$ 与真实曲线的对比折线图，黑色实线为真实值，红色虚线+菱形点为预测值。
2. **图 2**：切向量 $(q'_x, q'_y)$ 的散点图，叠加单位圆（$\|\mathbf{q}'\|=1$）以验证单位切向量约束是否满足。
3. **图 3**：逐节点平均误差 $\bar{e}_q^{(k)}$ 和 $\bar{e}_{q'}^{(k)}$ 沿弧长节点编号的折线图。

---

### 15.4 `SavedParameters.py`

#### `hyperparams(case, pctg)` 函数（行 4–24）

```python
def hyperparams(case, pctg):
    best_vals = pd.DataFrame()
    best_vals["percentage_train"] = np.array([0.1, 0.2, 0.4, 0.8, 0.8])
    best_vals["datacase"]         = np.array([1,   1,   1,   1,   2  ])
    best_vals["gamma"]            = np.array([0.007044405451814177,
                                               0.006335851468590373,
                                               0.009004175808977003,
                                               0.003853035138801786,
                                               0.0073229668983443436])
    best_vals["n_layers"]         = np.array([4, 4, 4, 4, 3])
    best_vals["hidden_nodes"]     = np.array([950, 978, 997, 985, 616])

    vals = best_vals[(best_vals['datacase'] == case) & (best_vals['percentage_train'] == pctg)]

    nlayers      = vals.iloc[0]["n_layers"]
    hidden_nodes = vals.iloc[0]["hidden_nodes"]
    gamma        = vals.iloc[0]["gamma"]

    params = {'n_layers': int(nlayers), 'hidden_nodes': int(hidden_nodes), 'gamma': gamma}
    return params
```

该函数存储了通过 Optuna 搜索得到的各实验配置下最优超参数，如下表所示。

| `datacase` | `percentage_train` | $\gamma$ | `n_layers` | `hidden_nodes` |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.1 | $7.044 \times 10^{-3}$ | 4 | 950 |
| 1 | 0.2 | $6.336 \times 10^{-3}$ | 4 | 978 |
| 1 | 0.4 | $9.004 \times 10^{-3}$ | 4 | 997 |
| 1 | 0.8 | $3.853 \times 10^{-3}$ | 4 | 985 |
| 2 | 0.8 | $7.323 \times 10^{-3}$ | 3 | 616 |

根据传入的 `case` 和 `pctg` 进行行过滤，返回对应的 `params` 字典，供 `define_best_model` 使用。

---

## 总结

`DiscreteNetwork_main.py` 实现了一个完整的**神经网络代理模型（Surrogate Model）**流程，用于近似求解欧拉弹性体（Euler's Elastica）的离散化边值问题：

1. **问题设置**：给定弹性链两端节点的边界条件（位置 $\mathbf{q}^{(0)}, \mathbf{q}^{(N-1)}$ 和切向量 $\mathbf{q}'^{(0)}, \mathbf{q}'^{(N-1)}$），预测所有内部节点的位置和切向量。

2. **神经网络结构**：多层感知机（MLP），输入维度 8，输出维度 $4(N-2)$，通过添加差分损失 $\gamma \cdot \mathcal{L}_{\text{diff}}$ 使网络学习到曲线的局部几何一致性。

3. **超参数优化**：通过 Optuna 框架在 $n\_\text{layers} \in [0,10]$、$d_h \in [10,1000]$、$\gamma \in [0, 10^{-2}]$ 的空间内搜索最优配置，以验证集 MSE 为最小化目标。

4. **训练目标**（总损失）：

$$\boxed{\mathcal{L}_{\text{total}} = \underbrace{\frac{1}{B}\sum_{i=1}^B \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2}_{\mathcal{L}_{\text{MSE}}} + \gamma \cdot \underbrace{\frac{1}{B}\sum_{i=1}^B \|\Delta\hat{\mathbf{z}}_i - \Delta\mathbf{z}_i\|_2^2}_{\mathcal{L}_{\text{diff}}}}$$
