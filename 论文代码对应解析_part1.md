# 论文公式与代码对应解析 —— Part 1：第 2、3 节

本文档以"代码在前、公式在后"的格式，逐一解读论文 *Neural networks for the approximation of Euler's elastica* 中各节公式与仓库底层计算代码之间的对应关系。所有代码均来自仓库 [SiyaoCao/LearningEulersElastica](https://github.com/SiyaoCao/LearningEulersElastica/tree/copilot/explain-continuousnetwork-main)，代码链接直达具体文件与行号。

---

## 第 2 节：欧拉弹性曲线模型（Euler's elastica model）

### 2.0 弯曲能量泛函（Euler-Bernoulli 能量）

论文在第 2 节开头引入了弹性曲线的能量泛函，即需最小化的弯曲能量积分。该最小化过程在数据生成 Notebook 中由 SciPy 优化器执行。

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 22：主求解循环）

```python
# Cell 22 —— 最小化弯曲能量（bending_Sd_bc），即对弯曲能量泛函的离散化进行最小化
for i in range(np.size(N)):
    
    # initial guess
    Qspline = ip.CubicSpline((0,(param.n_nodes-1)*s_step), (Q0,Qn), bc_type=((1,V0[i,:]),(1,Vn[i,:])))
    Vspline = Qspline.derivative()
    QV = np.hstack((Qspline(coord),Vspline(coord))).transpose()
    
    cvsplines[i,:] = QV.flatten(order='F')
    QVL = joinQVL(QV,Lambda[i,:],param)
    
    QVL1, QVLbound  = separateBC(QVL)
    
    # solution of min(Sd)
    sol_min = op.minimize(bending_Sd_bc, QVL1.flatten(order='F'),
                          args=(QVLbound.flatten(order='F'), param,),
                          method='trust-constr', constraints=cons,
                          options={'maxiter': 10000})
    print(sol_min.message)
    
    QVL2 = inflateQVL(sol_min.x, param, length=param.n_nodes-2)
    QVL2_all = joinBC(QVL2, QVLbound, param)
    QV2, L0 = separateQVL(QVL2_all, param)
    QVL_min = joinQVL(QV2, np.hstack([0,np.array(sol_min.v).flatten(),0]), param)
    print('sol min, action', complete_Sd(QVL_min.flatten(order='F'), param))
    
    QVL2_full = joinBC(QVL2, QVLbound, param)
    QV2, Lambda2 = separateQVL(QVL2_full, param)
    sol_min_matrix[i,:] = QV2.flatten(order='F')
    
    # guarantee the DEL=0 with min solution
    sol = op.root(DEL, QVL2.flatten(order='F'),
                  args=(QVLbound.flatten(order='F'), param),
                  jac=DEL_automatic_jacobian)
    QVL1_sol = inflateQVL(sol.x, param, length=param.n_nodes-2)
    print(sol.message)
    QVL_sol = joinBC(QVL1_sol, QVLbound, param)
    print('sol DEL, action', complete_Sd(QVL_sol.flatten(order='F'), param))
    
    QV_sol, Lambda_sol = separateQVL(QVL_sol, param)
    
    Lambda[i,:] = Lambda_sol.flatten(order='F')
    trajectories1[i,:] = QV_sol.flatten(order='F')
```

**对应公式（能量泛函）：**

\[
\int_0^L \kappa(s)^2 \, ds
\]

其中 \(\kappa(s)\) 为曲线 \(\mathbf{q}(s)\) 的曲率，\(s\) 为弧长参数。该积分即为弹性曲线问题的最小化目标。在代码中，`bending_Sd_bc` 函数计算离散化弯曲能量（见下文公式 (6)），由 `op.minimize` 以 `trust-constr` 方法最小化。

---

### 2.1 拉格朗日函数（公式 1）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 7）

```python
# Cell 7 —— 拉格朗日函数定义
def Lagrangian(q, v, a, param):
    # 二阶拉格朗日：L(q, q', q'') = 1/2 * EI * ||q''||^2
    # axis=0 若第一维给出问题的维数
    return 0.5 * param.EI * np.sum(a**2, axis=0)
```

其中 `param.EI = 10` 为弯曲刚度，`a` 对应曲率向量 \(\mathbf{q}''\)，`np.sum(a**2, axis=0)` 计算 \(\|\mathbf{q}''\|^2\)。

**对应公式（公式 1）：**

\[
\mathcal{L}(\mathbf{q}, \mathbf{q}', \mathbf{q}'') = \frac{1}{2} EI \|\mathbf{q}''\|^2
\]

该拉格朗日量定义在二阶切丛 \(T^{(2)}Q\) 上，\(EI\) 为弯曲刚度（由材料弹性模量 \(E\) 与截面二次矩 \(I\) 之积给出），\('\) 表示空间导数。

---

### 2.2 约束方程（公式 2）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 7）

```python
# Cell 7 —— 弧长约束（inextensibility constraint）
def arclength(v, Lambda, param):
    # Phi(q, q') = ||q'||^2 - 1 = 0（无轴向应变）
    return Lambda * (np.sum(v**2, axis=0) - 1)
```

其中 `v` 对应切向量 \(\mathbf{q}'\)，`np.sum(v**2, axis=0) - 1` 即 \(\|\mathbf{q}'\|^2 - 1\)，由拉格朗日乘子 `Lambda` 加权进入增广拉格朗日函数。

在连续网络训练中，该约束以惩罚项形式出现在损失函数里：

GitHub 链接：[ContinuousNetwork/Scripts/Training.py，第 37 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Training.py#L37)

```python
# ContinuousNetwork/Scripts/Training.py，第 37 行
# 惩罚项：gamma * (||q'||^2 - 1)^2，强制切向量单位化
loss += 1e-2 * torch.mean(
    (torch.linalg.norm(model.derivative(s, q1, q2, v1, v2), ord=2, dim=1)**2 - 1.)**2
)
```

**对应公式（公式 2）：**

\[
\Phi(\mathbf{q}, \mathbf{q}') = \|\mathbf{q}'\|^2 - 1 = 0
\]

该约束强制曲线以弧长参数化，即切向量 \(\mathbf{q}'(s)\) 处处具有单位范数，从而保证曲线的不可伸长性（inextensibility）。

---

### 2.3 增广拉格朗日函数（公式 3）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 8）

```python
# Cell 8 —— 离散拉格朗日（结合 Lagrangian 与弧长约束）
def discrete_L(qk, vk, qkp1, vkp1, param):
    # 计算离散二阶导数近似（曲率近似）
    ak  = ((-2*vkp1 - 4*vk ) * s_step + 6*(qkp1 - qk)) / s_step**2
    akp1 = ((4*vkp1 + 2*vk ) * s_step - 6*(qkp1 - qk)) / s_step**2
    # 离散拉格朗日（梯形规则）
    Ld_k   = Lagrangian(qk,   vk,   ak,   param)   # L(q_k, q'_k, a_k)
    Ld_kp1 = Lagrangian(qkp1, vkp1, akp1, param)  # L(q_{k+1}, q'_{k+1}, a_{k+1})
    return 0.5 * s_step * (Ld_k + Ld_kp1)

def discrete_arclength(vk, lambdak, param):
    # lambda_k * (||q'_k||^2 - 1)（内部节点处的约束项）
    return arclength(vk, lambdak, param)
```

该函数同时实现了增广拉格朗日的离散化，将 `Lagrangian`（对应 \(\mathcal{L}\)）和 `discrete_arclength`（对应 \(\Lambda \Phi\)）组合在 Cell 15 的 `complete_Sd` 中（见下文公式 6）。

**对应公式（公式 3）：**

\[
\tilde{\mathcal{L}}(\mathbf{q}, \mathbf{q}', \mathbf{q}'', \Lambda)
= \mathcal{L}(\mathbf{q}, \mathbf{q}', \mathbf{q}'') + \Lambda \Phi(\mathbf{q}, \mathbf{q}')
\]

其中 \(\Lambda(s)\) 为拉格朗日乘子，负责将约束 \(\Phi = 0\) 嵌入到变分原理中。代码中 `Lambda`（`lambdak`）数组存储每个节点处的乘子数值。

---

### 2.4 连续作用量泛函（公式 4）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 15）

```python
# Cell 15 —— 完整离散作用量 S_d（对应连续作用量 S 的离散化）
def complete_Sd(QVL_flat, param):
    # 求和所有节点对的离散拉格朗日（弯曲能量部分）
    Ld_all = np.sum(apply_along_axis(oneD_Ld, 0, QVL_flat, param))
    # 求和所有内部节点的弧长约束项
    gl_all = np.sum(oneD_arclen(QVL_flat, param))
    return Ld_all + gl_all
```

其中 `oneD_Ld` 对每一段 \([s_k, s_{k+1}]\) 计算 \(\tilde{\mathcal{L}}_d\)，`oneD_arclen` 计算 \(\lambda_k(\|\mathbf{q}'_k\|^2 - 1)\)，两者之和近似连续作用量 \(\mathcal{S}[\mathbf{q}]\)。

Cell 13 实现了 `oneD_Ld` 和 `oneD_arclen`（辅助函数）：

```python
# Cell 13
def oneD_Ld(QVL_flat, param):
    QV = separateQVL(inflateQVL(QVL_flat, param, param.n_nodes), param)[0]
    Qk   = QV[:,0,0:-1]
    Vk   = QV[:,1,0:-1]
    Qkp1 = QV[:,0,1:]
    Vkp1 = QV[:,1,1:]
    return discrete_L(Qk, Vk, Qkp1, Vkp1, param)

def oneD_arclen(QVL_flat, param):
    QV, Lambda = separateQVL(inflateQVL(QVL_flat, param, param.n_nodes), param)
    Vk      = QV[:,1,1:-1]   # 仅内部节点
    Lambdak = Lambda[1:-1]   # 仅内部节点
    return discrete_arclength(Vk, Lambdak, param)
```

**对应公式（公式 4）：**

\[
\mathcal{S}[\mathbf{q}] = \int_0^L \tilde{\mathcal{L}}(\mathbf{q}, \mathbf{q}', \mathbf{q}'', \Lambda) \, ds
\]

该泛函以弧长 \(s\) 为自变量，对增广拉格朗日函数沿整条曲线积分。代码中以梯形规则对积分进行离散化，`complete_Sd` 即为其离散近似。

---

### 2.5 曲率近似公式（离散化辅助公式）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 8）

```python
# Cell 8 —— 基于低阶导数对 q'' 的分段线性近似
def discrete_L(qk, vk, qkp1, vkp1, param):
    # 区间左端（s_k）处的曲率近似：(q''_k)^-
    ak   = ((-2*vkp1 - 4*vk ) * s_step + 6*(qkp1 - qk)) / s_step**2

    # 区间右端（s_{k+1}）处的曲率近似：(q''_{k+1})^+
    akp1 = ((4*vkp1 + 2*vk ) * s_step - 6*(qkp1 - qk)) / s_step**2

    Ld_k   = Lagrangian(qk,   vk,   ak,   param)
    Ld_kp1 = Lagrangian(qkp1, vkp1, akp1, param)
    return 0.5 * s_step * (Ld_k + Ld_kp1)
```

`s_step` 为节点间距 \(h = L / N\)，`vk` 对应 \(\mathbf{q}'_k\)，`vkp1` 对应 \(\mathbf{q}'_{k+1}\)，`qk`、`qkp1` 对应相邻节点位置。

**对应公式（两个曲率近似）：**

\[
\mathbf{q}''(s_k) \approx (\mathbf{q}''_k)^{-}
= \frac{-2\mathbf{q}'_{k+1} - 4\mathbf{q}'_k}{h} + \frac{6(\mathbf{q}_{k+1} - \mathbf{q}_k)}{h^2}
\]

\[
\mathbf{q}''(s_{k+1}) \approx (\mathbf{q}''_{k+1})^{+}
= \frac{4\mathbf{q}'_{k+1} + 2\mathbf{q}'_k}{h} - \frac{6(\mathbf{q}_{k+1} - \mathbf{q}_k)}{h^2}
\]

这是对区间 \([s_k, s_{k+1}]\) 上曲率的分段线性不连续近似。代码中 `ak` 和 `akp1` 分别对应两端的曲率向量，将其代入 `Lagrangian()` 即得到离散化的二阶拉格朗日量。

---

### 2.6 离散拉格朗日函数（无编号公式）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 8）

```python
# Cell 8 —— 离散拉格朗日（梯形规则）
def discrete_L(qk, vk, qkp1, vkp1, param):
    ak   = ((-2*vkp1 - 4*vk ) * s_step + 6*(qkp1 - qk)) / s_step**2
    akp1 = ((4*vkp1 + 2*vk ) * s_step - 6*(qkp1 - qk)) / s_step**2
    Ld_k   = Lagrangian(qk,   vk,   ak,   param)
    Ld_kp1 = Lagrangian(qkp1, vkp1, akp1, param)
    # 梯形规则：(h/2) * [L(q_k, q'_k, a_k) + L(q_{k+1}, q'_{k+1}, a_{k+1})]
    return 0.5 * s_step * (Ld_k + Ld_kp1)
```

**对应公式（离散拉格朗日）：**

\[
\tilde{\mathcal{L}}_d(\mathbf{q}_k, \mathbf{q}'_k, \mathbf{q}_{k+1}, \mathbf{q}'_{k+1}, \Lambda_k, \Lambda_{k+1})
= \frac{h}{2} \left[
  \tilde{\mathcal{L}}(\mathbf{q}_k, \mathbf{q}'_k, (\mathbf{q}''_k)^{-}, \Lambda_k)
  + \tilde{\mathcal{L}}(\mathbf{q}_{k+1}, \mathbf{q}'_{k+1}, (\mathbf{q}''_{k+1})^{+}, \Lambda_{k+1})
\right]
\]

该离散拉格朗日基于梯形规则，对连续增广拉格朗日量 \(\tilde{\mathcal{L}}\) 在相邻节点 \(s_k\) 和 \(s_{k+1}\) 处进行评估和加权求和。代码中 `0.5 * s_step * (Ld_k + Ld_kp1)` 即对应 \(\frac{h}{2}[\cdot]\)。

---

### 2.7 连续欧拉-拉格朗日方程（公式 5）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 18）

```python
# Cell 18 —— 离散欧拉-拉格朗日方程（DEL），即公式 (5) 的离散化
def DEL(QVL1_flat, QVLbound_flat, param):
    QVL1    = inflateQVL(QVL1_flat,    param, param.n_nodes-2)
    QVLbound = inflateQVL(QVLbound_flat, param, 2)
    # 对作用量 complete_Sd 关于内部节点 QVL1 自动求梯度，令其等于零
    return ag.grad(
        lambda y: complete_Sd(joinBC(y, QVLbound, param).flatten(order='F'), param)
    )(QVL1).flatten(order='F')

# Cell 18 —— DEL 的雅可比矩阵（用于 op.root 求解）
def DEL_automatic_jacobian(QVL1_flat, QVLbound_flat, param):
    return ag.jacobian(lambda y: DEL(y, QVLbound_flat, param))(QVL1_flat)
```

连续欧拉-拉格朗日方程通过对离散作用量 `complete_Sd` 关于内部节点的自动微分（`autograd.grad`）来实现，即令作用量对内部节点的梯度为零。

**对应公式（公式 5）：**

\[
\frac{d^2}{ds^2}\!\left(\frac{\partial \mathcal{L}}{\partial \mathbf{q}''}\right)
- \frac{d}{ds}\!\left(\frac{\partial \mathcal{L}}{\partial \mathbf{q}'}\right)
+ \frac{\partial \mathcal{L}}{\partial \mathbf{q}}
= \frac{d}{ds}\!\left(\frac{\partial \Phi}{\partial \mathbf{q}'}\Lambda\right)
  - \frac{\partial \Phi}{\partial \mathbf{q}}\Lambda,
\qquad
\|\mathbf{q}'\|^2 - 1 = 0
\]

该方程需与边界条件 \((\mathbf{q}(0), \mathbf{q}'(0)) = (\mathbf{q}_0, \mathbf{q}'_0)\) 和 \((\mathbf{q}(L), \mathbf{q}'(L)) = (\mathbf{q}_N, \mathbf{q}'_N)\) 联立求解。代码中 `DEL=0` 即为该方程的离散近似，由 `op.root` 数值求解。

---

### 2.8 离散作用量（公式 6）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 15）

```python
# Cell 15 —— 完整离散作用量 S_d
def complete_Sd(QVL_flat, param):
    # 第一项：sum_{k=0}^{N-1} L_d(q_k, q'_k, q_{k+1}, q'_{k+1})
    Ld_all = np.sum(apply_along_axis(oneD_Ld, 0, QVL_flat, param))
    # 第二项：sum_{k=1}^{N-1} lambda_k * (||q'_k||^2 - 1)
    gl_all = np.sum(oneD_arclen(QVL_flat, param))
    return Ld_all + gl_all
```

两项的分别对应：`Ld_all` 为所有区间的离散拉格朗日之和（弯曲能量部分），`gl_all` 为所有内部节点处弧长约束惩罚之和。

**对应公式（公式 6）：**

\[
\mathcal{S}_d = \sum_{k=0}^{N-1}
\tilde{\mathcal{L}}_d(\mathbf{q}_k, \mathbf{q}'_k, \mathbf{q}_{k+1}, \mathbf{q}'_{k+1}, \Lambda_k, \Lambda_{k+1})
\]

该离散作用量是连续作用量泛函 \(\mathcal{S}[\mathbf{q}]\)（公式 4）的离散近似。注意约束项 \(\lambda_k(\|\mathbf{q}'_k\|^2 - 1)\) 已经被包含在 \(\tilde{\mathcal{L}}_d\) 中（通过 `oneD_arclen` 加入 `complete_Sd`），从而完整表达增广离散作用量。

---

### 2.9 离散欧拉-拉格朗日方程（公式 7）

**对应代码（数据生成 Notebook）：**

GitHub 链接：[DataSets/elastica_Lagrangian_static.ipynb](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DataSets/elastica_Lagrangian_static.ipynb)（Cell 18 及 Cell 22）

```python
# Cell 18 —— 离散欧拉-拉格朗日方程（DEL）的函数定义
def DEL(QVL1_flat, QVLbound_flat, param):
    QVL1     = inflateQVL(QVL1_flat,    param, param.n_nodes-2)
    QVLbound = inflateQVL(QVLbound_flat, param, 2)
    # D_3 L_d + D_1 L_d = 0，D_4 L_d + D_2 L_d = 0，D_6 L_d + D_5 L_d = 0
    # 通过 autograd 自动计算 partial S_d / partial (q_k, q'_k, lambda_k)
    return ag.grad(
        lambda y: complete_Sd(joinBC(y, QVLbound, param).flatten(order='F'), param)
    )(QVL1).flatten(order='F')

def DEL_automatic_jacobian(QVL1_flat, QVLbound_flat, param):
    return ag.jacobian(lambda y: DEL(y, QVLbound_flat, param))(QVL1_flat)
```

```python
# Cell 22 —— 用 op.root 验证 DEL=0（同时满足离散变分原理）
sol = op.root(DEL, QVL2.flatten(order='F'),
              args=(QVLbound.flatten(order='F'), param),
              jac=DEL_automatic_jacobian)
```

**对应公式（公式 7）：**

\[
D_3 \tilde{\mathcal{L}}_d(\mathbf{q}_{k-1}, \mathbf{q}'_{k-1}, \mathbf{q}_k, \mathbf{q}'_k, \Lambda_{k-1}, \Lambda_k)
+ D_1 \tilde{\mathcal{L}}_d(\mathbf{q}_k, \mathbf{q}'_k, \mathbf{q}_{k+1}, \mathbf{q}'_{k+1}, \Lambda_k, \Lambda_{k+1}) = 0,
\]
\[
D_4 \tilde{\mathcal{L}}_d(\mathbf{q}_{k-1}, \mathbf{q}'_{k-1}, \mathbf{q}_k, \mathbf{q}'_k, \Lambda_{k-1}, \Lambda_k)
+ D_2 \tilde{\mathcal{L}}_d(\mathbf{q}_k, \mathbf{q}'_k, \mathbf{q}_{k+1}, \mathbf{q}'_{k+1}, \Lambda_k, \Lambda_{k+1}) = 0,
\]
\[
D_6 \tilde{\mathcal{L}}_d(\mathbf{q}_{k-1}, \mathbf{q}'_{k-1}, \mathbf{q}_k, \mathbf{q}'_k, \Lambda_{k-1}, \Lambda_k)
+ D_5 \tilde{\mathcal{L}}_d(\mathbf{q}_k, \mathbf{q}'_k, \mathbf{q}_{k+1}, \mathbf{q}'_{k+1}, \Lambda_k, \Lambda_{k+1}) = 0,
\]

对 \(k = 1, \ldots, N-1\)，其中 \(D_i\) 表示对第 \(i\) 个自变量求偏导。代码用 `autograd.grad` 自动计算这些偏导数，整体表达为对内部节点的梯度条件 \(\nabla S_d = 0\)，再由 `op.root` 求解 `DEL(QVL1_flat, QVLbound_flat, param) = 0`。

---

## 第 3 节：用神经网络近似（Approximation with Neural Networks）

### 3.1 神经网络的层次组合（公式 8）

**对应代码——离散网络（DiscreteNetwork_main.py）：**

GitHub 链接：[DiscreteNetwork_main.py，第 90–102 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork_main.py#L90)

```python
# DiscreteNetwork_main.py，第 90–102 行
# 网络前向传播：f_ell ∘ ... ∘ f_j ∘ ... ∘ f_1(x)
def forward(self, x):

    if self.is_norm:
        # 输入标准化：q_x 分量归一化到 [-1, 1]
        x[:,0] = (x[:,0] - 1.5) / 1.5
        x[:,4] = (x[:,4] - 1.5) / 1.5
    # f_1 层：线性 + 激活
    x = self.act(self.first(x))
    for i in range(self.nlayers):
        if self.is_res:  # 残差网络（ResNet）
            x = x + self.act(self.linears[i](x))
        else:  # 多层感知机（MLP）
            x = self.act(self.linears[i](x))
    # 输出层（无激活）
    return self.last(x)
```

**对应代码——连续网络（ContinuousNetwork/Scripts/Network.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Network.py，第 171–180 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Network.py#L171)

```python
# ContinuousNetwork/Scripts/Network.py，第 171–180 行
# forward 方法：将 parametric_part（神经网络主体，公式 8）与边界修正组合
def forward(self, s, q1, q2, v1, v2):
    if self.is_norm:
        q1n, q2n, v1n, v2n = self.normalize(q1, q2, v1, v2)
        net_part = self.parametric_part(s, q1n, q2n, v1n, v2n)
    else:
        net_part = self.parametric_part(s, q1, q2, v1, v2)
    if self.correct_functional:
        correction_bcs = self.correction_bcs(s, q1, q2, v1, v2)
        return net_part + correction_bcs
    else:
        return net_part
```

**对应公式（公式 8）：**

\[
f_{\rho} \coloneqq f_{\ell} \circ \cdots \circ f_j \circ \cdots \circ f_1
\]

神经网络为多个变换 \(f_j\)（层）的复合，参数 \(\rho \in \Psi\)。代码中，`self.first`、`self.linears`、`self.last` 分别对应嵌入层、隐藏层序列和输出投影层，通过 `forward` 方法顺序组合（即函数复合）。

---

### 3.2 MLP 层（公式 9）

**对应代码——离散网络（DiscreteNetwork_main.py）：**

GitHub 链接：[DiscreteNetwork_main.py，第 95–100 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork_main.py#L95)

```python
# DiscreteNetwork_main.py，第 95–100 行
# MLP 层：f_j^MLP(x) = sigma(A_j * x + b_j)
x = self.act(self.first(x))           # 第一层：sigma(A_1 x + b_1)
for i in range(self.nlayers):
    if self.is_res:  # 残差
        x = x + self.act(self.linears[i](x))   # x + sigma(A_{i+1} x + b_{i+1})
    else:            # 纯 MLP
        x = self.act(self.linears[i](x))        # sigma(A_{i+1} x + b_{i+1})
```

**对应代码——连续网络 MLP 分支（ContinuousNetwork/Scripts/Network.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Network.py，第 93–103 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Network.py#L93)

```python
# ContinuousNetwork/Scripts/Network.py，第 93–103 行（parametric_part 的 MLP/ResNet 分支）
else:
    # 提升层
    input = self.act(self.lift(input))

    for i in range(self.nlayers):
        if self.is_res:  # 残差网络（ResNet）：x + sigma(A x + b)
            input = input + self.linearsO[i](self.act(self.linears[i](input)))
        else:            # 多层感知机（MLP）：sigma(A x + b)
            input = self.act(self.linears[i](input))

output = self.proj(input)
return output
```

**对应公式（公式 9）：**

\[
f_j^{\mathrm{MLP}}(\mathbf{x}) = \sigma(\mathbf{A}_j \mathbf{x} + \mathbf{b}_j) \in \mathbb{R}^{n_j}
\]

其中 \(\sigma\) 为逐分量作用的非线性激活函数（代码中默认为 `tanh`），\(\mathbf{A}_j \in \mathbb{R}^{n_j \times n_{j-1}}\) 和 \(\mathbf{b}_j \in \mathbb{R}^{n_j}\) 分别为第 \(j\) 层的权重矩阵和偏置向量（即 `nn.Linear` 层的参数），\(n_j\) 为该层输出维度（即 `hidden_nodes`）。

---

### 3.3 损失函数（无编号）

**对应代码——离散网络训练（DiscreteNetwork/Scripts/Training.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/Training.py，第 37–56 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/Training.py#L37)

```python
# DiscreteNetwork/Scripts/Training.py，第 37–56 行
for _, data in enumerate(trainloader):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()
    predicted = model(inputs)

    # 基础 MSE 损失：(1/M) * sum ||q_rho^d(x^i) - y^i||^2
    loss = criterion(predicted, labels)

    # 拼接边界条件后计算带权重的前向差分
    predicted = torch.cat((inputs[:,:4], predicted, inputs[:,4:]), dim=1)
    labels    = torch.cat((inputs[:,:4], labels,    inputs[:,4:]), dim=1)

    predicted_first   = predicted[:,:-4]
    predicted_forward = predicted[:,4:]

    labels_first   = labels[:,:-4]
    labels_forward = labels[:,4:]

    diff_predicted = predicted_forward - predicted_first  # (G * q_rho^d)(x^i)
    diff_labels    = labels_forward    - labels_first     # (G * y^i)

    # 加权差分项：gamma * ||G*(q_rho^d(x^i) - y^i)||^2
    loss += gamma * criterion(diff_predicted, diff_labels)

    train_loss += loss.item()
    loss.backward()
    optimizer.step()
```

**对应代码——连续网络训练（ContinuousNetwork/Scripts/Training.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Training.py，第 35–37 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Training.py#L35)

```python
# ContinuousNetwork/Scripts/Training.py，第 35–37 行
res_q = model(s, q1, q2, v1, v2)
# MSE on positions + MSE on tangents + gamma * inextensibility penalty
loss = criterion(res_q, qs) \
     + criterion(model.derivative(s, q1, q2, v1, v2), vs) \
     + 1e-2 * torch.mean(
           (torch.linalg.norm(model.derivative(s, q1, q2, v1, v2), ord=2, dim=1)**2 - 1.)**2
       )
```

**对应公式（损失函数，无编号）：**

\[
\mathrm{Loss}(\rho) = \frac{1}{M} \sum_{i=1}^{M} \left\| f_{\rho}(\mathbf{x}^i) - \mathbf{y}^i \right\|^2
\]

该均方误差（MSE）损失度量网络预测 \(f_{\rho}(\mathbf{x}^i)\) 与真实输出 \(\mathbf{y}^i\) 之间的距离。离散网络中还额外添加了基于 \(G = S^4 - I\)（前向移位算子）的加权项 \(\gamma \|G \cdot (\hat{\mathbf{y}} - \mathbf{y})\|^2\)，以增强相邻节点差分的一致性；连续网络中则加入了切向量规范化约束惩罚项 \(\gamma(\|\mathbf{q}'\|^2 - 1)^2\)。

---

### 3.4 梯度下降（GD / Adam，无编号）

**对应代码（DiscreteNetwork/Scripts/Training.py 与 ContinuousNetwork/Scripts/Training.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/Training.py，第 58–61 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/Training.py#L58)

```python
# DiscreteNetwork/Scripts/Training.py，第 58–61 行
train_loss += loss.item()

loss.backward()       # 计算 nabla Loss(rho^(k))（反向传播）
optimizer.step()      # rho^(k) <- rho^(k) - eta * nabla Loss(rho^(k))（Adam 步）
counter += 1
```

GitHub 链接：[ContinuousNetwork/Scripts/Training.py，第 38–40 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Training.py#L38)

```python
# ContinuousNetwork/Scripts/Training.py，第 38–40 行
loss.backward()       # 反向传播
optimizer.step()      # Adam 参数更新
running_loss += loss.item()
```

`optimizer` 在主入口文件中定义，例如 `ContinuousNetwork_main.py` 中使用 `torch.optim.Adam(...)`，学习率 \(\eta = 5 \times 10^{-3}\)，并配有 `StepLR` 调度器（每 45 个 epoch 将学习率乘以 0.1）。

**对应公式（梯度下降更新，无编号）：**

\[
\rho^{(k)} \mapsto \rho^{(k)} - \eta \nabla \mathrm{Loss}\!\left(\rho^{(k)}\right)
\coloneqq \rho^{(k+1)}
\]

论文中使用的是 Adam 优化器——梯度下降的加速变体，通过 `optimizer.step()` 执行参数更新。`loss.backward()` 计算 \(\nabla \mathrm{Loss}(\rho^{(k)})\)（通过 PyTorch 自动微分），随后 `optimizer.step()` 按 Adam 规则更新参数 \(\rho\)。

---

### 3.5 边界条件（公式 10）

**对应代码——数据加载（ContinuousNetwork/Scripts/GetData.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/GetData.py，第 100–113 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/GetData.py#L100)

```python
# ContinuousNetwork/Scripts/GetData.py，第 100–113 行
for sample in range(number_samples):
    for node in list_nodes:
        if sample < Ntrain:
            data_train["s"].append(x_range[node])
            data_train["sample_number"].append(sample)

            # 左端边界条件：q1 = q(0)，v1 = q'(0)
            data_train["q1"].append(trajectories_train[sample][0:2])    # q_0^i
            data_train["v1"].append(trajectories_train[sample][2:4])    # (q_0^i)'

            # 右端边界条件：q2 = q(L)，v2 = q'(L)
            data_train["q2"].append(trajectories_train[sample][-4:-2])  # q_N^i
            data_train["v2"].append(trajectories_train[sample][-2:])    # (q_N^i)'

            # 节点 s_k 处的真实值
            data_train["qs"].append(trajectories_train[sample][4*node:4*node+2])
            data_train["vs"].append(trajectories_train[sample][4*node+2:4*node+4])
```

**对应代码——离散网络数据加载（DiscreteNetwork/Scripts/GetData.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/GetData.py，第 51–53 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/GetData.py#L51)

```python
# DiscreteNetwork/Scripts/GetData.py，第 51–53 行
# 输入 x^i：边界条件（q_0^i, (q_0^i)', q_N^i, (q_N^i)'）
x_full_train = np.concatenate((data_train[:,:4], data_train[:,-4:]), axis=1)
# 输出 y^i：内部节点处的位置和切向量
y_full_train = data_train[:,4:-4]
```

**对应代码——边界条件提取工具函数（ContinuousNetwork/Scripts/Utils.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Utils.py，第 5–9 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Utils.py#L5)

```python
# ContinuousNetwork/Scripts/Utils.py，第 5–9 行
def getBCs(trajectories):
    bcs = {
        "q1": trajectories[:, :2],    # q(0)：左端点位置
        "q2": trajectories[:,-4:-2],  # q(L)：右端点位置
        "v1": trajectories[:, 2:4],   # q'(0)：左端点切向量
        "v2": trajectories[:, -2:]    # q'(L)：右端点切向量
    }
    return bcs
```

**对应公式（公式 10）：**

\[
\left\{
\mathbf{q}^i(0) = \mathbf{q}_0^i,\quad
\mathbf{q}^i(L) = \mathbf{q}_N^i,\quad
(\mathbf{q}^i)'(0) = (\mathbf{q}_0^i)^{\prime},\quad
(\mathbf{q}^i)'(L) = (\mathbf{q}_N^i)^{\prime}
\right\}
\]

其中 \((\mathbf{q}_0^i, \mathbf{q}_N^i, (\mathbf{q}_0^i)', (\mathbf{q}_N^i)') \in \mathbb{R}^8\)。代码中，每条弹性曲线轨迹的前 4 个分量存储左端边界条件（`q1`、`v1`），最后 4 个分量存储右端边界条件（`q2`、`v2`），组成输入特征向量 \(\mathbf{x}^i \in \mathbb{R}^8\)。

---

*本文档继续于 [论文代码对应解析_part2.md](./论文代码对应解析_part2.md)，涵盖第 4、5 节及附录 A 的全部公式。*
