# 作品代码说明

项目简介

> 为了更好地在量子机器学习领域实现Transformer中的注意力机制，本作品主要研究了基于变分量子算法的图Transformer模型架构，并针对其消耗线路资源过多带来的贫瘠高原等问题提出了结构自适应的模型优化方案。
>
> 本项目为结构自适应的量子经典混合图Transformer模型的仿真实现代码。

代码说明

> 以下主要针对作品书中提及的两种结构搜索框架，实现模型方案的仿真程序。具体为`基于超网的对抗搜索(Supernet-based Adversarial Bandit, SAB)框架`与`梯度-代价多目标交替（Gradient-Cost Multi-objective Alternate, GCMA）搜索框架`。

## 基本信息

> 下面为文件夹中各文件的简要介绍，并说明可手动设置的参数。

### SAB方案

SAB的仿真程序包括源代码`SAB_TRI.cpp`和头文件`qcirc.h`。

- `qcirc.h`：主要实现量子计算模拟器
- `SAB_TRI.cpp`：主要实现对数据集`TRI_n.json`（n取5~9）的读取、量子线路的构建和SAB算法的主体部分

### GCMA方案
GCMA的仿真程序包括源代码`GCMA_TRI.cpp`、`GCMA_BENZENE.cpp`、`GCMA_IMDB.cpp`和头文件`qcirc.h`、`qrand.h`、`qsubnet.h`。

- `qcirc.h`：主要实现量子计算模拟器
- `qrand.h`：主要实现随机数相关的函数
- `qsubnet.h`：主要实现量子线路结构的跨层约束和子网生成

源代码主要实现对数据集的读取、量子线路的构建和GCMA算法的主体部分。其中：

- `GCMA_TRI.cpp`：读取数据集`TRI_n.json`（n取5~9）
- `GCMA_BENZENE.cpp`：读取数据集`BENZENE_DETECTION_8.json`
- `GCMA_IMDB.cpp`：读取数据集`IMDB-MULTI_10.json`

### 源代码中的常数
SAB和GCMA的源代码中都包含以下可以手动修改的常数（定义在文件开始处）：

- `input_file_name`：数据集文件名

  > 如果希望程序读取其它数据集，需修改此常数，并根据数据集的属性设置其它常数（如`n_nodes`、`n_graphs`）

- `n_nodes`：数据集中图的节点个数

- `n_layers`：量子线路层数

  > 一般情况下，层数越多，训练效果越好，但训练速度也越慢

- `n_graphs`：数据集中图的个数

- `n_threads`：并行线程数

  > 由于计算量较大，程序运行速度较慢，有条件时可增加并行线程数以加快训练速度

## 编译与运行
源代码在Windows环境和Linux环境下都可使用
```bash
g++ <file> -O3 -fopenmp
```
由于算法结果受随机数影响，程序需要确定随机数种子。在Windows环境下使用
```bash
<file> <seed>
```
或在Linux环境下使用
```bash
./<file> <seed>
```
运行程序，则程序直接使用`seed`的值作为随机数种子。
若不加参数直接运行程序，则需要手动输入随机数种子。

## 程序输出
程序运行结束后生成两个文件`Records_time_seed`和`Final_result_time_seed`，其中`time`为时间戳，`seed`为随机数种子。`Records_time_seed`保存训练过程中的中间结果和数据，`Final_result_time_seed`保存训练的最终结果。

`Final_result_time_seed`中保存了质量可能较高的一个或多个线路结构。对于每个线路结构，第一行的`n_layers`个16进制数是其每一层的编号，第二行记录其代价函数值和准确率，接下来的若干行是其参数的数值。

