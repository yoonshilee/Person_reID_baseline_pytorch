# Experiment Report-2: Representation Learning

---

## Requirements

### Objectives

This experiment explores representation learning through the lens of Person Re-identification (ReID), a critical task in visual AI with direct implications for AI safety, privacy, and fairness. You will:

- Understand how deep models learn discriminative identity features from images.
- Reproduce a baseline Re-ID system and analyze its limitations.
- Experiment with different network architectures and loss functions.
- Reflect on ethical considerations: When is Re-ID helpful? When could it be harmful?

### Preparation

Before coding, review the following resources:

- Codebase: [https://github.com/layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
- Tutorial (8-min read): [https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial)
- Dataset: Market-1501 (automatically prepared by prepare.py)
- Hardware

Make sure you can run:

```bash
python prepare.py
python test.py --name ft_ResNet50 --which_epoch 59 
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32 
python evaluate_gpu.py
```

### Tasks

Complete the following during your lab session:

1. Follow the tutorial to train and evaluate the baseline ResNet-50 model on Market-1501. Record your final Rank@1 and mAP scores.
2. Answer the “Quick Questions” in the tutorial (e.g., Why use AdaptiveAvgPool2d? Why L2-normalize features?).
3. Try one variant from the codebase or documentation:
   - Change backbone: Use DenseNet (--use_dense) or HRNet (--use_hr).
   - Change loss: Add Circle Loss (--circle) or Instance Loss (--instance).
   - (Optional) Test on DukeMTMC-reID dataset-does the Market-trained model generalize?
4. Identify at least two failure cases (e.g., same person missed due to pose/camera change; wrong match due to similar clothing).

### Lab Report Requirements

- Submit your report before 23:59 on 28 February via UM Moodle.
- Minimum length: 2 pages using the provided LaTeX template.
- Your report must include:
  - Summary of baseline performance and your variant’s results.
  - Answers to at least three “Quick Questions” from the tutorial.
  - Analysis of failure modes with concrete examples (describe or reference image IDs).
  - At least two improvement proposals (e.g., better data augmentation, attention modules, re-ranking, domain adaptation).
  - A short reflection on AI safety: How could Re-ID be misused? What safeguards would you suggest?

### Notes

- The baseline uses train_all (no validation split)-this is intentional for simplicity, but discuss its impact.
- If GPU memory is tight, reduce --batchsize (e.g., to 16) and lower the learning rate accordingly.
- Check the requirements.txt and ensure PyTorch version compatibility.

## Updates

1. Report-1 DDL is extended to 8 Feb.
Report-2 DDL is extended to 8 Mar.
2. If you have completed Report-2, please do Report3 after 8 Mar.
3. In most cases, the error is about path.
   - `./`
   means current folder;
   - `../`  means upper folder;
   - `../../`  means upper folder of upper folder.
   - Or using absolute path like /home/user/Market/pytorch
4. The pseudo-code in the tutorial is for illustrating logic only; it is not meant to be executable.
Please do not modify the code following Tutorial.
5. demo.py is based on the extracted feature, which is saved in the `pytorch_feature.mat`
Please run the test.py to extract feature before running the demo.py.
6. Please use the newly-added `prepare-Duke.py` or modify the `prepare.py` for Duke Dataset.

---

## 中文翻译

### 实验目标

本实验以行人重识别（ReID）为切入点探究表征学习。行人重识别是视觉人工智能领域的核心任务，与AI安全、隐私保护和算法公平性直接相关。你将完成以下内容：

- 理解深度模型如何从图像中学习具有判别性的身份特征
- 复现一套行人重识别基线系统，并分析其局限性
- 针对不同的网络架构和损失函数开展对比实验
- 思考相关伦理问题：行人重识别技术在哪些场景下能发挥正向作用？又可能在哪些场景中造成危害？

### 实验准备

编写代码前，请先学习以下相关资源：

- 代码库：[https://github.com/layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
- 教程（8分钟阅读量）：[https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial)
- 数据集：Market-1501（可通过prepare.py脚本自动完成预处理）
- 硬件环境

请确保你可以成功运行以下指令：

```bash
python prepare.py
python test.py --name ft_ResNet50 --which_epoch 59
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32
python evaluate_gpu.py
```

### 实验任务

请在实验课期间完成以下任务：

1. 参照教程，在Market-1501数据集上完成ResNet-50基线模型的训练与评估，记录最终的首位命中率（Rank@1）和平均精度均值（mAP）指标。
2. 回答教程中的“快速问答”题目（例如：为什么使用自适应平均池化层AdaptiveAvgPool2d？为什么要对特征做L2归一化？）。
3. 基于代码库或说明文档，完成至少一种模型变体实验：
   - 更换主干网络：使用密集连接网络DenseNet（指令参数--use_dense）或高分辨率网络HRNet（指令参数--use_hr）
   - 更换损失函数：添加Circle Loss（指令参数--circle）或Instance Loss（指令参数--instance）
   - （可选）在DukeMTMC-reID数据集上完成测试，验证基于Market数据集训练的模型是否具备泛化能力
4. 找出至少两个模型失效案例（例如：因姿态、相机视角变化导致同一行人匹配失败；因衣着相似导致不同行人错误匹配）。

### 实验报告要求

- 请于2月28日23:59前，通过澳门大学Moodle平台提交报告
- 最低篇幅要求：使用指定LaTeX模板撰写，不少于2页
- 报告必须包含以下内容：

  - 基线模型的性能总结，以及你所设计的模型变体的实验结果
  - 教程中至少3道“快速问答”题目的解答
  - 结合具体案例对模型失效模式进行分析（可描述案例细节或标注对应图像ID）
  - 至少两项模型优化方案（例如：更优的数据增强策略、注意力模块、重排序机制、域自适应方法）
  - 关于AI安全的简短思考：行人重识别技术可能被如何滥用？你认为应当设置哪些防护措施？

### 注意事项

- 基线模型采用train_all模式（无验证集划分），该设置是为了简化实验流程，请在报告中分析该设置带来的影响
- 若GPU显存不足，可减小--batchsize参数（例如调整为16），并对应降低学习率
- 请检查requirements.txt文件，确保PyTorch版本与代码环境兼容

### 更新说明

1. 报告一截止日期延至2月8日。
   报告二截止日期延至3月8日。
2. 若已完成报告二，请在3月8日之后继续完成报告三。
3. 大多数情况下，报错原因与路径有关：
   - `./` 表示当前目录；
   - `../` 表示上一级目录；
   - `../../` 表示上上级目录；
   - 也可使用绝对路径，例如 `/home/user/Market/pytorch`。
4. 教程中的伪代码仅用于说明逻辑，不可直接执行。请勿按照教程修改代码。
5. `demo.py` 依赖提取好的特征文件，该文件保存在 `pytorch_feature.mat` 中。请先运行 `test.py` 提取特征，再运行 `demo.py`。
6. 如需使用Duke数据集，请使用新增的 `prepare-Duke.py`，或自行修改 `prepare.py`。
