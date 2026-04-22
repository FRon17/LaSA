# LaSA: Latent Statistical Alignment for Controllable Automotive Image Generation

<div align="center">
  <!-- 这里放一张你最满意的 2x2 生成结果图作为封面 -->
  <img width="626" height="626" alt="图片 17" src="https://github.com/user-attachments/assets/9e1a3753-403d-4bc8-8535-d45354a08cfb" />
</div>

<p align="center">
  <a href="https://github.com/FRon17/LaSA/stargazers"><img src="https://img.shields.io/github/stars/FRon17/LaSA.svg?style=flat-square" alt="Stars"></a>
  <a href="https://github.com/FRon17/LaSA/network/members"><img src="https://img.shields.io/github/forks/FRon17/LaSA.svg?style=flat-square" alt="Forks"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

> **中国传媒大学 广告学（计算广告） 本科毕业论文开源项目**
> 
> **中文标题：** 汽车图文广告的自动化标签体系与可控生成方法研究——以特斯拉 Model Y 为例
> 
> **English Title:** Research on Automated Tagging Systems and Controllable Generation Methods for Automotive Image-Text Advertisements: A Case Study of Tesla Model Y

## 📖 简介 (Introduction)

本项目针对汽车广告视觉物料生产中“车辆 ID 易漂移”与“多图宏观风格易割裂”的痛点，提出了一种**“数据洞察驱动 + 底层算法约束”**的全新 AIGC 内容生产范式。

基于当前最强的开源扩散模型 **FLUX.1-dev**，本项目创新性地提出了 **LaSA (Latent Statistical Alignment, 潜空间统计量对齐)** 模块，结合 In-Context LoRA (IC-LoRA) 机制，实现了在保持极高车辆 ID 准确率的前提下，生成具有高度宏观风格一致性的 2x2 网格成套广告物料。

## ✨ 核心特性 (Key Features)

- **🚗 自动化多模态数据工程：** 基于小红书真实社媒数据，构建了“检索-标签-营销”三层映射框架，提炼了 8 大视觉标签体系。
- **🧩 In-Context LoRA 排版学习：** 通过 2x2 网格拼接与结构化联合描述 (Joint Caption)，激活了 DiT 架构的全局上下文能力，实现智能排版。
- **⚖️ Dynamic LaSA 潜空间对齐：** 创新性提出 Leader-Follower 统计量对齐机制，直接在 VAE 潜空间约束宏观色彩与光影。
- **⏳ 动态时间步加权优化：** 设计了基于扩散时间步的自适应衰减权重策略，完美化解了车辆几何结构学习与风格对齐之间的梯度冲突。

## 🏗️ 算法架构 (Architecture)

### 1. 综合框架

  <!-- 替换为你的图1.1 -->
  <img width="810" height="706" alt="截屏2026-04-22 16 35 13" src="https://github.com/user-attachments/assets/b8a1f0c2-f1dd-4e29-82c1-d67cc7f14762" />
</div>

### 2. LaSA 训练流程
<div align="center">
  <!-- 替换为你的图4.1 -->
  <img width="380" height="596" alt="截屏2026-04-22 16 36 33" src="https://github.com/user-attachments/assets/516057a9-b909-4f0c-8b63-b9487b2ed73f" />
</div>

## 🖼️ 生成效果展示 (Results)

相比于原生的 FLUX 模型，本项目在色彩一致性上提升了 **55.6%**，彻底解决了多图生成中的宏观风格割裂问题。

### Baseline vs Ours

| 原生 FLUX.1-dev (Baseline) | LaSA + IC-LoRA (Ours) |
| :---: | :---: |
| <img width="450" alt="Baseline 2x2 Grid" src="https://github.com/user-attachments/assets/e50211de-66f1-46df-a417-a300cce3ad6c" /> | <img width="450" alt="Ours 2x2 Grid" src="https://github.com/user-attachments/assets/25e48659-d760-43ee-af42-9665c57b77ed" /> |
| **失败案例**：车辆 ID 混淆（前脸/车型不一），四个子图的色彩、光影与背景逻辑完全割裂。 | **成功案例**：车辆 ID 精准锁定，四个视角的宏观光影、色调高度统一，具备成套商业海报质感。 |

### 更多场景展示 (More Scenarios)
<img width="363" height="174" alt="截屏2026-04-22 16 38 57" src="https://github.com/user-attachments/assets/5cbfc387-542d-45a7-baa0-48adc7dd6800" />
<img width="366" height="174" alt="截屏2026-04-22 16 38 44" src="https://github.com/user-attachments/assets/f38a3a42-e922-46c4-98ff-c0f258c9dc11" />
<img width="361" height="173" alt="截屏2026-04-22 16 39 17" src="https://github.com/user-attachments/assets/6654ab1c-2fba-4a18-8d7b-02d8be8ae73c" />

## 📝数据准备 (Data Preparation)
本项目使用的数据集构建流程包含 YOLOv8 裁切、CLIP 相似度匹配与 Qwen-VL 联合打标。详见'DataEngineering'文件夹。

## 🥊模型训练与推理(Training & Inference)
训练与推理相关配置、代码详见AlgorithmLaSA文件夹。

## 📊评价指标 (Evaluation)
我们构建了包含局部/全局 CLIP Score、风格相似度、色彩直方图相似度的多维评价体系。详细的消融实验和定量分析请参考论文第五章。

## 🎁附赠手稿一张
<img width="1029" height="580" alt="截屏2026-04-22 16 23 00" src="https://github.com/user-attachments/assets/e2a6cbf9-1a26-498d-9f3c-cb549bafca2a" />

## 🫶🏻 引用 (Citation)
如果您觉得本项目对您的研究或工作有帮助，欢迎引用：

Bibtex
@thesis{fang2026lasa,
  title={汽车图文广告的自动化标签体系与可控生成方法研究——以特斯拉 Model Y 为例},
  author={方融},
  school={中国传媒大学},
  year={2026},
  type={本科毕业论文}
}

## 🤝 致谢 (Acknowledgements)
感谢我的导师王薇老师、左然老师的悉心指导。感谢中国传媒大学**广告与品牌学院**和**数据科学与智能媒体学院**的培养。

✉️ 联系方式 (Contact)
如有任何问题，欢迎提交 Issue 或通过邮件联系：fangrong17@163.com
