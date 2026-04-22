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
<div align="center"><img width="376" height="594" alt="截屏2026-04-22 16 36 08" src="https://github.com/user-attachments/assets/7ca87b0a-8c0c-41f2-a47f-e5870e725d5b" />

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
| <!-- 替换为图5.4 --> <img width="171" height="170" alt="截屏2026-04-22 16 37 29" src="https://github.com/user-attachments/assets/e50211de-66f1-46df-a417-a300cce3ad6c" />
 | <!-- 替换为图5.9 --> <img width="170" height="169" alt="截屏2026-04-22 16 38 07" src="https://github.com/user-attachments/assets/25e48659-d760-43ee-af42-9665c57b77ed" />
 |
| 车辆 ID 混淆，四个子图色彩、光影完全割裂 | **车辆 ID 精准，光影色调高度统一，具备商业质感** |

### 更多场景展示 (More Scenarios)
<img width="363" height="174" alt="截屏2026-04-22 16 38 57" src="https://github.com/user-attachments/assets/5cbfc387-542d-45a7-baa0-48adc7dd6800" />
<img width="366" height="174" alt="截屏2026-04-22 16 38 44" src="https://github.com/user-attachments/assets/f38a3a42-e922-46c4-98ff-c0f258c9dc11" />
<img width="361" height="173" alt="截屏2026-04-22 16 39 17" src="https://github.com/user-attachments/assets/6654ab1c-2fba-4a18-8d7b-02d8be8ae73c" />


## 附赠手稿一张
<img width="1029" height="580" alt="截屏2026-04-22 16 23 00" src="https://github.com/user-attachments/assets/e2a6cbf9-1a26-498d-9f3c-cb549bafca2a" />
