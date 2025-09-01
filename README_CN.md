# ComfyUI USO 节点

一个用于ComfyUI的自定义节点，集成USO（统一风格和主题驱动生成）模型，实现高质量的风格和主题控制图像生成。

## ✨ 特性

- 🎨 **统一风格与主题生成**: 基于FLUX架构的USO模型
- 🎯 **风格驱动生成**: 根据特定艺术风格生成图像
- 👤 **主题驱动生成**: 在生成过程中保持主题一致性
- 🔄 **多风格支持**: 在单次生成中结合多种风格
- ⚙️ **内存优化**: 支持FP8精度，适用于消费级GPU（约16GB显存）
- 🚀 **灵活控制**: 高级参数控制，精细调节生成结果

## 🔧 节点列表

### 核心节点
- **RH_USO_Loader**: 加载和初始化USO模型，包含优化选项
- **RH_USO_Generator**: 具有风格和主题控制的图像生成器

## 🚀 快速安装

### 步骤1: 安装节点
```bash
# 进入ComfyUI自定义节点目录
cd ComfyUI/custom_nodes

# 克隆仓库
git clone https://github.com/HM-RunningHub/ComfyUI_RH_USO

# 安装依赖
cd ComfyUI_RH_USO
pip install -r requirements.txt
```

### 步骤2: 下载所需模型
```bash
# 下载FLUX.1-dev模型（必需的基础模型）
huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir models/diffusers/FLUX.1-dev
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir models/diffusers/FLUX.1-dev

# 下载USO模型
huggingface-cli download bytedance-research/USO --local-dir models/uso

# 下载SigLIP模型
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir models/clip/siglip-so400m-patch14-384

# 最终模型结构应该如下：
models/
├── diffusers/
│   └── FLUX.1-dev/
│       ├── flux1-dev.safetensors
│       └── ae.safetensors
├── uso/
│   ├── assets/
│   │   └── uso.webp
│   ├── config.json
│   ├── download_repo_enhanced.py
│   ├── README.md
│   └── uso_flux_v1.0/
│       ├── dit_lora.safetensors
│       └── projector.safetensors
└── clip/
    └── siglip-so400m-patch14-384/
    
# 重启ComfyUI
```

## 📖 使用方法

### 基础工作流
```
[RH_USO_Loader] → [RH_USO_Generator] → [Save Image]
```

### 生成类型

#### 风格驱动生成
- 加载风格参考图像
- 输入描述内容的文本提示
- 生成指定风格的图像

#### 主题驱动生成
- 加载主题参考图像
- 输入包含场景描述的文本提示
- 生成保持主题身份的图像

#### 风格+主题生成
- 同时加载风格和主题参考图像
- 结合风格转换与主题一致性
- 生成具有统一风格且保持主题的图像

## 🛠️ 技术要求

- **GPU**: 16GB+显存（使用FP8优化）
- **内存**: 推荐32GB+
- **存储**: 约35GB用于所有模型
  - FLUX.1-dev: ~24GB (flux1-dev.safetensors + ae.safetensors)
  - USO模型: ~6GB
  - SigLIP: ~1.5GB
- **CUDA**: 优化性能需要CUDA支持

## ⚠️ 重要提示

- **模型路径**: 模型必须放置在特定目录：
  - FLUX.1-dev → `models/diffusers/FLUX.1-dev/`
  - USO模型 → `models/uso/`
  - SigLIP → `models/clip/siglip-so400m-patch14-384/`
- 推荐消费级GPU使用FP8模式（减少显存占用）
- 所有模型文件必须在首次使用前下载完成

## 📄 许可证

本项目采用Apache 2.0许可证。

## 🔗 参考链接

- [USO项目页面](https://bytedance.github.io/USO/)
- [USO论文](https://arxiv.org/abs/2508.18966)
- [USO HuggingFace](https://huggingface.co/bytedance-research/USO)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 🤝 贡献

欢迎贡献！请随时提交问题和拉取请求。

## ⭐ 引用

如果您觉得这个项目有用，请考虑引用原始USO论文：

```bibtex
@article{wu2025uso,
    title={USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning},
    author={Shaojin Wu and Mengqi Huang and Yufeng Cheng and Wenxu Wu and Jiahe Tian and Yiming Luo and Fei Ding and Qian He},
    year={2025},
    eprint={2508.18966},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```
