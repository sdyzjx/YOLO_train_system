# YOLO Train Backend 管理系统

这个项目提供了一个基于 Gradio 的 YOLO 训练/验证管理系统，用于统一查看、发起或管理训练任务，并扩展了特征图可视化、文件浏览等实用功能。以下内容帮助你快速了解项目结构、依赖要求以及运行方式。

## 项目结构概览

```
yolo_train_backend/
├── app/                 # Gradio 前端与后端逻辑（入口：app/app.py）
├── modules/             # 可选的本地 third-party 覆盖或扩展
│   └── ultralytics/     # 如需自定义 ultralytics，可放在这里
├── datasets/            # 数据集资源（每个子目录对应一个数据集，如 datasets/ttpla）
├   └──cfg/              # 数据集 YAML 配置，系统会自动扫描此处的配置文件
├── runs/                # 训练/验证结果输出
├── train_method/        # 自定义训练脚本（如 distill_p3p4_train.py）
├── models_cfg/          # 模型或训练配置
├── PROCESS.md           # 流程相关说明
├── README.md            # 本文件
└── *.pt                 # 预置模型权重
```

## 安装依赖

1. **基础环境**  
   推荐使用 Conda 或其他虚拟环境工具创建隔离环境，并确保 Python 版本满足需求。

2. **Gradio**  
   前端基于 Gradio，需先安装：
   ```bash
   pip install gradio
   ```

3. **Ultralytics**  
   - 日常使用建议直接安装官方发布版：
     ```bash
     pip install ultralytics
     ```
   - 如果需要自定义 ultralytics，可将 fork/修改后的代码放在 `modules/ultralytics/` 下，然后在该目录执行：
     ```bash
     pip install -e .
     ```
     这样管理系统就会优先加载本地版本，实现自定义功能。

其他训练依赖（如 PyTorch 等）请根据自身 GPU/CPU 环境选择合适的安装命令。

## 运行方式

在项目根目录下运行：
```bash
python -m app.app
```
这将启动 Gradio 界面，默认在本机生成一个可访问的 Web UI。

> **注意**：在 YOLO 管理系统中启动训练任务时，实际运行环境即为当前运行该管理系统的 Conda 环境。因此在启动前要保证该环境已正确安装所有训练所需依赖和驱动。

## 其他说明

- `runs/` 目录下的 `train/`、`val/` 会保存训练与验证产物，可根据需要清理或备份。
- `datasets/` 下每个子目录存放一个完整数据集；对应的 YAML 配置文件需放在 `datasets/cfg/`，系统会自动扫描该目录以生成下拉列表。请确保 YAML 中引用的绝对/相对路径与实际 `datasets/` 目录结构一致，否则运行训练时会报找不到数据。
- 如需扩展或定制功能，可在 `app/backend/`、`train_method/` 等目录添加新的模块或脚本。
- `datasets/` 目录用于放置数据集。
如有问题或需要进一步扩展，欢迎结合项目代码阅读或在对应模块内添加文档注释。祝使用顺利！🚀
