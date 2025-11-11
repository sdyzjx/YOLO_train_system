from ultralytics import YOLO
import os
import shutil

# --- 1. 初始化与路径设置 ---

# --- 核心配置 ---
# 只需要修改这里，下面的变量会自动生成
model_path = 'runs/obb/train99/weights/best.pt'
img_path = '216.jpg'
# ---------------------

# --- 自动从模型路径中提取训练名称 (例如 "train7") ---
try:
    # os.path.dirname() 会逐级返回上层目录
    # 'runs/obb/train7/weights/best.pt' -> 'runs/obb/train7/weights' -> 'runs/obb/train7'
    train_dir = os.path.dirname(os.path.dirname(model_path))
    # os.path.basename() 获取最后一级目录名
    # 'runs/obb/train7' -> 'train7'
    train_run_name = os.path.basename(train_dir)
except IndexError:
    print(f"错误：无法从模型路径 '{model_path}' 中解析训练名称。请确保路径格式为 '.../trainX/weights/best.pt'")
    exit()

# --- 自动设置输出目录 ---
output_project = os.path.dirname(train_dir) # 获取 'runs/obb' 作为项目根目录
output_name = f"{train_run_name}_visualized" # 动态命名为 'train7_visualized'
# ---------------------

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"错误：模型文件未找到，请检查路径: {model_path}")
    exit()
if not os.path.exists(img_path):
    print(f"错误：图片文件未找到，请检查路径: {img_path}")
    exit()

# 为了避免结果混淆，可以选择在运行前删除旧的输出文件夹
full_output_path = os.path.join(output_project, output_name)
if os.path.exists(full_output_path):
    print(f"警告：检测到旧的输出目录 '{full_output_path}'，将进行删除...")
    shutil.rmtree(full_output_path)

# 加载你的预训练模型
model = YOLO(model_path)
print("-" * 60)
print(f"成功加载模型: {model_path}")
print(f"待处理图像: {img_path}")
print(f"结果将保存至: {os.path.abspath(full_output_path)}")
print("-" * 60)

# --- 2. 执行推理并自动可视化 ---
print("正在执行模型推理并生成内置的特征图可视化...")

# 使用 model.predict() 并设置关键参数
results = model.predict(
    source=img_path,
    visualize=True,      # 激活内置的特征图可视化功能
    imgsz=512,           # 将输入图像统一调整到 512x512
    project=output_project, # 指定保存结果的根目录 (e.g., 'runs/obb')
    name=output_name        # 指定本次运行的子目录名 (e.g., 'train7_visualized')
)

# --- 3. 告知用户结果位置 ---
if results:
    save_directory = results[0].save_dir
    print("-" * 60)
    print("✅ 可视化成功！")
    print(f"所有结果（包括预测图和特征热力图）已自动保存至以下目录:")
    print(f"   {os.path.abspath(save_directory)}")
    print("-" * 60)
else:
    print("未能成功执行预测。")