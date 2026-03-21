"""
Author: 雷尚谕
"""
import torch
import json
from MLP import MLP  # 确保 MLP.py 在同目录下或 PYTHONPATH 可见
import onnx
import os

def export_onnx(use_dropout, dropout_p,
                hidden_dim, block_num,
                folder_name, outputdim, Stype, name_psd):

    # === 0. 输出文件名 ===
    out_name = f"{Stype}_{outputdim}.onnx"
    model_name = "mlp_model_best.pth"

    # === 1. 读取超参数 ===
    param_path = f"{folder_name}/params.json"
    print(f"foldername: {folder_name}")
    with open(param_path, "r") as f:
        params = json.load(f)

    # === 2. 创建模型并加载权重 ===
    model = MLP(
        use_dropout=use_dropout,
        dropout_p=dropout_p,
        hidden_dim=hidden_dim,
        block_num=block_num
    )

    state_dict = torch.load(f"{folder_name}/{model_name}", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # === 3. 准备示例输入（必须维度一致）===
    dummy_input = torch.randn(1, 9)

    # === 4. 导出到 ONNX ===
    os.makedirs(f"inference/models/{name_psd}", exist_ok=True)
    onnx_path = f"inference/models/{name_psd}/{out_name}"
    # onnx_path = f"{folder_name}/{out_name}"

    print("开始导出 ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,                   # 推荐版本
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={                      # 保持 batch 维度可变
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )

    print(f"已导出 ONNX 模型到：{onnx_path}")

    # === 5. 检查 ONNX 模型 ===
    print("检查 ONNX 模型结构...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型检查通过")

    return onnx_path
