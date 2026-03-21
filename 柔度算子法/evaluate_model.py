"""
Author: 雷尚谕
"""
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.externals.array_api_compat import device
from MLP import MLP
from sklearn.metrics import mean_squared_error
import os
import json
from train_model import loss_fn
import time

def error(y_pred, y_true, foldername, scale_S):
    print(f'y_pred shape{y_pred.shape}')
    ab_error = np.abs((y_true - y_pred))
    relative_errors = 100 * np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))
    # print("relative_errors",relative_errors)
    print(f'ab error shape{ab_error.shape}')
    mean_ab = np.mean(ab_error / scale_S, axis=0)
    max_ab = np.max(ab_error / scale_S, axis=0)
    mean_error = np.mean(relative_errors, axis=0)
    max_error = np.max(relative_errors,axis=0)
    print(f'max_ab shape{max_ab.shape}')
    ratio_1 = 100 * np.sum(relative_errors <= 5) / np.prod(relative_errors.shape)
    ratio_2 = 100 * np.sum(relative_errors <= 10) / np.prod(relative_errors.shape)
    ratio_5 = 100 * np.sum(relative_errors <= 20) / np.prod(relative_errors.shape)
    ratio_10 = 100 * np.sum(relative_errors <= 40) / np.prod(relative_errors.shape)
    ratio_50 = 100 * np.sum(relative_errors <= 60) / np.prod(relative_errors.shape)

    log = (
        f"Mean Absolute Error: {mean_ab}\n"
        f"Max Absolute Error: {max_ab}\n"
        f"Mean Relative Error (%): {mean_error}\n"
        f"Max Relative Error (%): {max_error}\n"
        f"Ratio < 5%: {ratio_1:.2f}%\n"
        f"Ratio < 10%: {ratio_2:.2f}%\n"
        f"Ratio < 20%: {ratio_5:.2f}%\n"
        f"Ratio < 40%: {ratio_10:.2f}%\n"
        f"Ratio < 60%: {ratio_50:.2f}%\n"
    )

    # 输出打印
    print(log)

    filename = f"{foldername}/error.txt"
    # 保存到文件
    with open(filename, "w") as f:
        f.write(log)
    # print("r1",ratio_1,"r2",ratio_2,"r5",ratio_5,"r10",ratio_10,"r50",ratio_50,"ab_mean",mean_ab,"max_ab",max_ab)
    return mean_error, max_error

def evaluate(folder_name, model_name, use_standerscale,
             outputdim, Stype, basePath):
    stime = time.time()
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16

    test_inputs_scaled = np.load(f"{basePath}/test/test_inputs_scaled.npy")
    test_labels_scaled = np.load(f"{basePath}/{Stype}/test/test_labels_scaled{outputdim}.npy")

    test_inputs_scaled = torch.tensor(test_inputs_scaled, dtype=dtype, device=device)
    test_labels_scaled = torch.tensor(test_labels_scaled, dtype=dtype, device=device)

    param_json = os.path.join(folder_name, "params.json")
    with open(param_json, "r") as f:
        params = json.load(f)
    use_dropout = params["use_dropout"]
    dropout_p = params["dropout_p"]
    hidden_dim = params["hidden_dim"]
    block_num = params["block_num"]
    batchsize = params["batchsize"]
    # batchsize = 100

    model_path = f"{folder_name}/{model_name}"
    model = MLP(
        use_dropout=use_dropout,
        dropout_p=dropout_p,
        hidden_dim=hidden_dim,
        block_num=block_num,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ✅ -------- 新增：分批推理部分 --------
    start_batch = time.time()

    with torch.no_grad():
        outputs_list = []
        num_samples = test_inputs_scaled.shape[0]

        for start in range(0, num_samples, batchsize):
            end = start + batchsize
            batch_inputs = test_inputs_scaled[start:end]
            batch_outputs = model(batch_inputs)
            outputs_list.append(batch_outputs)

        outputs = torch.cat(outputs_list, dim=0)
        loss = loss_fn(outputs, test_labels_scaled)

    end_batch = time.time()
    print(f"推理用时{end_batch-start_batch} 单次{(end_batch-start_batch)/(1400000/batchsize)}")
    # ✅ -------- 新增部分结束 --------

    labels_pred = outputs.cpu().numpy()
    labels_true = test_labels_scaled.cpu().numpy()

    labels_pred = np.asarray(labels_pred, dtype=np.float32)
    labels_true = np.asarray(labels_true, dtype=np.float32)

    # if use_standerscale:
    #     scaler_train_labels = joblib.load("{basePath}/scaler_labels.pkl")
    #     labels_pred = scaler_train_labels.inverse_transform(
    #         outputs.cpu().numpy())
    #     labels_true = scaler_train_labels.inverse_transform(
    #         test_labels_scaled.cpu().numpy())

    loss_true = mean_squared_error(labels_true, labels_pred)

    etime = time.time()
    print("S true", labels_true[:5])
    print("S pre ", labels_pred[:5])
    print("loss_true", loss_true, "time", etime - stime)

    param_json = os.path.join(f"{basePath}/{Stype}/scale_{Stype}.json")
    with open(param_json, "r") as f:
        params = json.load(f)
    scale_S = params["scale_S"]

    scale_S = np.asarray(scale_S, dtype=np.float32)

    mean_error, max_error = error(
        y_pred=labels_pred, y_true=labels_true, foldername=folder_name, scale_S=scale_S[outputdim]
    )
    print("mean_error", mean_error)
    print("max_error", max_error)
