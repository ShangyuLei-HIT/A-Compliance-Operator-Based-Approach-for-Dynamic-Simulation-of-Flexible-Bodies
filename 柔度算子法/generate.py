"""
Author: 雷尚谕
"""
import torch
import joblib
import numpy as np
from MLP import MLP
from sklearn.metrics import mean_squared_error
import os
import json
import time
import tool

def gen(outputdim, folder_name, model_name,
        use_standerscale, Stype, basePath):
    print(f'------gen-------')
    train_inputs = np.load(f"{basePath}/train/train_inputs.npy")
    train_labels = np.load(f"{basePath}/{Stype}/train/train_labels{outputdim}.npy")

    param_json = os.path.join(f"{basePath}/{Stype}/scale_{Stype}.json")
    with open(param_json, "r") as f:
        params = json.load(f)
    node_count1 = params["node_count"]
    
    idx_i = 124
    # node_count1 = 1569
    if Stype == "Dis":
        num = 9
    else:
        num = 18
    
    i_count1 = train_inputs.shape[0] / node_count1
    i_length1 = (node_count1) * num
    i_length1_dof = i_length1 / 3
    slist_true = np.zeros((int(node_count1), num))
    nodelist1 = np.zeros((int(node_count1), 9))

    print(f'icount {i_count1}')
    for idx_j in range(node_count1):
        if idx_j==86:
            print(f"i{idx_i} j {idx_j}")

        nodelist1[idx_j] = train_inputs[idx_i * node_count1 + idx_j, :]
        slist_true[idx_j] = train_labels[idx_i * node_count1 + idx_j, :]

    # print("nodelist1_截取",nodelist1[1:4,:])
    print("nodelist1_截取",nodelist1[1096,:])
    print(f"nodelist{nodelist1.shape}")
    S_pre1=generate(nodepair=nodelist1,folder_name=folder_name,
                   model_name=model_name,use_standerscale=use_standerscale,
                   outputdim=outputdim,Stype=Stype,basePath=basePath)
    # print("spre1_截取", S_pre1[1:4, :])
    print("spre1_截取", S_pre1[1096, :])
    slist_true = slist_true[:,outputdim].reshape(-1,1)
    S_error=100*np.abs(S_pre1-slist_true)/(np.abs(slist_true) + 1e-6)

    print(f'node list1{nodelist1.shape}')
    print(f'S true{slist_true[1096,:]}')
    print(f'S pre1{S_pre1[1096,:]}')

    for id in range(1):
        tool.plot4D(x=nodelist1[:,3],y=nodelist1[:,4],z=nodelist1[:,5],s=S_pre1[:,id],
            plot_name=f'S_pre1_{id}',folder_name=folder_name)
        tool.plot4D(x=nodelist1[:,3],y=nodelist1[:,4],z=nodelist1[:,5],s=slist_true[:,id],
            plot_name=f'S_true_{id}',folder_name=folder_name)
        tool.plot4D_error(x=nodelist1[:,3],y=nodelist1[:,4],z=nodelist1[:,5],s=S_error[:,id],
            plot_name=f'S_error_{id}',folder_name=folder_name)

        

def generate(nodepair,folder_name,model_name,use_standerscale,
             outputdim,Stype,basePath):
    stime = time.time()
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    device = torch.device("cpu")
    dtype=torch.float32

    param_json = os.path.join(folder_name, "params.json")
    with open(param_json, "r") as f:
        params = json.load(f)
    use_dropout = params["use_dropout"]
    dropout_p = params["dropout_p"]
    hidden_dim=params["hidden_dim"]
    block_num=params["block_num"]

    model_path = f'{folder_name}/{model_name}'
    model=MLP(use_dropout=use_dropout,dropout_p=dropout_p,
              hidden_dim=hidden_dim,block_num=block_num).to(
                  device=device,dtype=dtype)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    param_json = os.path.join(f"{basePath}/{Stype}/scale_{Stype}.json")
    with open(param_json, "r") as f:
        params = json.load(f)
    scale_node = params["scale_node"]
    scale_S = params["scale_S"]
    print(f'scale_node gen {scale_node}')


    nodepair=scale_node*nodepair.reshape(-1,9)

    # print(f'after scal gen {nodepair}')

    if use_standerscale:
        scaler_train_inputs = joblib.load(f"{basePath}/scaler_inputs.pkl")
        scaler_params = {
            "mean": scaler_train_inputs.mean_.tolist(),
            "scale": scaler_train_inputs.scale_.tolist()
        }

        os.makedirs(f"inference/{basePath}", exist_ok=True)
        param_json = os.path.join(f"inference/{basePath}", "scale.json")

        # 若已存在则读取
        if os.path.exists(param_json):
            with open(param_json, "r") as f:
                params = json.load(f)
        else:
            params = {}

        # 续写 / 合并
        params.update(scaler_params)

        # 写回 scale.json
        with open(param_json, "w") as f:
            json.dump(params, f, indent=4)

        # with open(f"inference/{basePath}/scaler_inputs.json", "w") as f:
        #     json.dump(params, f)

        nodepair=scaler_train_inputs.transform(nodepair)
    # print("nodepair af sc",nodepair)
    nodepair=torch.tensor(nodepair,dtype=dtype,device=device)

    print(f'gen input {nodepair}')
    with torch.no_grad():
        # stime=time.time()
        outputs=model(nodepair)


    # print("模型输出",outputs)
    labels_pred=outputs.cpu().numpy()
    labels_pred=tool.signed_pow10(labels_pred)
    labels_pred=labels_pred/scale_S[outputdim]
    # labels_pred=tool.signed_pow10(labels_pred)

    etime = time.time()
    print(f'向前传播用时{etime - stime}')
    print(f'scaleS{scale_S[outputdim]}')
    return labels_pred

