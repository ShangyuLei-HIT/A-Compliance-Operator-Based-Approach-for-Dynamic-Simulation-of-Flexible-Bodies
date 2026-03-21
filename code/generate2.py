"""
Author: 雷尚谕
"""
import torch
# import torch.nn as nn
import joblib
import numpy as np
from MLP import MLP
from sklearn.metrics import mean_squared_error
import os
import json
import time
import tool
from scipy.io import loadmat

def gen(outputdim, folder_name, model_name, use_standerscale):
    print("use gen")
    node_mat = loadmat("dataset/node.mat")
    S_mat = loadmat("dataset/S_fromV.mat")
    nodeCoor = node_mat['all_nodeCoor']
    S = S_mat['S_fromV']

    nodelist1=[]
    nodelist2=[]
    slist_true=[]
    i=18
    # i=0
    print("view node",nodeCoor[i])

    for j in range(len(nodeCoor)):
    # for j in range(1):
        nodepair1=np.concatenate([
                nodeCoor[i],nodeCoor[j],
                (nodeCoor[i]-nodeCoor[j])
            ])
        nodepair2=np.concatenate([
                nodeCoor[j],nodeCoor[i],
                (nodeCoor[j]-nodeCoor[i])
            ])
        nodelist1.append(nodepair1)
        nodelist2.append(nodepair2)

        spair = np.array([
                S[3 * i][3 * j],  # s11
                S[3 * i][3 * j + 1],  # s12
                S[3 * i][3 * j + 2],  # s13

                S[3 * i + 1][3 * j],  # s21
                S[3 * i + 1][3 * j + 1],  # s22
                S[3 * i + 1][3 * j + 2],  # s23

                S[3 * i + 2][3 * j],  # s31
                S[3 * i + 2][3 * j + 1],  # s32
                S[3 * i + 2][3 * j + 2],  # s33
            ])
        if j==37:
            print("nodepair",nodepair1,"spair",spair)
        slist_true.append(spair)

    nodelist1=np.array(nodelist1)
    nodelist2=np.array(nodelist2)
    slist_true=np.array(slist_true)
    # slist_true=slist_true[:,outputdim].reshape(-1,1)

    # print(f'in gen nodelist1_len{len(nodelist1)}')

    print("nodelist1_截取", nodelist1[1:4, :])
    S_pre1 = generate(nodepair=nodelist1, folder_name=folder_name,
                      model_name=model_name, use_standerscale=use_standerscale, outputdim=outputdim)
    print("spre1_截取", S_pre1[1:4, :])
    slist_true = slist_true[:, outputdim].reshape(-1, 1)
    S_error = 100 * np.abs(S_pre1 - slist_true) / np.abs(slist_true)

    print(f'node list1{nodelist1.shape}')
    print(f'S true{slist_true}')
    print(f'S pre1{S_pre1}')

    for id in range(1):
        tool.plot4D(x=nodelist1[:, 3], y=nodelist1[:, 4], z=nodelist1[:, 5], s=S_pre1[:, id],
                    plot_name=f'S_pre1_{id}', folder_name=folder_name)
        tool.plot4D(x=nodelist1[:, 3], y=nodelist1[:, 4], z=nodelist1[:, 5], s=slist_true[:, id],
                    plot_name=f'S_true_{id}', folder_name=folder_name)
        tool.plot4D_error(x=nodelist1[:, 3], y=nodelist1[:, 4], z=nodelist1[:, 5], s=S_error[:, id],
                          plot_name=f'S_error_{id}', folder_name=folder_name)


def generate(nodepair,folder_name,model_name,use_standerscale,outputdim):
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
    batchsize = params["batchsize"]

    model_path = f'{folder_name}/{model_name}'
    model=MLP(use_dropout=use_dropout,dropout_p=dropout_p,
              hidden_dim=hidden_dim,block_num=block_num).to(
                  device=device,dtype=dtype)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    param_json = os.path.join("dataset/scale.json")
    with open(param_json, "r") as f:
        params = json.load(f)
    scale_node = params["scale_node"]
    scale_S = params["scale_S"]

    print(f'scale_node gen {scale_node}')


    nodepair=scale_node*nodepair.reshape(-1,9)

    # print(f'after scal gen {nodepair}')

    if use_standerscale:
        scaler_train_inputs = joblib.load("dataset/scaler_inputs.pkl")
        params = {
            "mean": scaler_train_inputs.mean_.tolist(),
            "scale": scaler_train_inputs.scale_.tolist()
        }

        with open("dataset/scaler_inputs.json", "w") as f:
            json.dump(params, f)

        nodepair=scaler_train_inputs.transform(nodepair)
    # print("nodepair af sc",nodepair)
    nodepair=torch.tensor(nodepair,dtype=dtype,device=device)

    print(f'gen input {nodepair}')
    with torch.no_grad():
        # stime=time.time()
        outputs=model(nodepair)


    # print("模型输出",outputs)
    labels_pred=outputs.cpu().numpy()

    etime = time.time()
    print(f'向前传播用时{etime - stime}')
    print(f'scaleS{scale_S[outputdim]}')
    return labels_pred/scale_S[outputdim]

