"""
Author: 雷尚谕
"""
import numpy as np
import h5py
import os
import json
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import tool
from psdLoad import read_dataset_in_chunks
import time

def gen_labels(name_psd, S, Stype, num, test_indices):

    test_labels = S[test_indices]
    print(f"test inputs{test_inputs.shape}")

    train_labels = S

    scale_S = 1e3 / (np.max(np.abs(train_labels), axis=0))
    # scale_S = scale_S[:, np.newaxis]
    print(f"scaleS {scale_S} shape {scale_S.shape}")

    # train_labels = tool.signed_log10(train_labels)
    # test_labels = tool.signed_log10(test_labels)

    train_labels_scaled = train_labels * scale_S
    test_labels_scaled = test_labels * scale_S

    train_labels_scaled = tool.signed_log10(train_labels_scaled)
    test_labels_scaled = tool.signed_log10(test_labels_scaled)

    # epsilon = 1e-3
    # train_labels_scaled = train_labels_scaled + np.sign(train_labels_scaled) * epsilon
    # test_labels_scaled = test_labels_scaled + np.sign(test_labels_scaled) * epsilon

    print(f"label {train_labels[0:5,:]}")
    print(f"label scaled {train_labels_scaled[0:5,:]}")
    print(f"test {test_labels[0:5,:]}")
    print(f"test scaled {test_labels_scaled[0:5,:]}")

    # Stype = "Stress"
    labelPath = f"dataset/{name_psd}/{Stype}"
    os.makedirs(f"{labelPath}", exist_ok=True)
    param_json = os.path.join(f"{labelPath}/scale_{Stype}.json")
    params = {"node_count":node_count1,"scale_node": scale_node, "scale_S": scale_S.tolist()}
    with open(param_json, "w") as f:
        json.dump(params, f, indent=4)

    os.makedirs(f"{labelPath}/train", exist_ok=True)
    os.makedirs(f"{labelPath}/test", exist_ok=True)

    # np.save(f"{labelPath}/train/train_labels.npy", train_labels)
    # np.save(f"{labelPath}/test/test_labels.npy", test_labels)
    # np.save(f"{labelPath}/train/train_labels_scaled.npy", train_labels_scaled)
    # np.save(f"{labelPath}/test/test_labels_scaled.npy", test_labels_scaled)

    for dim in range(num):
        np.save(f"{labelPath}/train/train_labels{dim}.npy", train_labels[:, dim].reshape(-1, 1))
        print(train_labels[:, dim].reshape(-1, 1).shape)
        np.save(f"{labelPath}/test/test_labels{dim}.np", test_labels[:, dim].reshape(-1, 1))

        np.save(
            f"{labelPath}/train/train_labels_scaled{dim}.npy",
            train_labels_scaled[:, dim].reshape(-1, 1),
        )
        np.save(
            f"{labelPath}/test/test_labels_scaled{dim}.npy",
            test_labels_scaled[:, dim].reshape(-1, 1),
        )
    
    return scale_S

parser = argparse.ArgumentParser()
parser.add_argument("--Strain", action="store_true",
                    help="make strain data")
args = parser.parse_args()
make_strain = args.Strain

start = time.time()
print(f"make strain{make_strain}")
#name_psd = "gear19_2gearContact_geneData_Sp"
# name_psd = "D_ModalBodySp_gearHelical_Modal2"
name_psd = "kaocaoSingleMspNew"
basePath = f"dataset/{name_psd}"
h5_path = f"dataset/{name_psd}.psd"
# h5_path = f"../D_ModalBodySp_gearHelical_Modal.psd"
with h5py.File(f"{h5_path}", "r") as f:

    ModalId = 2
    ModalName = f"MultiBody/ModalData/ModalData_{ModalId}"
    node_count1 = f[f"{ModalName}/Node/Count"][:].item()
    # node_count2 = f["MultiBody/ModalData/ModalData_2/Node/Count"][:].item()

    # === 分块读取 ===
    node_Coordinate1 = read_dataset_in_chunks(
        f[f"{ModalName}/Node/Coordinate"]
    )
    # node_Coordinate2 = read_dataset_in_chunks(
    #     f["MultiBody/ModalData/ModalData_2/Node/Coordinate"]
    # )

    NodeDofStartId1 = read_dataset_in_chunks(
        f[f"{ModalName}/Mode/NodeDofStartId"]
    )

    AttachMode1 = read_dataset_in_chunks(
        f[f"{ModalName}/Mode/AttachMode"]
    )
    # AttachMode2 = read_dataset_in_chunks(
    #     f["MultiBody/ModalData/ModalData_2/Mode/AttachMode"]
    # )

    AttachModeDofId1 = read_dataset_in_chunks(
        f[f"{ModalName}/Mode/AttachModeDofId"]
    )
    # AttachModeDofId2 = read_dataset_in_chunks(
    #     f["MultiBody/ModalData/ModalData_2/Mode/AttachModeDofId"]
    # )

    if make_strain == True:
        AttachStrainMode1 = read_dataset_in_chunks(
            f[f"{ModalName}/Mode/AttachStrainMode"]
        )

    AttachStressMode1 = read_dataset_in_chunks(
        f[f"{ModalName}/Mode/AttachStressMode"]
    )

    # print(node_Coordinate1, node_Coordinate1.shape, node_Coordinate1.dtype)

# === 1. 准备拆分坐标 ===
node_Coordinate1_x = node_Coordinate1[0::3, :]
node_Coordinate1_y = node_Coordinate1[1::3, :]
node_Coordinate1_z = node_Coordinate1[2::3, :]

# print(node_Coordinate1_x, node_Coordinate1_x.shape, node_Coordinate1_x.dtype)

# === 2. 计算参数 ===
i_count1 = int(AttachMode1.shape[0] / (node_count1) / 9)
# i_count2 = int(AttachMode2.shape[0] / (node_count2) / 9)

print(i_count1)
# print(i_count2)

i_length1 = int(AttachMode1.shape[0] / i_count1)
i_length1_dof = int(AttachMode1.shape[0] / i_count1 / 3)

i_length1_s = int(AttachStressMode1.shape[0] / i_count1)
i_length1_dof_s = int(AttachStressMode1.shape[0] / i_count1 / 3)

# === 3. 分配结果矩阵 ===
total = int(i_count1 * node_count1)

S = np.zeros((total, 9))
Stress = np.zeros((total, 18))
Strain = np.zeros((total, 18))
nodePairs = np.zeros((total, 9))
S_node_idx = np.zeros((total, 2))

print(AttachModeDofId1[1].shape)

# === 4. 预计算 "每个 idx_i" 对应的 node_idx_i（向量化）===
# 原逻辑: node_idx_i = np.where(NodeDofStartId1 == AttachModeDofId1[idx_i*3])[0][0]
attach_values = AttachModeDofId1.reshape(-1)[0::3]   # shape = (i_count1,1)
node_idx_i_all = np.searchsorted(NodeDofStartId1.reshape(-1), attach_values)

# === 5. 生成 idx_i 与 idx_j 的二维网格 ===
idx_i_grid, idx_j_grid = np.meshgrid(
    np.arange(i_count1), 
    np.arange(node_count1), 
    indexing='ij'
)

idx_i_flat = idx_i_grid.reshape(-1)
idx_j_flat = idx_j_grid.reshape(-1)

# === 6. 生成 node_idx_i_flat ===
node_idx_i_flat = node_idx_i_all[idx_i_flat]

# === 7. 构造 nodePairs（完全向量化） ===
nodePairs[:, 0] = node_Coordinate1_x[node_idx_i_flat].reshape(-1)
nodePairs[:, 1] = node_Coordinate1_y[node_idx_i_flat].reshape(-1)
nodePairs[:, 2] = node_Coordinate1_z[node_idx_i_flat].reshape(-1)

nodePairs[:, 3] = node_Coordinate1_x[idx_j_flat].reshape(-1)
nodePairs[:, 4] = node_Coordinate1_y[idx_j_flat].reshape(-1)
nodePairs[:, 5] = node_Coordinate1_z[idx_j_flat].reshape(-1)

nodePairs[:, 6] = nodePairs[:, 0] - nodePairs[:, 3]
nodePairs[:, 7] = nodePairs[:, 1] - nodePairs[:, 4]
nodePairs[:, 8] = nodePairs[:, 2] - nodePairs[:, 5]

# === 8. 写入 S_node_idx（向量化）===
S_node_idx[:, 0] = node_idx_i_flat
S_node_idx[:, 1] = node_count1


# === 9. 处理 DOF（只剩下 dof_i/dof_j 2×2 的小循环）===
for dof_i in range(3):
    for dof_j in range(3):

        # === 主模式 S 部分 ===
        idx = (
            idx_i_flat * i_length1
            + dof_i * i_length1_dof
            + idx_j_flat * 3
            + dof_j
        )
        S[:, dof_i * 3 + dof_j] = AttachMode1[idx].reshape(-1)

        # === Stress 部分 ===
        idx_s = (
            idx_i_flat * i_length1_s
            + dof_i * i_length1_dof_s
            + idx_j_flat * 6
            + dof_j * 2
        )
        Stress[:, dof_i * 6 + dof_j * 2] = AttachStressMode1[idx_s].reshape(-1)
        Stress[:, dof_i * 6 + dof_j * 2 + 1] = AttachStressMode1[idx_s + 1].reshape(-1)

        # === Strain 部分（可选）===
        if make_strain:
            Strain[:, dof_i * 6 + dof_j * 2] = AttachStrainMode1[idx_s].reshape(-1)
            Strain[:, dof_i * 6 + dof_j * 2 + 1] = AttachStrainMode1[idx_s + 1].reshape(-1)

# print(S.shape, S[0:5,:])
# print(Stress.shape, Stress[0:5,:])
# print(Strain.shape, Strain)
# print(S_node_idx.shape, S_node_idx)
# print(nodePairs.shape, nodePairs)

test_ration = 1.0
total_size = nodePairs.shape[0]
print(f"total size {total_size}")
test_size = int(test_ration * total_size)
print(f"test size {test_size}")

np.random.seed(42)
indices = np.random.permutation(total_size)
test_indices = indices[:test_size]

test_inputs = nodePairs[test_indices]
train_inputs = nodePairs

scale_node = 1e1 / np.max(np.abs(train_inputs))

train_inputs_scaled = train_inputs * scale_node
test_inputs_scaled = test_inputs * scale_node

os.makedirs(f"{basePath}", exist_ok=True)
use_standerscale = True
if use_standerscale:
    scaler_train_inputs = StandardScaler()
    train_inputs_scaled = scaler_train_inputs.fit_transform(train_inputs_scaled)
    test_inputs_scaled = scaler_train_inputs.transform(test_inputs_scaled)

    joblib.dump(scaler_train_inputs, f"{basePath}/scaler_inputs.pkl")

os.makedirs(f"{basePath}/train", exist_ok=True)
os.makedirs(f"{basePath}/test", exist_ok=True)

np.save(f"{basePath}/train/train_inputs.npy", train_inputs)
np.save(f"{basePath}/test/test_inputs.npy", test_inputs)

np.save(f"{basePath}/train/train_inputs_scaled.npy", train_inputs_scaled)
np.save(f"{basePath}/test/test_inputs_scaled.npy", test_inputs_scaled)

scale_Dis = gen_labels(name_psd, S, "Dis", 9, test_indices)
scale_Stress = gen_labels(name_psd, Stress, "Stress", 18, test_indices)

scale_Strain = scale_Stress * 0
if make_strain:
    scale_Strain = gen_labels(name_psd, Strain, "Strain", 18, test_indices)

os.makedirs(f"inference/{basePath}", exist_ok=True)
param_json = os.path.join(f"inference/{basePath}/scale.json")
params = {"node_count":node_count1,
          "scale_node": scale_node,
          "scale_Dis": scale_Dis.tolist(),
          "scale_Stress": scale_Stress.tolist(),
          "scale_Strain": scale_Strain.tolist()
}
end = time.time()
print(f"time {end-start}")
with open(param_json, "w") as f:
    json.dump(params, f, indent=4)
