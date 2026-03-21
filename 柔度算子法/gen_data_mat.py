"""
Author: 雷尚谕
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import json
from tool import scaled
from tool import Loadmat
from tool import signed_log10
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def copy_to_reversed(origin):
    reversed=np.concatenate([origin[:,3:6],origin[:,0:3],-origin[:,6:]],axis=1)
    return reversed

def double_batch(origin,reversed):
    # print("origin",origin)
    # print("rever",reversed)
    out=np.empty((2*len(origin),origin.shape[1]))
    out[0::2]=origin
    out[1::2]=reversed
    return out

node_mat = loadmat("dataset/node.mat")
S_mat = loadmat("dataset/S_fromV.mat")
nodeCoor  = node_mat['all_nodeCoor']
S = S_mat['S_fromV']

# ✅ 可视化所有节点的点云
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# all_nodeCoor 是 n×3 数组
x = nodeCoor[:, 0]
y = nodeCoor[:, 1]
z = nodeCoor[:, 2]

ax.scatter(x, y, z, s=5, c='b', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Node Coordinates Point Cloud')

plt.show()

#S = signed_log10(S)
# scale_node=1e1/np.max(np.abs(nodeCoor))
# scale_S=1e2/np.max(np.abs(S[:,output_dim]))

# nodeCoor=scaled(origin=nodeCoor,scale=scale_node)
# S=scaled(origin=S,scale=scale_S)
# S[np.abs(S)<1]=0
# param_json = os.path.join("dataset/scale.json")
# params = {
#     "scale_node": scale_node,
#     "scale_S": scale_S,
# }
# with open(param_json, "w") as f:
#         json.dump(params, f, indent=4)

# print(x_node.shape)
print("S shape",S.shape)
# print(S)
print("nodecoor shape",nodeCoor.shape)
print("use generate data")
# print(nodeCoor)

node_num=nodeCoor.shape[0]
nodePairs=[]
SPairs=[]
for i in range(node_num):
    for j in range(node_num):
        # print(f'i{i}j{j}')
        nodepair1=np.concatenate([
            nodeCoor[i],nodeCoor[j],
            (nodeCoor[i]-nodeCoor[j])
        ])

        spair=np.array([
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
        
        nodePairs.append(nodepair1)
        SPairs.append(spair)

nodePairs=np.array(nodePairs)
SPairs=np.array(SPairs)

# SPairs=SPairs[:,output_dim].reshape(-1,1)

print(nodePairs.shape)
# print("nodePairs???",nodePairs)
print(SPairs.shape)
# print("Spairs1???",SPairs1/scale_S)
# print("Spairs2???", SPairs2/scale_S)

test_ration = 1.0
total_size = nodePairs.shape[0]
print(f"total size {total_size}")
test_size = int(test_ration * total_size)
print(f"test size {test_size}")

np.random.seed(42)
indices = np.random.permutation(total_size)
test_indices = indices[:test_size]

test_inputs = nodePairs[test_indices]
test_labels = SPairs[test_indices]
print(f"test inputs{test_inputs.shape}")

train_inputs = nodePairs
train_labels = SPairs

# for i in range(9):
#     scale_S[i] = 1e2 / np.max(np.abs(train_lables[:,i]))
#     train_lables[:,i] = train_lables[:,i] / scale_S[i]

scale_node = 1e1 / np.max(np.abs(train_inputs))
scale_S = 1e3 / np.max(np.abs(train_labels), axis=0)
# scale_S = scale_S[:, np.newaxis]
print(f"scaleS {scale_S} shape {scale_S.shape}")

train_inputs_scaled = train_inputs * scale_node
train_labels_scaled = train_labels * scale_S

test_inputs_scaled = test_inputs * scale_node
test_labels_scaled = test_labels * scale_S

print(f"label {train_labels[0:3,:]}")
print(f"label scaled {train_labels_scaled[0:3,:]}")

param_json = os.path.join("dataset/scale.json")
params = {"scale_node": scale_node, "scale_S": scale_S.tolist()}
with open(param_json, "w") as f:
    json.dump(params, f, indent=4)

use_standerscale = True
if use_standerscale:
    scaler_train_inputs = StandardScaler()
    train_inputs_scaled = scaler_train_inputs.fit_transform(train_inputs_scaled)
    test_inputs_scaled = scaler_train_inputs.transform(test_inputs_scaled)

    joblib.dump(scaler_train_inputs, "dataset/scaler_inputs.pkl")

os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

np.save("dataset/train/train_inputs.npy", train_inputs)
np.save("dataset/test/test_inputs.npy", test_inputs)
# np.save("dataset/train/train_labels.npy", train_labels)
# np.save("dataset/test/test_labels.npy", test_labels)

np.save("dataset/train/train_inputs_scaled.npy", train_inputs_scaled)
# np.save("dataset/train/train_labels_scaled.npy", train_labels_scaled)
np.save("dataset/test/test_inputs_scaled.npy", test_inputs_scaled)
# np.save("dataset/test/test_labels_scaled.npy", test_labels_scaled)

for dim in range(9):
    np.save(f"dataset/train/train_labels{dim}.npy", train_labels[:, dim].reshape(-1, 1))
    print(train_labels[:, dim].reshape(-1, 1).shape)
    np.save(f"dataset/test/test_labels{dim}.np", test_labels[:, dim].reshape(-1, 1))

    np.save(
        f"dataset/train/train_labels_scaled{dim}.npy",
        train_labels_scaled[:, dim].reshape(-1, 1),
    )
    np.save(
        f"dataset/test/test_labels_scaled{dim}.npy",
        test_labels_scaled[:, dim].reshape(-1, 1),
    )
