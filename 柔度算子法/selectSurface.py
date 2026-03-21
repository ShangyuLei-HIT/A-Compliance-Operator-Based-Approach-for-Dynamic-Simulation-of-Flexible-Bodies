from scipy.io import savemat
from scipy.io import loadmat
import numpy as np
import h5py
import matplotlib.pyplot as plt
# node_mat=loadmat("dataset/node.mat")
# nodeCoor=node_mat['surface_ball1'][0,0]['nodeCoor']
# nodeIndex=node_mat['surface_ball1'][0,0]['nodeIndex']

data = loadmat("dataset/surface_rod_ball0d5_3_2_2_2_2.mat")

surface_rod = data['surface_rod']  # shape: (1, 5) struct

print(surface_rod.dtype)
print(surface_rod[0,0].dtype)

contact_data = {}

for i in range(surface_rod.shape[1]):
    geom_struct = surface_rod[0, i]

    nodeIndex = geom_struct['nodeIndex']
    nodeCoor  = geom_struct['nodeCoor']

    geom_name = f'contactGeom-{i+1}'
    contact_data[geom_name] = {
        'nodeIndex': nodeIndex,
        'nodeCoor': nodeCoor
    }

    print(f"{geom_name}: nodeIndex {nodeIndex.shape}, nodeCoor {nodeCoor.shape}")

# 4. 示例：访问 contactGeom-1 的数据
nodeIndex1 = contact_data['contactGeom-1']['nodeIndex']
nodeCoor1  = contact_data['contactGeom-1']['nodeCoor']

nodeIndex2 = contact_data['contactGeom-2']['nodeIndex']
nodeCoor2  = contact_data['contactGeom-2']['nodeCoor']

all_nodeIndex = np.vstack([contact_data[f'contactGeom-{i+1}']['nodeIndex'] for i in range(5)])
all_nodeCoor  = np.vstack([contact_data[f'contactGeom-{i+1}']['nodeCoor'] for i in range(5)])

print("合并后：")
print("nodeIndex shape:", all_nodeIndex.shape)
print("nodeCoor shape:", all_nodeCoor.shape)

savemat("dataset/node.mat", {
    'all_nodeIndex': all_nodeIndex,
    'all_nodeCoor': all_nodeCoor
})
print(" 已保存 node.mat 文件（包含 all_nodeIndex 与 all_nodeCoor）")

#  可视化所有节点的点云
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# all_nodeCoor 是 n×3 数组
x = all_nodeCoor[:, 0]
y = all_nodeCoor[:, 1]
z = all_nodeCoor[:, 2]

ax.scatter(x, y, z, s=5, c='b', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Node Coordinates Point Cloud')

plt.show()

with h5py.File("dataset/S_rod_ball0d5_3_2_2_2_2.mat", 'r') as f:
# with h5py.File("dataset/S_revoluteJoint_0d4_2.mat", 'r') as f:
    print(list(f.keys()))
    S_v = f['S'][()]  # 提取数据集 'S' 的值

# S_v=np.array(S_v)
# print(S_v.shape)

node_num = all_nodeCoor.shape[0]
idx = (all_nodeIndex.flatten() - 1).astype(int)
id3 = np.repeat(idx * 3, 3) + np.tile(np.arange(3), node_num)

S = S_v[np.ix_(id3, id3)]

savemat("dataset/S_fromV.mat", {'S_fromV': S})


# node_num=all_nodeCoor.shape[0]
# S=np.empty((3*node_num,3*node_num))
# S_base=np.empty((3*node_num,3*node_num))
# for i in range(node_num):
#     for j in range(node_num):
#         i_id=all_nodeIndex[i]-1
#         j_id=all_nodeIndex[j]-1
#
#         for ii in range(3):
#             for jj in range(3):
#                 # print(S_v[3*i_id+ii,3*j_id+jj].shape)
#                 S[3 * i + ii, 3 * j + jj] = S_v[3 * i_id + ii, 3 * j_id + jj].item()
#
#                 # if j>=i:
#                 #     S_base[3*i+ii,3*j+jj]=(S_v[3*i_id+ii,3*j_id+jj].item()+S_v[3*j_id+ii,3*i_id+jj].item())*0.5
#                 #     S_base[3*j+ii,3*i+jj]=S[3*i+ii,3*j+jj]
#
# savemat("dataset/S_fromV.mat", {
#         'S_fromV': S
#     })
# savemat("dataset/S_fromV_base.mat", {
#         'S_fromV_base': S_base
#     })


