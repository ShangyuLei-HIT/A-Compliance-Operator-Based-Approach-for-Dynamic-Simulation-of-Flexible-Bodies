import numpy as np
from scipy.io import savemat

# 创建三个节点的坐标 (x,y,z)
nodeCoor = np.array([
    [1.0, 1.0, 1.0],  # 节点0
    [2.0, 2.0, 2.0],  # 节点1
    [3.5, 3.5, 3.0]   # 节点2
])

# 创建对应的S矩阵 (3N x 3N)
N = len(nodeCoor)
S = np.zeros((3*N, 3*N))

# 填充S矩阵的示例值（对称矩阵）
for i in range(N):
    for j in range(N):
        # 对角线块
        if i == j:
            S[3*i:3*i+3, 3*j:3*j+3] = np.eye(3) * (i+1)
        # 非对角线块
        else:
            dist = np.linalg.norm(nodeCoor[i] - nodeCoor[j])
            S[3*i:3*i+3, 3*j:3*j+3] = np.ones((3,3)) * 0.1 * (i+j)

# 保存为MATLAB格式
savemat("test_node.mat", {
    'surface_ball1': [[{
        'nodeCoor': nodeCoor
    }]]
})

savemat("test_S.mat", {
    'SS': S
})