"""
Author: 雷尚谕
"""
import numpy as np
import torch
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

def scaled(origin,scale):
    return scale*origin

def signed_log10(x):
    return np.sign(x) * np.log10(np.abs(x)+1e-3)

def signed_pow10(x):
    print(f'pow input {x}')
    pow=np.pow(10,np.abs(x))
    print(f'pow {pow}')
    print(f'sign pow {np.sign(x) * (np.pow(10,np.abs(x)))}')
    print(f'pow output {np.sign(x) * (np.pow(10,np.abs(x))-1e-3)}')
    return np.sign(x) * (np.pow(10,np.abs(x))-1e-3)

def signed_log10_torch(x):
    return torch.sign(x) * torch.log10(torch.abs(x)+1)

def Loadmat():
    # node_mat=loadmat("dataset/surface2.mat")
    node_mat = loadmat("dataset/node.mat")
    # S_mat=loadmat("dataset/S.mat")
    S_mat=loadmat("dataset/S_fromV.mat")
    print("node",node_mat.keys())
    print("S",S_mat.keys())

    nodeCoor=node_mat['surface_ball1'][0,0]['nodeCoor']
    # nodeCoor=node_mat['surface_rev'][0,0]['nodeCoor']
    # S=S_mat['SS']
    S=S_mat['S_fromV']

    return nodeCoor,S

def plot_loss(loss_list,plot_name,folder_name):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label=plot_name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(folder_name, f'{plot_name}.png'), dpi=300)
    # plt.show()

    with open(os.path.join(folder_name, f"{plot_name}.txt"), 'w') as f:
        f.write(f"{plot_name} Values:\n")
        f.write("\n".join([f"{x:.6f}" for x in loss_list]))

def plot_loss_dual(loss_list1, loss_list2, plot_name, folder_name):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list1, label="train")
    plt.plot(loss_list2, label="test")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(folder_name, f'{plot_name}.png'), dpi=300)
    # plt.show()

    with open(os.path.join(folder_name, f"{plot_name}.txt"), 'w') as f:
        f.write("Train Loss Values:\n")
        f.write("\n".join([f"{x:.6f}" for x in loss_list1]) + "\n\n")
        f.write("Test Loss Values:\n")
        f.write("\n".join([f"{x:.6f}" for x in loss_list2]))

def plot3D(x1, y1, z1, x2, y2, z2, folder_name, plot_name):
    trace1 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.7
        ),
        name='Field 1'
    )

    trace2 = go.Scatter3d(
        x=x2,
        y=y2,
        z=z2,
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.7
        ),
        name='Field 2'
    )

    fig = go.Figure(data=[trace1, trace2])

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        title=plot_name,
        legend=dict(title='Field Legend')
    )

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    html_path = os.path.join(folder_name, f"{plot_name}.html")
    fig.write_html(html_path)

def plot4D_1e(x, y, z, s, folder_name, plot_name):
    color_vals = np.zeros_like(s, dtype=np.int32)
    color_vals[s >= 1e-6] = 0  # green
    color_vals[(s >= 1e-7) & (s < 1e-6)] = 1  # blue
    color_vals[(s >= 1e-8) & (s < 1e-7)] = 2  # purple
    color_vals[(s >= 1e-9) & (s < 1e-8)] = 3  # yellow
    color_vals[(s >= 1e-10) & (s < 1e-9)] = 4

    color_map = {
        0: 'rgb(0,255,0)',  # green
        1: 'rgb(0,0,255)',  # blue
        2: 'rgb(128,0,128)',  # purple
        3: 'rgb(255,255,0)',  # yellow
        4: 'rgb(255,0,0)',  # red
    }

    # ???????????????????????????
    colors = [color_map[val] for val in color_vals]

    # ??????????????????
    legend_labels = {
        0: '6',
        1: '7',
        2: '8',
        3: '9',
        4: '10',  # red
    }

    # ??????3D?????????
    fig = go.Figure()

    # ?????????????????????????????????trace??????????????????
    for color_val, color_rgb in color_map.items():
        mask = (color_vals == color_val)
        if np.any(mask):
            fig.add_trace(go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode='markers',
                marker=dict(
                    size=3,
                    color=color_rgb,
                ),
                name=legend_labels[color_val],
                showlegend=True
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        title=plot_name,
        legend=dict(
            title="Relative Error (%)",
            itemsizing='constant'
        ),
    )

    # ????????????????????????
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    html_path = os.path.join(folder_name, f"{plot_name}.html")
    fig.write_html(html_path)

def plot4D(x, y, z, s, folder_name, plot_name):

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=s,
            colorscale='viridis',
            colorbar=dict(title='w value')
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        title=plot_name
    )

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    html_path = os.path.join(folder_name, f"{plot_name}.html")
    fig.write_html(html_path)


def plot4D_error(x, y, z, s, folder_name, plot_name):
    # ????????????????????????????????????
    color_vals = np.zeros_like(s, dtype=np.int32)
    color_vals[s < 5] = 0  # green
    color_vals[(s >= 5) & (s < 10)] = 1  # blue
    color_vals[(s >= 10) & (s < 20)] = 2  # purple
    color_vals[(s >= 20) & (s < 30)] = 3  # yellow
    color_vals[(s >= 30) & (s < 100)] = 4  # orange
    color_vals[s >= 100] = 5  # red

    color_map = {
        0: 'rgb(0,255,0)',  # green
        1: 'rgb(0,0,255)',  # blue
        2: 'rgb(128,0,128)',  # purple
        3: 'rgb(255,255,0)',  # yellow
        4: 'rgb(255,165,0)',  # orange
        5: 'rgb(255,0,0)',  # red
    }

    # ???????????????????????????
    colors = [color_map[val] for val in color_vals]

    # ??????????????????
    legend_labels = {
        0: '???5%',
        1: '5-10%',
        2: '10-20%',
        3: '20-30%',
        4: '30-100%',
        5: '>100%'
    }

    # ??????3D?????????
    fig = go.Figure()

    # ?????????????????????????????????trace??????????????????
    for color_val, color_rgb in color_map.items():
        mask = (color_vals == color_val)
        if np.any(mask):
            fig.add_trace(go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode='markers',
                marker=dict(
                    size=3,
                    color=color_rgb,
                ),
                name=legend_labels[color_val],
                showlegend=True
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
        ),
        title=plot_name,
        legend=dict(
            title="Relative Error (%)",
            itemsizing='constant'
        ),
    )

    # ????????????????????????
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    html_path = os.path.join(folder_name, f"{plot_name}.html")
    fig.write_html(html_path)

def gen_mat():
# ??????????????????????????? (x,y,z)
    nodeCoor = np.array([
        [1.0, 1.0, 1.0],  # ??????0
        [2.0, 2.0, 2.0],  # ??????1
        [3.5, 3.5, 3.0]   # ??????2
    ])

    # ???????????????S?????? (3N x 3N)
    N = len(nodeCoor)
    S = np.zeros((3*N, 3*N))

    # ??????S????????????????????????????????????
    for i in range(N):
        for j in range(N):
            # ????????????
            if i == j:
                S[3*i:3*i+3, 3*j:3*j+3] = np.eye(3) * (i+1)
            # ???????????????
            else:
                dist = np.linalg.norm(nodeCoor[i] - nodeCoor[j])
                S[3*i:3*i+3, 3*j:3*j+3] = np.ones((3,3)) * 0.1 * (i+j)

    # ?????????MATLAB??????
    savemat("test_node.mat", {
        'surface_ball1': [[{
            'nodeCoor': nodeCoor
        }]]
    })

    savemat("test_S.mat", {
        'SS': S
    })