import torch
import torch.nn.functional as F
import numpy as np
import h5py
from plyfile import PlyData
import os


def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)    

def load_ply(ply_filename):     
    plydata=PlyData.read(ply_filename)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    tmp = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=-1)
    tmp = np.concatenate((tmp,z.reshape(-1,1)),axis=-1)
    return tmp
def index_points(points, idx):   
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1        
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)   # [B S]
    new_points = points[batch_indices, idx, :]    
    return new_points


def draw_points(points, idx):  
    device = points.device
    B, N, C = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    labels = torch.ones((B, N, 1)).to(device)
    labels[batch_indices, idx, :] = 0
    points_np = np.array(points.cpu())
    labels_np = np.array(labels.cpu())
    drawed_points = np.concatenate((points_np, labels_np), axis=-1)
    return drawed_points


def farthest_point_sample(xyz, npoint):  
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)     
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  
    return centroids

def sample_index_save(in_path,out_path,samply_numbers):  
    xyz=load_ply(in_path).reshape(1,-1,3)
    xyz = torch.from_numpy(xyz).to('cuda')
    centroids = farthest_point_sample(xyz, samply_numbers)
    centroids=np.array(centroids.cpu())
    np.save(out_path,centroids)    


def FPS_sample_for_data(root_path,samply_number): 
    data = os.listdir(root_path)
    size = len(data)  
    for i in range(0, size):
        model_dir = os.listdir(root_path + '/' + data[i])
        for j in model_dir:
            path_j = root_path + '/' + data[i] + '/' + j
            if os.path.isfile(path_j) and j.endswith(".ply"):
                out_path=root_path + '/' + data[i] + '/'+j[0:-4]+'.npy'   
                sample_index_save(path_j,out_path,samply_number)


def KNN_sample(source_set: torch.Tensor, center_set: torch.Tensor, patch_size: int):
    device = source_set.device
    B = source_set.shape[0]
    S = center_set.shape[1]
    result = torch.zeros(B, S, patch_size, dtype=torch.long).to(device)
    for b in range(B):
        source2d, center2d = source_set[b], center_set[b]
        for i in range(S):
            L2_distance = torch.norm(center2d[i] - source2d, dim=-1)
            min_val, min_idx = torch.topk(L2_distance, patch_size, largest=False, sorted=True)
            result[b, i] = min_idx
    return result

def patch_sample_for_data(root_path,patch_size): 
    data = os.listdir(root_path)
    size = len(data) 
    for i in range(0, size):
        model_dir = os.listdir(root_path + '/' + data[i])
        center_set_list=[]
        for j in model_dir: 
            path_j = root_path + '/' + data[i] + '/' + j
            if os.path.isfile(path_j) and j.endswith(".ply"):  
                source_set=load_ply(path_j).reshape(1,-1,3) 
                source_set=torch.from_numpy(source_set).to('cuda')
                refer_set=source_set
                center_index=np.load(root_path + '/' + data[i] + '/'+j[0:-4]+'.npy') 
                center_index=torch.from_numpy(center_index).to('cuda')
                center_set=index_points(refer_set,center_index)
                center_set_list.append(center_set.cpu())
                result=KNN_sample(source_set,center_set,patch_size)
                out_path=root_path + '/' + data[i] + '/' +j[0:-4]+'_sample_patch.npy'
                np.save(out_path,np.array(result.cpu()))
        center_set=center_set_list[0].to('cuda')
        for j in model_dir:
            if os.path.isdir(root_path + '/' + data[i] + '/' + j):
                for k in os.listdir(root_path + '/' + data[i] + '/' + j):
                    path = root_path + '/' + data[i] + '/' + j + '/' + k
                    if k.endswith(".ply"):
                        source_set = load_ply(path).reshape(1,-1,3)
                        source_set = torch.from_numpy(source_set).to('cuda')
                        result=KNN_sample(source_set,center_set,patch_size)
                        out_path = root_path + '/' + data[i] + '/'+ j + '/' + k[0:-4]+'_sample_patch.npy'
                        np.save(out_path,np.array(result.cpu()))


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # B x S X N  [[[0,1...,N-1],[0,1,....N-1]..]]
    sqrdists = square_distance(new_xyz, xyz)   
    group_idx[sqrdists > radius ** 2] = N      
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]     
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) 
    new_xyz = index_points(xyz, fps_idx)    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)    
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)   

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm  
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def relative_cordinate(xyz,centroids):
    B,S,C = centroids.shape
    return xyz - centroids.view(B,S,1,C)



if __name__ == '__main__':
    root_path = './data5'
    FPS_sample_for_data(root_path,72)
    patch_sample_for_data(root_path,256)







