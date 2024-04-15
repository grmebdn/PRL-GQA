import torch
from torch.utils.data import Dataset, DataLoader
from FPS import load_ply,draw_points,farthest_point_sample,index_points,KNN_sample,query_ball_point,relative_cordinate
import numpy as np


def read_root_txt(root_txt_path):
    assert root_txt_path.endswith('.txt')     
    path1_list, path2_list, labels_list ,distortion_type= [], [], [],[]
    try:
        with open(root_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  
            for line in lines:
                _, path1, path2, label,distortion = line.strip().split(' ')
                label = int(label)
                path1_list.append(path1)
                path2_list.append(path2)
                labels_list.append(label)
                distortion_type.append(distortion)
    except UnicodeDecodeError:
        with open(root_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  
            for line in lines:
                _, path1, path2, label,distortion = line.strip().split(' ')
                label = int(label)
                path1_list.append(path1)
                path2_list.append(path2)
                labels_list.append(label)
                distortion_type.append(distortion)
    return path1_list, path2_list, labels_list,distortion_type

def read_sample_txt_files(txt_full_path):
    assert txt_full_path.endswith('.txt')
    with open(txt_full_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines()  
        info = lines[0].strip()
        info = info[1: -1]
        split_infos = info.split(',')
        split_infos = [float(elem) for elem in split_infos]
        return split_infos


def pc_normalize(pc):
    centroid = np.mean(pc, axis=1)  # B X D
    B = pc.shape[0]
    data =np.zeros(pc.shape,dtype=pc.dype.type)
    for batch in range(B):
        data[batch]=pc[batch]-centroid[batch]      # B x N x D
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=-1)),axis=-1)  # (B,)
    for batch in range(B):
        pc[batch] =data[batch]/m[batch]
    return pc



def index_to_points(points, idx):
    B = points.shape[0]
    S = idx.shape[1]
    C = points.shape[2]
    new_point=np.zeros((B,S,C),dtype=points.dtype.type)
    for batch in range(B):
        points_batch=points[batch]
        new_point[batch]=points_batch[idx[batch]]
    return new_point


def random_select(model1,patch_npy1,model2,patch_npy2,random_size):
    S=patch_npy1.shape[1]
    B=patch_npy1.shape[0]
    patch_size=patch_npy1.shape[2]
    C=model1.shape[2]
    index=np.arange(0,S)
    select_index=np.random.choice(index,random_size,replace=False)  
    select_index=select_index.reshape(-1,random_size)
    select_index=np.repeat(select_index,B,axis=0)   
    select_patch_index1=index_to_points(patch_npy1,select_index)  
    select_patch_index2=index_to_points(patch_npy2,select_index)
    result1 = np.zeros((B,random_size,patch_size,C),dtype=np.float)
    result2 = np.zeros((B, random_size, patch_size, C), dtype=np.float)
    for patch in range(random_size):
        patch_index=select_patch_index1[:,patch,:]   
        value =index_to_points(model1,patch_index)        
        for batch in range(B):
            result1[batch][patch]=value[batch]
    for patch in range(random_size):
        patch_index=select_patch_index2[:,patch,:]   # [B patch_size]
        value =index_to_points(model2,patch_index)        # [B patch_size C]
        for batch in range(B):
            result2[batch][patch]=value[batch]
    return result1,result2


class PointCloudDataset(Dataset):
    def __init__(self, root_txt_path,catergory=False,select_patch_numbers=64):
        super(PointCloudDataset, self).__init__()
        assert isinstance(root_txt_path, str) and root_txt_path.endswith('.txt')
        self.sample_path1_list, self.sample_path2_list, self.labels_list,self.distortion_type = read_root_txt(root_txt_path)
        self.category=catergory   
        self.select_patch_numbers=select_patch_numbers

    def __getitem__(self, index):
        path1 = self.sample_path1_list[index]
        path2 = self.sample_path2_list[index]  
        label = self.labels_list[index]
        model1 = load_ply(path1).reshape(1,-1,3)  
        model2 = load_ply(path2).reshape(1,-1,3)
        model1 = torch.from_numpy(model1)
        model2 = torch.from_numpy(model2)

        centroids_index = farthest_point_sample(model1, self.select_patch_numbers)  
        centroids = index_points(model1, centroids_index)   
        result1 = query_ball_point(0.2, 516, model1, centroids)
        result2 = query_ball_point(0.2, 516, model2, centroids)  
        result1_np = result1.numpy()
        result2_np = result2.numpy()
        B, S, patch_size = result1_np.shape
        result1_value = np.zeros((B, S, patch_size, 3), dtype=np.float)
        result2_value = np.zeros((B, S, patch_size, 3), dtype=np.float)
        model1_numpy = model1.numpy() 
        for patch in range(S):
            patch_index = result1_np[:, patch, :]  
            value = index_to_points(model1_numpy, patch_index)  
            for batch in range(B):
                result1_value[batch][patch] = value[batch]  
        model2_numpy = model2.numpy()  
        for patch in range(S):
            patch_index = result2_np[:, patch, :]  
            value = index_to_points(model2_numpy, patch_index)  
            for batch in range(B):
                result2_value[batch][patch] = value[batch]  
        data1_tensor = torch.tensor(result1_value, dtype=torch.float)
        data2_tensor = torch.tensor(result2_value, dtype=torch.float)

        data1_tensor = relative_cordinate(data1_tensor, centroids)
        data2_tensor = relative_cordinate(data2_tensor, centroids)
        data1_tensor =data1_tensor[0]
        data2_tensor =data2_tensor[0]    
        label_tensor = torch.tensor(label, dtype=torch.float)
        label_tensor = label_tensor.unsqueeze(-1)
        if self.category:
            return data1_tensor, data2_tensor, label_tensor,self.distortion_type[index]
        else:
            return data1_tensor, data2_tensor, label_tensor

    def __len__(self):
        return len(self.sample_path1_list)






if __name__ == '__main__':

    PCDataset = PointCloudDataset('./rank_pair_train.txt',True)
    trainloader = DataLoader(PCDataset, batch_size=1, num_workers=1, shuffle=True,
                             drop_last=False)
    for i, (sample1_tensor, sample2_tensor, label_tensor,type_tensor) in enumerate(trainloader):
        print(label_tensor)
        print(type_tensor[0]=='com&down')



