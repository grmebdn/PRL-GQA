import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import network as net
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from FPS import load_ply,farthest_point_sample,index_points,query_ball_point,relative_cordinate
import numpy as np
from data_loader import index_to_points
from fine_turn.coefficient_calu import corr_value




def FT_txt_gen(_input_dir_path, _out_train_txt, _out_test_txt, _ratio_for_train):   
    data = os.listdir(_input_dir_path)
    size = len(data)  
    pair_txt_train = open(_out_train_txt, mode="w", encoding="utf-8")
    pair_txt_test = open(_out_test_txt, mode="w", encoding="utf-8")
    scores = pd.read_csv("./G-PCD/subjective scores/desktop setup/subj_desktop_dsis.csv")   
    data_dict={}
    for row in range(scores.shape[0]):
        if(scores.iloc[row,0].endswith('hidden')):
            data_dict[scores.iloc[row,0][0:-11]]=scores.iloc[row,1:].mean()
        else:
            data_dict[scores.iloc[row,0][:]]=scores.iloc[row,1:].mean()

    for i in range(0, int(size * _ratio_for_train)):
        model= _input_dir_path + '/' + data[i] 
        pair_txt_train.write('%s %f\n' %(model,data_dict[data[i][0:-4]]))

    for i in range(int(size * _ratio_for_train), size):
        model = _input_dir_path + '/' + data[i] 
        pair_txt_test.write('%s %f\n' % (model, data_dict[data[i][0:-4]]))


def read_min_txt(root_txt_path):     
    assert root_txt_path.endswith('.txt')
    path_list,score_list = [],[]
    with open(root_txt_path, "r", encoding='utf-8') as f_txt:
        lines = f_txt.readlines() 
        for line in lines:
            path,score = line.strip().split(' ')
            score = float(score)
            path_list.append(path)
            score_list.append(score)
    return path_list,score_list

class MiniDataset(Dataset):     # GPCD
    def __init__(self, root_txt_path):
        super(MiniDataset, self).__init__()
        assert isinstance(root_txt_path, str) and root_txt_path.endswith('.txt')
        self.path_list,self.score_list = read_min_txt(root_txt_path)

    def __getitem__(self, index):

        path = self.path_list[index]
        score =self.score_list[index]
        model = load_ply(path).reshape(1,-1,3)  
        model = torch.from_numpy(model)
        centroids_index = farthest_point_sample(model, 64) 
        centroids = index_points(model, centroids_index)  
        result1 = query_ball_point(0.2, 516, model, centroids)
        result1_np = result1.numpy()
        B, S, patch_size = result1_np.shape
        result1_value = np.zeros((B, S, patch_size, 3), dtype=np.float)
        model1_numpy = model.numpy() 
        for patch in range(S):
            patch_index = result1_np[:, patch, :]  # [B patch_size]
            value = index_to_points(model1_numpy, patch_index)  # [B patch_size C]
            for batch in range(B):
                result1_value[batch][patch] = value[batch]  # B X S X patch_size X C
        data1_tensor = torch.tensor(result1_value, dtype=torch.float)

        data1_tensor = relative_cordinate(data1_tensor, centroids)
        data1_tensor =data1_tensor[0] # S X patch_size X C

        label_tensor = torch.tensor(score, dtype=torch.float)
        label_tensor = label_tensor.unsqueeze(-1)
        return  data1_tensor,label_tensor

    def __len__(self):
        return len(self.path_list)


def test(test_loader, net, criterion,_PLCC_file=None,_SRCC_file=None,_KRCC_file=None):
    loss = 0
    num = 0
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_mos=[]
    mos=[]
    for idx, (data1,label) in enumerate(test_loader):
        data1 = data1.to(device)
        data1 = torch.transpose(data1, -1, -2)
        label = label.to(device)
        out = net(data1)
        predict_mos.append(out.cpu().item())
        mos.append(label.cpu().item())
        loss_net = criterion(out, label)
        loss += loss_net.cpu().detach().item()*data1.size()[0]
        num+=data1.size()[0]
    mos = np.array(mos,dtype=np.float)
    predict_mos = np.array(predict_mos,dtype=np.float)
    PLCC,SRCC,KRCC,RMSE = corr_value(mos,predict_mos,False)
    print('\nTest set: Average loss: {:.4f}\n'.format(loss/num))
    print('PLCC :{0},SRCC:{1},KRCC:{2},RMSE:{3}'.format(PLCC,SRCC,KRCC,RMSE))
    if _PLCC_file and _SRCC_file and _KRCC_file :
        _PLCC_file.write(str(PLCC) + '\n')
        _SRCC_file.write(str(SRCC) + '\n')
        _KRCC_file.write(str(KRCC) + '\n')
    # return loss / num
    return PLCC,SRCC

def train_and_test():
    save_model = torch.load('../../params.pth')
    score_Net=net.weight_score()
    score_Net.load_state_dict(save_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score_Net=score_Net.to(device)
    train_Dataset = MiniDataset('./GPCD_train.txt')
    test_Dataset = MiniDataset('./GPCD_test.txt')
    trainloader = DataLoader(train_Dataset, batch_size=2, num_workers=0, shuffle=True,
                                 drop_last=False)
    testloader = DataLoader(test_Dataset, batch_size=1, num_workers=0, shuffle=False,
                                drop_last=False)
    criterion = nn.MSELoss()
    optimizer =optim.SGD(score_Net.parameters(), lr=0.001, momentum=0.9)
    epochs = 20
    SRCC_value = 2
    PLCC_value = 2
    best_epoch = 0
    lossfile_path = './MSEloss.txt'
    f_loss = open(lossfile_path, 'w', encoding='utf-8')
    PLCC_file = open('plcc.txt', 'w', encoding='utf-8')
    SRCC_file = open('SRCC.txt', 'w', encoding='utf-8')
    KRCC_file = open('KRCC.txt', 'w', encoding='utf-8')
    for epoch in range(1, epochs + 1):
        score_Net.train()
        # loss = 0
        for idx, (data1,label) in enumerate(trainloader):
            data1 = data1.to(device)
            data1 = torch.transpose(data1, -1, -2) 
            label = label.to(device)
            out = score_Net(data1)
            loss_net = criterion(out, label)
            # loss += loss_net.cpu().detach().item()
            optimizer.zero_grad()
            loss_net.backward()
            optimizer.step()
            print(loss_net.cpu().detach().item())

        PLCC,SRCC= test(testloader,score_Net,criterion,PLCC_file,SRCC_file,KRCC_file)
        if SRCC > SRCC_value:
            SRCC_value = SRCC
            PLCC_value = PLCC
            best_epoch = epoch
            torch.save(score_Net.state_dict(), 'minNET_params.pth')
    print('\nbest_epoch: {:d}, PLCC: {:.4f},SRCC : {:.4f}\n'.format(
            best_epoch, SRCC_value,PLCC_value))
    print("finish training")
    f_loss.close()







if __name__ == '__main__':


    train_and_test()