import torch
import network as net
import torch.nn as nn

esp = 1e-8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import pandas as pd
import FPS as fps
import numpy as np
from data_loader import index_to_points
from fine_turn.coefficient_calu import corr_value
from tqdm import tqdm


data = pd.read_csv("../fine_turn/GPCD/G-PCD/subjective scores/desktop setup/subj_desktop_dsis.csv")
data_dict = {}
for row in range(data.shape[0]):
    if (data.iloc[row, 0].endswith('hidden')):
        data_dict[data.iloc[row, 0][0:-11]] = data.iloc[row, 1:].mean()
    else:
        data_dict[data.iloc[row, 0][:]] = data.iloc[row, 1:].mean()
# save_model = torch.load('../new_params.pth')
save_model = torch.load('new_params_lz.pth')
score_Net = net.weight_score()
score_Net.load_state_dict(save_model)
test_path = '../fine_turn/GPCD/G-PCD/stimuli' 
model_list = os.listdir(test_path)
# patch_number = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]
patch_number = [112]
PLCC_val = []
SRCC_val = []
KRCC_val = []
RMSE_val = []
repeat_number = 10
pbar = tqdm(total=len(patch_number) * repeat_number * len(model_list))
for num in patch_number:
    PLCC_accu = []
    SRCC_accu = []
    KRCC_accu = []
    RMSE_accu = []
    for k in range(repeat_number): 
        score_dict = {}
        for i in model_list:
            pbar.update(1)
            if i.endswith('.ply'):
                path = test_path + '/' + i
                model = fps.load_ply(path).reshape(1, -1, 3) 
                model = torch.from_numpy(model)
                centroids_index = fps.farthest_point_sample(model, num) 
                centroids = fps.index_points(model, centroids_index)
                result1 = fps.query_ball_point(0.2, 516, model, centroids)
                result1_np = result1.numpy()
                B, S, patch_size = result1_np.shape
                result1_value = np.zeros((B, S, patch_size, 3), dtype=float)
                model1_numpy = model.numpy() 
                for patch in range(S):
                    patch_index = result1_np[:, patch, :]  # [B patch_size]
                    value = index_to_points(model1_numpy, patch_index)  # [B patch_size C]
                    for batch in range(B):
                        result1_value[batch][patch] = value[batch]  # B X S X patch_size X C
                data1_tensor = torch.tensor(result1_value, dtype=torch.float)
                data1_tensor = fps.relative_cordinate(data1_tensor, centroids)
                data_tensor_trans = torch.transpose(data1_tensor, -1, -2)  # B X S x C X patch_size
                score = score_Net(data_tensor_trans)  # B X 1
                score = score[0]
                score_dict[i[0:-4]] = score.item()
        predict_MOS = np.zeros((len(score_dict), 2), dtype=float)
        key_list = list(score_dict.keys())
        for i in range(len(score_dict)):
            predict_MOS[i] = np.asarray([score_dict[key_list[i]], data_dict[key_list[i]]])
        x = predict_MOS[:, 0]
        y = predict_MOS[:, 1]
        PLCC, SRCC, KRCC, RMSE = corr_value(y, x)

        PLCC_accu.append(PLCC)
        SRCC_accu.append(SRCC)
        KRCC_accu.append(KRCC)
        RMSE_accu.append(RMSE)
    PLCC_val.append(PLCC_accu)
    SRCC_val.append(SRCC_accu)
    KRCC_val.append(KRCC_accu)
    RMSE_val.append(RMSE_accu)

np.save('PLCC_for_patch_number.npy', np.array(PLCC_val))
np.save('SRCC_for_patch_number.npy', np.array(SRCC_val))
np.save('KRCC_for_patch_number.npy', np.array(KRCC_val))
np.save('RMSE_for_patch_number.npy', np.array(RMSE_val))

print('PLCC_val', PLCC_val)
print('SRCC_val', SRCC_val)
print('KRCC_val', KRCC_val)
print('RMSE_val', RMSE_val)

print('PLCC_val mean', np.mean(PLCC_val))
print('SRCC_val mean', np.mean(SRCC_val))
print('KRCC_val mean', np.mean(KRCC_val))
print('RMSE_val mean', np.mean(RMSE_val))
