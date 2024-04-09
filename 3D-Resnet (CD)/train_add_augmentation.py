'''本代码实现将前面matlab半自动框之后一层一个h5的数据读取进来，然后将各个患者的ct,mask,grade保存到各自序号的h5文件中
h5文件里面分别放患者的ct,mask,grade'''
# sep = os.sep
import os
import csv
import copy
import cv2
import math
import xlwt
import numpy as np
import pandas as pd
from torch.nn import init
import argparse
import torch
import torch.nn as nn
import h5py
import sklearn.metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score
from sklearn.preprocessing import label_binarize
from resnet3d_50 import ResNet101, ResNet101_m
import matplotlib.pyplot as plt
import random
import time
from scipy.ndimage import zoom
from BCEfocalloss import BCEFocalLoss
import pickle
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import precision_score,recall_score,f1_score


root_path = r'/data/nas/阑尾炎数据/FYH/experiment/Cr/Second/data/h5_data_for_zx3Dresnet/H5_save/' # 数据路径
# root_path_aug = r'/media/ds/新加卷/LZX_Crohn/fibrosis_scanner_data/scanners30_random10'
info_pkl_path = open(r'/data/nas/阑尾炎数据/FYH/experiment/Cr/Second/data/Cr_information.pkl','rb')
pkl_info = pickle.load(info_pkl_path)
test_root_path = r'../scanner1_test/'
file = os.listdir(root_path)

exp_name = "exp1"
root_result = r'/data/fanyaheng/LWY/ZX_3dresnet_335Cr/result/'+exp_name
model_save_dir = "/data/fanyaheng/LWY/ZX_3dresnet_335Cr/models/"+exp_name

train_aug = False
Augtimes = 5
expend_voxel_size = 20
input_shape = (224, 224)


parser = argparse.ArgumentParser()
parser.add_argument('--root', default=root_path, help='path to dataset (images list file)')
parser.add_argument('--block_z_slice', type=int, default=32, help='3d_block_z_slice')
parser.add_argument('--block_z_sample_slice', type=int, default=16, help='3d_block_z_sample_slice')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--optim', type=str, default='Adam', help='optim for training, Adam / SGD (default)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD / Adam')
parser.add_argument('--gpu', type=bool, default=True, help='use GPU or not')
args = parser.parse_known_args()[0]
print(args)

# models
model = ResNet101_m()
# models = resnet3d(1, 1)

if args.gpu:
    model = model.cuda()


# criterion
criterion = BCEFocalLoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss().cuda()
# criterion = nn.BCELoss()

# optim
if args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    raise NotImplementedError('Other optimizer is not implemented')

### transform utilss#########################################################################

def grade_revise(ct, grade): #按照ct的shape矫正grade
    g = np.max(grade)
    if g == 0:
        G = np.zeros((ct.shape[0], 1, 1))
    elif g == 1:
        G = np.ones((ct.shape[0], 1, 1))
    return G

def mask_revise(mask):
    mask[mask<0.5] = 0
    mask[mask>=0.5] = 1
    return mask

def data_revise(data):
    data[data>1] = 1
    data[data<0] = 0
    return data

def getListIndex(arr, value) :
    dim1_list = dim2_list = dim3_list = []
    if (arr.ndim == 3):
        index = np.argwhere(arr == value)
        dim1_list = index[:, 0].tolist()
        dim2_list = index[:, 1].tolist()
        dim3_list = index[:, 2].tolist()

    else :
        raise ValueError('The ndim of array must be 3!!')

    return dim1_list, dim2_list, dim3_list

def ROI_cutting(img, mask, extra_expend_voxel):

    [I1, I2, I3] = getListIndex(mask, 1)
    d1_min = min(I1)
    d1_max = max(I1)
    d2_min = min(I2)
    d2_max = max(I2)
    d3_min = min(I3)
    d3_max = max(I3)

    if extra_expend_voxel > 0:
        # d1_min -= expend_voxel
        # d1_max += expend_voxel
        d2_min -= extra_expend_voxel
        d2_max += extra_expend_voxel
        d3_min -= extra_expend_voxel
        d3_max += extra_expend_voxel

        # d1_min = d1_min if d1_min>0 else 0
        # d1_max = d1_max if d1_max<data.shape[0]-1 else data.shape[0]-1
        d2_min = d2_min if d2_min>0 else 0
        d2_max = d2_max if d2_max<img.shape[1]-1 else img.shape[1]-1
        d3_min = d3_min if d3_min>0 else 0
        d3_max = d3_max if d3_max<img.shape[2]-1 else img.shape[2]-1

    data = img[d1_min:d1_max+1,d2_min:d2_max+1,d3_min:d3_max+1]
    roi = mask[d1_min:d1_max+1,d2_min:d2_max+1,d3_min:d3_max+1]

    return data, roi

def resize_3d(img, transform_size=None, transform_rate=None):
    data = img
    if transform_size:
        o_width, o_height, o_queue = data.shape
        width, height, queue = transform_size
        data = zoom(data, (width/o_width, height/o_height, queue/o_queue))
    elif transform_rate:
        data = zoom(data, transform_rate)

    return data

def my_resize(img, resize_shape):
    z, x, y = img.shape
    data0 = img
    scale_z = (x/resize_shape[0]+y/resize_shape[1])/2
    resize_z = round(z/scale_z)
    data1 = resize_3d(data0, transform_size=(resize_z, resize_shape[0], resize_shape[1]))
    return data1

def train_data_normalization(img, mask, grade, extra_expend_voxel, resize_shape):
    I1 = img
    L1 = mask_revise(mask)
    G1 = grade
    I, L = ROI_cutting(I1, L1, extra_expend_voxel=extra_expend_voxel)
    I = my_resize(I, resize_shape=resize_shape)
    L = my_resize(L, resize_shape=resize_shape)
    L = mask_revise(L)
    G = grade_revise(I, G1)
    return I, L, G

# 调整图像形态
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

def rotate_3d(data_s,angl):
    data_s1 = []
    for i in range(len(data_s)):
        data_r = rotate(data_s[i], angl, None, 1.0)
        if i == 0:
            data_s1 = data_r[np.newaxis, :]
        else:
            data_r = data_r[np.newaxis, :]
            data_s1 = np.concatenate((data_s1, data_r), axis=0)
    return data_s1

def flip_3d(img, flip_flag):
    for z in range(len(img)):
        img_s = img[z]
        img_s = cv2.flip(img_s, flip_flag)
        if z == 0:
            img_f = img_s[np.newaxis, :]
        else:
            img_s = img_s[np.newaxis, :]
            img_f = np.concatenate((img_f, img_s), axis=0)
    return img_f

class Rotation(object):
    def __init__(self, angle=(-90, 90), p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:

            angle_factor = torch.tensor(1.0).uniform_(self.angle[0], self.angle[1]).item()
            img = rotate_3d(img, angle_factor)

        return img

class Flip(object):
    def __init__(self, flip_flag=(-1.5, 1.5), p=0.5):
        self.flip_flag = flip_flag
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            flip_factor = torch.tensor(1.0).uniform_(self.flip_flag[0], self.flip_flag[1]).item()
            flip_factor = round(flip_factor)
            img = flip_3d(img, flip_factor)

        return img

def data_augmentation(img):
    transform_dict = {1: Rotation(), 2: Flip()}
    for f in range(1, len(transform_dict) + 1):
        TF = transform_dict[f]
        img = TF(img)
        if np.sum(np.isnan(img).astype(int)) > 0:
            raise ValueError("INPUT NAN !!!!")
        img = data_revise(img)
    return img

def TTA(img):
    transform_dict = {1: Rotation(), 2: Flip()}
    for f in range(1, len(transform_dict) + 1):
        TF = transform_dict[f]
        img = TF(img)
        if np.sum(np.isnan(img).astype(int)) > 0:
            raise ValueError("INPUT NAN !!!!")
        img = data_revise(img)
    return img

###########################################################################################

def get_h5_perpatient_perslice(root_path, file, train=True, three_d=False):
    num_list_slice = []
    if train == True and three_d == False: # 2D训练
        ct_slice = []
        seg_gt_slice = []
        grade_slice = []
        for i in range(len(file)): # 患者
            patient_num_path = root_path + '/' + file[i]  # 每个患者文件夹的路径
            patient_slice = os.listdir(patient_num_path)  # 每个患者文件夹的多个h5文件
            for ii in range(len(patient_slice)): # 每个患者有病灶层数
                patient_slice_path = patient_num_path + '/' + patient_slice[ii]  # 一层一个h5文件的路径
                with h5py.File(patient_slice_path, 'r') as f:
                    ct = f["data"][:]
                    seg_gt = f["roi"][:]
                    grade = f["grade"][:]

                    ct_slice.append(ct)  # 所有患者每一层有病灶的ct图像
                    seg_gt_slice.append(seg_gt)  # 所有患者每一层有病灶的分割金标准
                    grade_slice.append(grade)

                    num = int(file[i])
                    num_list_slice.append(num)  # 每一层对应是哪个患者编号的
        return list(zip(ct_slice, seg_gt_slice, grade_slice)), num_list_slice

    else: # 3D训练
        num_patient = []
        ct_seg_grade_patient = []
        for i in range(len(file)):  # 患者
            ct_slice = []
            seg_gt_slice = []
            grade_slice = []
            num_list_slice = []
            patient_num_path = root_path + '/' + file[i]  # 每个患者文件夹的路径
            patient_slice = os.listdir(patient_num_path)  # 每个患者文件夹的多个h5文件 注意：读进来的文件顺序是乱的！！
            '''对文件名层数进行排序，不然后面得到的3d_block不是连续层的！'''
            patient_slice.sort(key=lambda x: int(x.split('.')[0]))
            for ii in range(len(patient_slice)):  # 每个患者有病灶层数
                patient_slice_path = patient_num_path + '/' + patient_slice[ii]  # 一层一个h5文件的路径
                with h5py.File(patient_slice_path, 'r') as f:
                    ct = f["data"][:]
                    seg_gt = f["roi"][:]
                    grade = f["grade"][:]

                ct_slice.append(ct)  # 所有患者每一层有病灶的ct图像
                seg_gt_slice.append(seg_gt)  # 所有患者每一层有病灶的分割金标准
                grade_slice.append(grade)
                slice_ct_seg = list(zip(ct_slice, seg_gt_slice, grade_slice)) # 每一层ct,seg,grade合并到同一个list,元组结果按矩阵最短的算，所以grade弄成列表

                num = int(file[i])
                # num_list_slice.append(num)  # 每一层对应是哪个患者编号的
                num_list_slice = num          # 每个患者的编号
                # num_list.append(num_list_slice) # 每一层的编号

            ct_seg_grade_patient.append(slice_ct_seg)  #  所有患者合并list
            num_patient.append(num_list_slice)  # 所有患者每一层的编号

        if three_d == False:
            return ct_seg_grade_patient, num_patient
        else:
            data_3d = change_to_3d_data(ct_seg_grade_patient)
            return data_3d, num_patient

def change_to_3d_data(data):
    data_3d = []
    for i in range(len(data)):
        for ii in range(len(data[i])):
            ct_slice = data[i][ii][0]
            seg_slice = data[i][ii][1]
            grade_slice = data[i][ii][2]
            if ii == 0:
                ct_3d = ct_slice[np.newaxis, :]
                seg_3d = seg_slice[np.newaxis, :]
                grade_3d = grade_slice[np.newaxis, :]
            else:
                ct_slice = ct_slice[np.newaxis, :]
                ct_3d = np.concatenate((ct_3d, ct_slice), axis=0)

                seg_slice = seg_slice[np.newaxis, :]
                seg_3d = np.concatenate((seg_3d, seg_slice), axis=0)

                grade_slice = grade_slice[np.newaxis, :]
                grade_3d = np.concatenate((grade_3d, grade_slice), axis=0)
        ct_3d, seg_3d, grade_3d = train_data_normalization(ct_3d, seg_3d, grade_3d, extra_expend_voxel=expend_voxel_size, resize_shape=input_shape)
        a = list([ct_3d])
        b = list([seg_3d])
        c = list([grade_3d])
        da = list(zip(a, b, c))
        data_3d.append(da)
    return data_3d

def show_2d(data, mask):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(data, cmap='gray')
    # ax.imshow(data[1, :, :], cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    ax = fig.add_subplot(122)
    ax.imshow(mask, cmap='gray')
    # ax.imshow(mask[1, :, :], cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    # plt.imshow(data2,cmap='gray')	# 也可以这样show image，但我不知道怎么同时显示两张 ////img.set_cmap('gray')  # 'hot' 是热量图
    plt.show()

def show_3d(data, mask, slice):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # ax.imshow(data, cmap='gray')
    ax.imshow(data[slice, :, :], cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    ax = fig.add_subplot(122)
    # ax.imshow(mask, cmap='gray')
    ax.imshow(mask[slice, :, :], cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    # plt.imshow(data2,cmap='gray')	# 也可以这样show image，但我不知道怎么同时显示两张 ////img.set_cmap('gray')  # 'hot' 是热量图
    plt.show()

def get_block_res(data, p, z_slice, block_slice, remian_slice):
    # 对剩余层数补零层
    # block_res = []
    # ---剩余层
    ct_res = data[p][0][0][block_slice * z_slice:]
    seg_res = data[p][0][1][block_slice * z_slice:]
    grade_res = data[p][0][2][block_slice * z_slice:]
    zero_slice = z_slice - remian_slice
    # ---补充层
    for m in range(zero_slice):
        ct_zero = np.zeros(input_shape).astype(np.float32)
        ct_zero = ct_zero[np.newaxis, :]

        if grade_res[0][0][0] == 0:
            grade_zero = np.zeros((1, 1)).astype(np.float32)
            grade_zero = grade_zero[np.newaxis, :]
        elif grade_res[0][0][0] == 1:
            grade_zero = np.ones((1, 1)).astype(np.float32)
            grade_zero = grade_zero[np.newaxis, :]
        ct_res = np.concatenate((ct_res, ct_zero), axis=0)
        seg_res = np.concatenate((seg_res, ct_zero), axis=0)
        grade_res = np.concatenate((grade_res, grade_zero), axis=0)

    d = list([ct_res])
    e = list([seg_res])
    f = list([grade_res])
    block_zero = list(zip(d, e, f))
    # block_res.append(block_zero)
    return block_zero

def my_get_3d_block(data, block_z_slice, sample_slice): # 训练数据采用：每隔16层采样一次，不足16的向前采样
    z_slice = block_z_slice
    block_all_per_patient = []
    block_all = []
    for p in range(len(data)):  # 患者list
        block_sum = []
        slice = len(data[p][0][0])  # 每个患者有多少层

        block_slice, remian_slice = divmod(slice, z_slice)
        if block_slice == 0: # 不足block_z_slice层，补0层
            block_zero = get_block_res(data, p, z_slice, block_slice, remian_slice)
            block_sum.append(block_zero)
        elif block_slice != 0: # 大于1个block_z_slice的每隔16采样一次，最后采样不足block_z_slice层数的向前采样（即不再补0）
            sample_block_slice, sample_remain_slice = divmod(slice, sample_slice)
            for n in range(sample_block_slice - 1):
                ct = data[p][0][0][n * sample_slice:n * sample_slice + z_slice]
                seg = data[p][0][1][n * sample_slice:n * sample_slice + z_slice]
                grade = data[p][0][2][n * sample_slice:n * sample_slice + z_slice]
                a = list([ct])
                b = list([seg])
                c = list([grade])
                block = list(zip(a, b, c))  # 三者合并成一个list(在一起～)
                block_sum.append(block)  # 图像块append
            if sample_remain_slice != 0:
                ct = data[p][0][0][slice-z_slice:]
                seg = data[p][0][1][slice-z_slice:]
                grade = data[p][0][2][slice-z_slice:]
                a = list([ct])
                b = list([seg])
                c = list([grade])
                block = list(zip(a, b, c))  # 三者合并成一个list(在一起～)
                block_sum.append(block)  # 图像块append

        block_all_per_patient.append(block_sum)  # 按照患者保存各自的block
        block_all.extend(block_sum)  # 合并所有患者的block
    return block_all_per_patient, block_all

# lr_reduce
def lr_opt(lr, epoch):
    # e_max = 60
    # if epoch <= e_max:
    #     lr_new = lr * (0.1 ** (float(epoch) / 15))
    # else:
    #     lr_new = lr * (0.1 ** (float(e_max) / 15))

    # # method 1
    # if epoch < 30:
    #     lr_new = lr
    # else:
    #     lr_new = lr * 0.1
    # method 2
    if epoch < 20:
        lr_new = lr
    elif epoch < 50:
        lr_new = lr * 0.5
    else:
        lr_new = lr * 0.1

    return lr_new

def batch(iterable, batch_size):  # 一次输入多个数据（batch size）
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):  # i从0开始，t是data里面的数据
        b.append(t[0])
        if (i + 1) % batch_size == 0:
            yield b  # yield很像return，只是下次进来这个代码会从上一次停止的地方继续而不是从头开始
            b = []

    if len(b) > 0:
        yield b

# train
def train_net(data, model, criterion, optimizer, epoch, lr_cur, fold):
    loss_epoch = []
    random.shuffle(data)
    dataloader = batch(data, args.batch_size)  # 每次读入一个batch的多个数据 4*16*256*256
    model.train()
    print('***************************** Train **************************************')
    all_possibility, all_pred_class, all_grade = [], [], []
    for i, b in enumerate(dataloader):  # 对图像进行训练
        imgs = np.array([i[0] for i in b]).astype(np.float32)  # b[0][0] i依次等于b[0]b[1],然后每个b都有一个image和mask,所以分别用i[0]i[1]代表提取，并设置好类型
        grades = np.array([i[2] for i in b]).astype(np.float32)
        for bs in range(len(grades)):
            if grades[bs].max() == 0:
                grade = np.zeros(1)
            elif grades[bs].max() == 1:
                grade = np.ones(1)
            if bs == 0:
                grade_input = grade[np.newaxis, :]
            else:
                grade = grade[np.newaxis, :]
                grade_input = np.concatenate((grade_input, grade), axis=0)
        img_all = imgs
        # 转tensor并进行训练______________________________________________________________________________________________
        imgs = torch.from_numpy(img_all).unsqueeze(1)  # 转成tensor；并，对指定位置，unsqueeze是对数据维度进行扩充，squeeze是对数据维度进行压缩
        grades_label = torch.from_numpy(grade_input.astype(np.float32))  # .unsqueeze(1)

        if args.gpu:
            imgs = imgs.float()
            imgs = imgs.cuda()
            grades_label = grades_label.cuda()

        # 输入模型________________________________________________________________________________________________________
        outputs = model(imgs)
        loss = criterion(outputs, grades_label)
        loss_epoch.append(loss.item())
        loss.backward()  # 即反向传播求梯度
        optimizer.step()  # 即更新所有参数
        optimizer.zero_grad()  ### 原来在这里！！！！！！！！！！！！！！！！
        # 累计同一患者每个3d块的预测结果和分类金标准
        if args.gpu:
            outputs_np = outputs.cpu().detach().numpy()
            grades_label_np = grades_label.cpu().detach().numpy()
        outputs_list = outputs_np.squeeze().tolist()
        grades_label_list = grades_label_np.squeeze().tolist()
        pred_list = [0 if x < 0.5 else 1 for x in outputs_list]
        fpr, tpr, threshold = roc_curve(grades_label_list, outputs_list)
        auc_train = auc(fpr, tpr)
        acc_train = accuracy_score(grades_label_list, pred_list)
        #print('Fold:{}, Epoch:{}, batch:{}/{}, Lr:{}, loss:{}, auc:{}, acc:{}'.format(fold, epoch, i+1, len(data)//args.batch_size+1, lr_cur, loss.item(), auc_train, acc_train))

        all_possibility = all_possibility + outputs_list
        all_pred_class = all_pred_class + pred_list
        all_grade = all_grade + grades_label_list

    fpr, tpr, threshold = roc_curve(all_grade, all_possibility)
    all_train_auc = auc(fpr, tpr)
    all_train_acc = accuracy_score(all_grade, all_pred_class)
    print('Epoch:{} Finished, Lr:{}, Loss:{}, Auc:{}, Acc:{}'.format(epoch, lr_cur, np.mean(loss_epoch), all_train_auc, all_train_acc))

    return loss_epoch

# val
def eval_net(data, model, epoch, fold, val=True):
    """Evaluation validation or test"""
    model.eval()
    if val == True:
        loss_val_all = []
        all_probability = []
        all_grade = []
        all_pred_class = []
        print('***************************** Validation **************************************')
        for ind, patient in enumerate(data):  #  读每个人的所有3d块
            loss_val = []
            # dataloader = batch(patient, 1)
            for i, b in enumerate(patient):    #  读每一3d块
                imgs = np.array([i[0] for i in b]).astype(np.float32)  # b[0][0] i依次等于b[0]b[1],然后每个b都有一个image和mask,所以分别用i[0]i[1]代表提取，并设置好类型
                grades = np.array([i[2] for i in b]).astype(np.float32)
                #  弄grade一维数组，在有batch size不等于1的时候扩充一维，得到batch size * 1的数组，batch size等于1时，得到1.数组或者扩充后变成1 * 1的数组
                for bs in range(len(grades)):
                    if grades[bs].max() == 0:
                        grade = np.zeros(1)
                    elif grades[bs].max() == 1:
                        grade = np.ones(1)
                    # 验证测试时batch size=1，所以下面其实可以不用扩充一维。
                    if bs == 0:
                        grade_input = grade[np.newaxis, :]
                    else:
                        grade = grade[np.newaxis, :]
                        grade_input = np.concatenate((grade_input, grade), axis=0)
                # 转tensor并进行训练______________________________________________________________________________________________
                imgs = torch.from_numpy(imgs).unsqueeze(1)  # 转成tensor；并，对指定位置，unsqueeze是对数据维度进行扩充，squeeze是对数据维度进行压缩
                grades_label = torch.from_numpy(grade_input.astype(np.float32))  # .unsqueeze(1)
                # 数据用gpu跑
                if args.gpu:
                    imgs = imgs.cuda()
                    grades_label = grades_label.cuda()
                # 输入到模型________________________________________________________________________________________________________
                outputs = model(imgs)
                loss = criterion(outputs, grades_label)
                loss_val.append(loss.item())
                # tensor转成数组
                if args.gpu:
                    outputs_np = outputs.cpu().detach().numpy()
                    grades_label_np = grades_label.cpu().detach().numpy()
                # 累计同一患者每个3d块的测试结果和分类金标准
                if i == 0:
                    outputs_all = outputs_np
                    grades_label_all = grades_label_np
                else:
                    outputs_all = np.concatenate((outputs_all, outputs_np), axis=0)
                    grades_label_all = np.concatenate((grades_label_all, grades_label_np), axis=0)
                #print('Fold:{}, Epoch:{}, 【patient_num:{}】, num:{}/{}, loss:{}， probability:{}\t'.format(fold, epoch, val_patient[ind], i+1, len(patient), loss.item(), outputs_np[0][0]))

            # 每个人平均概率，以及所有人的平均概率、目标类别累计
            patient_probability = np.mean(outputs_all)
            #print('Fold:{}, Epoch:{}, 【patient_num:{}】, Loss:{}, probability:{}, label:{}\t'.format(fold, epoch, val_patient[ind], np.mean(loss_val), patient_probability, int(grades_label_np[0][0])))
            #print('------------------------------------------------------------')

            loss_val_all.append(np.mean(loss_val))
            all_probability.append(patient_probability)
            all_grade.append(int(grades_label_np[0][0]))
            if patient_probability < 0.5:
                all_pred_class.append(0)
            else:
                all_pred_class.append(1)

        # 计算ROC和AUC
        fpr, tpr, threshold = roc_curve(all_grade, all_probability)
        all_val_auc = auc(fpr, tpr)
        all_val_acc = accuracy_score(all_grade, all_pred_class)

        print('Fold:{}, VALIDATION Finished, loss:{}, Auc:{}, Acc:{}'.format(fold, np.mean(loss_val_all), all_val_auc, all_val_acc))
        print('------------------------------------------------------------')

        possibility_save(epoch, val_patient, [all_probability, all_grade, all_val_auc], fold)

        return loss_val_all, all_probability, all_grade, all_val_auc, all_val_acc

# test
def test_net(model, data, epoch, fold, TTA=False, TTA_times=None):
    model.eval()
    loss_val_all = []
    all_probability = []
    all_grade = []
    all_pred_class = []
    if TTA == False:
        print('***************************** Testing **************************************')
        for ind, patient in enumerate(data):  # 读每个人的所有3d块
            loss_val = []
            for i, b in enumerate(patient):  # 读每一3d块
                imgs = np.array([i[0] for i in b]).astype(
                    np.float32)  # b[0][0] i依次等于b[0]b[1],然后每个b都有一个image和mask,所以分别用i[0]i[1]代表提取，并设置好类型
                grades = np.array([i[2] for i in b]).astype(np.float32)
                #  弄grade一维数组，在有batch size不等于1的时候扩充一维，得到batch size * 1的数组，batch size等于1时，得到1.数组或者扩充后变成1 * 1的数组
                for bs in range(len(grades)):
                    if grades[bs].max() == 0:
                        grade = np.zeros(1)
                    elif grades[bs].max() == 1:
                        grade = np.ones(1)
                    # 验证测试时batch size=1，所以下面其实可以不用扩充一维。
                    if bs == 0:
                        grade_input = grade[np.newaxis, :]
                    else:
                        grade = grade[np.newaxis, :]
                        grade_input = np.concatenate((grade_input, grade), axis=0)
                # 转tensor并进行训练______________________________________________________________________________________________
                imgs = torch.from_numpy(imgs).unsqueeze(1)  # 转成tensor；并，对指定位置，unsqueeze是对数据维度进行扩充，squeeze是对数据维度进行压缩
                grades_label = torch.from_numpy(grade_input.astype(np.float32))  # .unsqueeze(1)
                # 数据用gpu跑
                if args.gpu:
                    imgs = imgs.cuda()
                    grades_label = grades_label.cuda()
                # 输入到模型________________________________________________________________________________________________________
                outputs = model(imgs)
                loss = criterion(outputs, grades_label)
                loss_val.append(loss.item())
                # tensor转成数组
                if args.gpu:
                    outputs_np = outputs.cpu().detach().numpy()
                    grades_label_np = grades_label.cpu().detach().numpy()
                # 累计同一患者每个3d块的测试结果和分类金标准
                if i == 0:
                    outputs_all = outputs_np
                    grades_label_all = grades_label_np
                else:
                    outputs_all = np.concatenate((outputs_all, outputs_np), axis=0)
                    grades_label_all = np.concatenate((grades_label_all, grades_label_np), axis=0)
                print('Fold:{}, Epoch:{}, 【patient_num:{}】, num:{}/{}, loss:{}， probability:{}\t'.format(fold, epoch, val_patient[ind], i+1, len(patient), loss.item(), outputs_np[0][0]))

            # 每个人平均概率，以及所有人的平均概率、目标类别累计
            patient_probability = np.mean(outputs_all)
            print('Fold:{}, Epoch:{}, 【patient_num:{}】, Loss:{}, probability:{}, label:{}\t'.format(fold, epoch, val_patient[ind], np.mean(loss_val), patient_probability, int(grades_label_np[0][0])))
            print('------------------------------------------------------------')

            loss_val_all.append(np.mean(loss_val))
            all_probability.append(patient_probability)
            all_grade.append(int(grades_label_np[0][0]))
            if patient_probability < 0.5:
                all_pred_class.append(0)
            else:
                all_pred_class.append(1)

        # 计算ROC和AUC
        fpr, tpr, threshold = roc_curve(all_grade, all_probability)
        all_val_auc = auc(fpr, tpr)
        all_val_acc = accuracy_score(all_grade, all_pred_class)

        print('Fold:{}, Test Finished, loss:{}, Auc:{}, Acc:{}'.format(fold, np.mean(loss_val_all), all_val_auc, all_val_acc))
        print('------------------------------------------------------------')

        possibility_save(epoch, val_patient, [all_probability, all_grade, all_val_auc], fold)

        return loss_val_all, all_probability, all_grade, all_val_auc, all_val_acc

    elif TTA:
        print('***************************** Testing TTA **************************************')
        for ind, patient in enumerate(data):  # 读每个人的所有3d块
            loss_val = []
            for i, b in enumerate(patient):  # 读每一3d块
                imgs = np.array([i[0] for i in b]).astype(np.float32)  # b[0][0] i依次等于b[0]b[1],然后每个b都有一个image和mask,所以分别用i[0]i[1]代表提取，并设置好类型
                grades = np.array([i[2] for i in b]).astype(np.float32)
                #  弄grade一维数组，在有batch size不等于1的时候扩充一维，得到batch size * 1的数组，batch size等于1时，得到1.数组或者扩充后变成1 * 1的数组
                for bs in range(len(grades)):
                    if grades[bs].max() == 0:
                        grade = np.zeros(1)
                    elif grades[bs].max() == 1:
                        grade = np.ones(1)
                    # 验证测试时batch size=1，所以下面其实可以不用扩充一维。
                    if bs == 0:
                        grade_input = grade[np.newaxis, :]
                    else:
                        grade = grade[np.newaxis, :]
                        grade_input = np.concatenate((grade_input, grade), axis=0)
                # 转tensor并进行训练______________________________________________________________________________________________

                ##TTA
                loss_aug = []
                for tta_times in range(TTA_times):
                    imgs_aug = TTA(imgs[0])
                    imgs_aug = imgs_aug[np.newaxis, :]
                    imgs_aug = torch.from_numpy(imgs_aug).unsqueeze(1)  # 转成tensor；并，对指定位置，unsqueeze是对数据维度进行扩充，squeeze是对数据维度进行压缩
                    grades_label = torch.from_numpy(grade_input.astype(np.float32))  # .unsqueeze(1)
                    # 数据用gpu跑
                    if args.gpu:
                        imgs_aug = imgs_aug.cuda()
                        grades_label = grades_label.cuda()
                    # 输入到模型________________________________________________________________________________________________________
                    outputs = model(imgs_aug)
                    loss = criterion(outputs, grades_label)
                    loss_aug.append(loss.item())
                    # tensor转成数组
                    if args.gpu:
                        outputs_np = outputs.cpu().detach().numpy()
                        grades_label_np = grades_label.cpu().detach().numpy()

                    if tta_times == 0:
                        outputs_all_p = outputs_np
                    else:
                        outputs_all_p = np.concatenate((outputs_all_p, outputs_np), axis=0)

                outputs_final_p = np.mean(outputs_all_p, axis=0)[np.newaxis, :]
                loss_aug_mean = sum(loss_aug) / (TTA_times)
                loss_val.append(loss_aug_mean)
                # 累计同一患者每个3d块的测试结果和分类金标准
                if i == 0:
                    outputs_all = outputs_final_p
                    grades_label_all = grades_label_np
                else:
                    outputs_all = np.concatenate((outputs_all, outputs_final_p), axis=0)
                    grades_label_all = np.concatenate((grades_label_all, grades_label_np), axis=0)

                print('Fold:{}, Epoch:{}, 【patient_num:{}】, num:{}/{}, loss:{}， probability:{}\t'.format(fold, epoch,val_patient[ind], i+1, len(patient), loss.item(), outputs_np[0][0]))

            # 每个人平均概率，以及所有人的平均概率、目标类别累计
            patient_probability = np.mean(outputs_all)
            print('Fold:{}, Epoch:{}, 【patient_num:{}】, Loss:{}, probability:{}, label:{}\t'.format(fold, epoch, val_patient[ind], np.mean(loss_val), patient_probability, int(grades_label_np[0][0])))
            print('------------------------------------------------------------')

            loss_val_all.append(np.mean(loss_val))
            all_probability.append(patient_probability)
            all_grade.append(int(grades_label_np[0][0]))
            if patient_probability < 0.5:
                all_pred_class.append(0)
            else:
                all_pred_class.append(1)

        # 计算ROC和AUC
        fpr, tpr, threshold = roc_curve(all_grade, all_probability)
        all_val_auc = auc(fpr, tpr)
        all_val_acc = accuracy_score(all_grade, all_pred_class)

        print('Fold:{}, Test TTA Finished, loss:{}, Auc:{}, Acc:{}'.format(fold, np.mean(loss_val_all), all_val_auc, all_val_acc))
        print('------------------------------------------------------------')

        possibility_save(epoch, val_patient, [all_probability, all_grade, all_val_auc], fold)

        return loss_val_all, all_probability, all_grade, all_val_auc, all_val_acc

def possibility_save(epoch, val_patient_num, content, fold):

    path = os.path.join(root_result, 'val_result', str(fold))
    os.makedirs(path, exist_ok=True)

    patient_list = []
    possibility_list = []
    label_list = []
    for i in range(len(content[0])):
        patient_list.append(str(val_patient_num[i]))
        possibility_list.append(str(content[0][i]))
        label_list.append(str(content[1][i]))
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("Sheet1")
    worksheet.write(0, 0, "patient")
    worksheet.write(0, 1, "possibility")
    worksheet.write(0, 2, "label")
    for ii in range(len(patient_list)):
        worksheet.write(ii+1, 0, patient_list[ii])
        worksheet.write(ii+1, 1, possibility_list[ii])
        worksheet.write(ii+1, 2, label_list[ii])
    xl_filename = 'epoch' + str(epoch) + '_auc_' + format(content[-1], '.4f') + '.xls'
    workbook.save(path + '/' + xl_filename)

def AUC_excel_save(path, epoch_list, auc_list):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("Sheet1")
    worksheet.write(0, 0, "epoch")
    worksheet.write(0, 1, "AUC")
    if len(epoch_list) != len(auc_list):
        print("excel write error")
    else:
        for i in range(len(epoch_list)):
            worksheet.write(epoch_list[i]+1, 0, str(epoch_list[i]))
            worksheet.write(epoch_list[i]+1, 1, str(auc_list[i]))
    workbook.save(path)

def LOSS_excel_save(path, epoch_list, loss_list):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("Sheet1")
    worksheet.write(0, 0, "epoch")
    worksheet.write(0, 1, "loss")
    if len(epoch_list) != len(loss_list):
        print("excel write error")
    else:
        for i in range(len(epoch_list)):
            worksheet.write(epoch_list[i]+1, 0, str(epoch_list[i]))
            worksheet.write(epoch_list[i]+1, 1, str(loss_list[i]))
    workbook.save(path)

def save_result(loss_train):
    # save train loss per iteration
    plt.figure(1)
    plt.plot(loss_train, 'r')
    plt.title('loss_train', fontsize='large', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.xlabel('Iteration', fontweight='bold')
    plt.savefig(os.path.join(root_result, 'loss_train.png'))
    plt.close()

def save_result_epochlast(loss_train, loss_val):
    # save train and validation loss per epoch
    plt.figure(1)
    plt.plot(loss_train, 'r')
    plt.plot(loss_val, 'b')
    plt.title('loss', fontsize='large', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.legend(['train', 'val'])
    plt.savefig(os.path.join(root_result, 'loss.png'))
    plt.close()

def save_test_result(root_result, ind, i, img, mask_true, mask_pred, mask_pred_01):
    # save test result
    path = os.path.join(root_result, 'test', str(ind))
    os.makedirs(path, exist_ok=True)
    plt.figure(1)
    plt.imshow(img[0][0],cmap='gray')
    plt.savefig(os.path.join(path, str(i+1) + '_img.png'))
    plt.close()
    plt.figure(2)
    plt.imshow(mask_true[0][0],cmap='gray')
    plt.savefig(os.path.join(path, str(i+1) + '_mask_true.png'))
    plt.close()
    plt.figure(3)
    plt.imshow(mask_pred[0][0],cmap='gray')
    plt.savefig(os.path.join(path, str(i+1) + '_mask_pred.png'))
    plt.close()
    plt.figure(4)
    plt.imshow(mask_pred_01[0][0],cmap='gray')
    plt.savefig(os.path.join(path, str(i+1) + '_mask_pred_01.png'))
    plt.close()

if __name__ == "__main__":
    fold = 1
    # train_patient = os.listdir(root_path)
    # val_patient = os.listdir(test_root_path)
    train_patient = []
    val_patient = []
    # 读取数据
    for id in pkl_info.keys():
        if pkl_info[id]['Use'] == 'Y': #能用的数据
            if pkl_info[id]['Type'] == 'Train':
                train_patient.append(str(id))
            else:
                val_patient.append(str(id))

    # train_patient_aug = os.listdir(root_path_aug)

    print('train_patient:', train_patient)
    print('val_patient:', val_patient)
    print(len(train_patient), len(val_patient))
# load data --------------------------------------------------------------------------------------------------------
    '''
    train=True, three_d=False, 2d分类网络进行训练（读取2d训练数据）
    else：
       train=False, three_d=False, 2d分类网络进行测试（按患者为单位读取测试数据）
       else（three_d=True）：
            train=True, three_d=True, 3d分类网络进行训练（读取3d训练数据）
            train=False, three_d=True, 3d分类网络进行验证测试（读取3d验证测试数据）
    这里的验证测试在上面set_CV设置了是一样的。
    '''
    # 训练集数据
    train_data, train_num_list_slice = get_h5_perpatient_perslice(root_path, train_patient, train=True, three_d=True) #################

    # train_data_aug, train_num_list_slice_aug = get_h5_perpatient_perslice_aug(root_path_aug, train_patient_aug, augtimes=11, train=True, three_d=True)  #################
    # for nn in range(len(train_data_aug)):
    #     train_data.append(train_data_aug[nn])

    # 验证集数据
    #val_data, val_num_list_slice = get_h5_perpatient_perslice(test_root_path, val_patient, train=False,three_d=True)
    val_data, val_num_list_slice = get_h5_perpatient_perslice(root_path, val_patient, train=False, three_d=True)
# get 3d block -----------------------------------------------------------------------------------------------------
    '''train_block_all_per_patient - 每个患者一个list,每个list显示有几个3d block
       train_block_all - 显示所有患者共有多少个3d block
    '''
    train_block_all_per_patient, train_block_all = my_get_3d_block(train_data, args.block_z_slice, args.block_z_sample_slice)
    val_block_all_per_patient, val_block_all = my_get_3d_block(val_data, args.block_z_slice, args.block_z_sample_slice)
    print("all train block num:", len(train_block_all))
    # print(len(train_block_all[0]), len(train_block_all[0][0]), len(train_block_all[0][0][0]))

# train and val ----------------------------------------------------------------------------------------------------
    loss_train_epoch_mean = []
    loss_val_epoch_mean = []
    all_epoch_list = []
    all_val_auc_list = []
    all_val_acc_list = []

    if not os.path.exists(root_result):
        os.makedirs(root_result)

    for epoch in range(args.start_epoch, args.epochs):
        """实时扩增"""
        if train_aug:
            train_block_aug = train_block_all.copy()
            all_train_block_aug = []
            for augtimes in range(Augtimes):
                block_sum = []
                print("--------Augmentation "+str(augtimes+1)+"--------")
                for n_block in range(len(train_block_aug)):
                    imgs = train_block_aug[n_block][0][0]
                    mask = train_block_aug[n_block][0][1]
                    grade = train_block_aug[n_block][0][2]

                    imgs = data_augmentation(imgs)

                    a = list([imgs])
                    b = list([mask])
                    c = list([grade])
                    block = list(zip(a, b, c))  # 三者合并成一个list(在一起～)
                    block_sum.append(block)
                all_train_block_aug.extend(block_sum)  # 合并所有患者的block

            print("augmentation train block num:", len(all_train_block_aug))
            # print(len(all_train_block_aug[0]), len(all_train_block_aug[0][0]), len(all_train_block_aug[0][0][0]))

        all_epoch_list.append(epoch)
        # print('epoch:{}'.format(epoch))
        # lr_cur = lr_opt(args.lr, epoch)  # speed change   调整学习率，epoch小于60学习率逐渐减少，大于60学习率不变
        lr_cur = args.lr
        for param_group in optimizer.param_groups:  # optimizer.param_groups集合了优化器的各项参数。。。。动态修改学习率
            param_group['lr'] = lr_cur

        # train for one epoch_______________________________________________________________________________________
        if train_aug:
            loss_train_epoch = train_net(all_train_block_aug, model, criterion, optimizer, epoch, lr_cur, fold)
        else:
            loss_train_epoch = train_net(train_block_all, model, criterion, optimizer, epoch, lr_cur, fold)  # 一个epoch中每个batch的loss（列表）

        loss_train_epoch_mean.append(np.mean(loss_train_epoch))
        LOSS_excel_save(os.path.join(root_result, "train_loss_fold"+str(fold)+".xls"), all_epoch_list, loss_train_epoch_mean)

        # evaluate on validation set________________________________________________________________________________
        loss_val_all, all_probability_val, all_grade_val, auc_val, acc_val = eval_net(val_block_all_per_patient, model, epoch, fold, val=True)
        loss_val_epoch_mean.append(np.mean(loss_val_all))
        all_val_auc_list.append(auc_val)
        all_val_acc_list.append(acc_val)
        LOSS_excel_save(os.path.join(root_result, "val_loss_fold"+str(fold)+".xls"), all_epoch_list, loss_val_epoch_mean)
        AUC_excel_save(os.path.join(root_result, "val_auc_fold"+str(fold)+".xls"), all_epoch_list, all_val_auc_list)

        # save model________________________________________________________________________________________________
        root_model = os.path.join(model_save_dir, str(fold))
        os.makedirs(root_model, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(root_model, 'epoch_' + str(epoch) + '_model.pkl'))  # 保存网络结构的名字和参数

        # draw loss_____________________________________________________________________________________________
        ## train loss
        # x = all_epoch_list
        # y1 = loss_train_epoch_mean
        # y2 = loss_val_epoch_mean
        # plt.plot(x, y1, label='train loss')
        # plt.plot(x, y2, label='val loss')
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.title("Loss")
        # plt.savefig(os.path.join(root_result, "loss_fold"+str(fold)+".jpg"))
        # plt.close()


