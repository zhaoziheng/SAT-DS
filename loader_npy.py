import pandas as pd
# from typing import List
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import torch
import monai
from einops import repeat, rearrange, reduce
import cv2
from skimage import io
import os
import SimpleITK as sitk
from pathlib import Path
import argparse
from tqdm import tqdm
import nibabel as nib

# datum = {"image": "/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ACDC/database/testing/patient138/patient138_frame01.nii.gz", "mask": "/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ACDC/database/testing/patient138/patient138_frame01_gt.nii.gz", "label": ["left heart ventricle", "right heart ventricle", "myocardium", "heart ventricle"], "modality": "MRI", "dataset": "ACDC", "official_split": "test", "renorm_image": "/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/ACDC/renorm_image/0.npy", "chwd": [1, 295, 350, 28], "patch_path": ["/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/ACDC/renorm_image/0/0_288_0_288_0_28.npy", "/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/ACDC/renorm_image/0/0_288_62_350_0_28.npy", "/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/ACDC/renorm_image/0/7_295_0_288_0_28.npy", "/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/ACDC/renorm_image/0/7_295_62_350_0_28.npy"], "patch_y1y2_x1x2_z1z2": [[0, 288, 0, 288, 0, 28], [0, 288, 62, 350, 0, 28], [7, 295, 0, 288, 0, 28], [7, 295, 62, 350, 0, 28]], "renorm_segmentation_dir": "/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v4/ACDC/renorm_segmentation/0", "renorm_y1x1z1_y2x2z2": [[131, 145, 0, 188, 198, 28], [160, 108, 2, 227, 208, 22], [120, 130, 0, 198, 213, 28], [131, 108, 0, 227, 208, 28]], "patient_id": "patient138_frame01.nii.gz", "visible_label_idx": [0, 1, 2, 3], "roi_y1x1z1_y2x2z2": [120, 108, 0, 227, 213, 28]}

def npy_loader(datum):
    # load image
    dataset_name = datum['dataset']
    
    image_path = datum['renorm_image']
    seg_dir = datum['renorm_segmentation_dir']
    image = torch.tensor(np.load(image_path))   # 1 h w d

    # load mask
    _, h, w, d = datum['chwd']
    labels = datum['label'] # laryngeal cancer or hypopharyngeal cancer
    mask_paths = [f"{seg_dir}/{label}.npy" for label in labels] # /remote-home/share/SAM/processed_files/Challenge_4C2021/segmentation/27/laryngeal cancer or hypopharyngeal cancer.npy
    y1x1z1_y2x2z2_ls = datum['renorm_y1x1z1_y2x2z2']
    mc_mask = []
    for mask_path, y1x1z1_y2x2z2 in zip(mask_paths, y1x1z1_y2x2z2_ls):
        mask = torch.zeros((h, w, d), dtype=torch.bool)
        # not empty, load and embed non-empty cropped_volume
        if y1x1z1_y2x2z2 != False:
            y1, x1, z1, y2, x2, z2 = y1x1z1_y2x2z2
            mask[y1:y2, x1:x2, z1:z2] = torch.tensor(np.load(mask_path))
        mc_mask.append(mask)
    mc_mask = torch.stack(mc_mask, dim=0)   # n h w d
    
    return image.float(), mc_mask.float(), datum['label'], datum['modality'], 'segmentation', image_path, seg_dir


def checksample(root, path2jsonl, sample_idx=0):
    """
    root: dir to save visualization files
    path2jsonl: path to the jsonl file
    sample_idx: choose a sample (by index) from the jsonl file
    """
    
    # 模拟读取
    with open(path2jsonl, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    batch = npy_loader(data[sample_idx])
    img_tensor, mc_mask, text_ls, modality, mask_type, image_path, mask_path = batch
    
    """   
    print(mc_mask.shape)
    for i in range(12):
        print(torch.sum(torch.where(mc_mask[i]==1, 1.0, 0.0))+torch.sum(torch.where(mc_mask[i]==0, 1.0, 0.0)))
    exit()
    """
    
    # 检查数据
    dataset_name = data[sample_idx]['dataset']
    assert torch.sum(torch.where(mc_mask==0, 1, 0)).item() + torch.sum(torch.where(mc_mask==1, 1, 0)).item() == mc_mask.shape[0]*mc_mask.shape[1]*mc_mask.shape[2]*mc_mask.shape[3]
    print('* Dataset %s has %d samples *'%(dataset_name, len(data)))
    print('* image path * : ', image_path)
    print('* mask path * : ', mask_path)
    print('* modality * : ', modality)
    print('* labels * : ', text_ls)
    print('* img_tensor.shape * : ', img_tensor.shape)  # [c h w d]
    print('* img_tensor.dtype * : ', img_tensor.dtype)
    print('* mc_mask.shape * : ', mc_mask.shape)    # [c h w d]
    print('* mc_mask.dtype * : ', mc_mask.dtype)
    print('* sum(mc_mask) * : ', torch.sum(mc_mask))
    
    mc_mask = mc_mask.numpy()
    img_tensor = img_tensor.numpy()
    if mc_mask.shape[-1] > 0:
        # 3D按nifiti存
        results = np.zeros((img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])) # hwd
        for j, label in enumerate(text_ls):            
            results += mc_mask[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
            Path(f'{root}/visualization_files/{dataset_name}/(npy_loader)sample_{sample_idx}/segmentations').mkdir(exist_ok=True, parents=True)
            # 每个label单独一个nii.gz
            segobj = nib.nifti2.Nifti1Image(mc_mask[j, :, :, :], np.eye(4))
            nib.save(segobj, f'{root}/visualization_files/{dataset_name}/(npy_loader)sample_{sample_idx}/segmentations/{label}.nii.gz')
        segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
        nib.save(segobj, f'{root}/visualization_files/{dataset_name}/(npy_loader)sample_{sample_idx}/seg.nii.gz')
        
        imgobj = nib.nifti2.Nifti1Image(img_tensor[0], np.eye(4))   # hwd
        nib.save(imgobj, f'{root}/visualization_files/{dataset_name}/(npy_loader)sample_{sample_idx}/img.nii.gz')

    # 按slice存
    for slice_idx in tqdm(range(mc_mask.shape[-1])):
        Path(f'{root}/visualization_files/%s/(npy_loader)sample_%d/slice_%d'%(dataset_name, sample_idx, slice_idx)).mkdir(parents=True, exist_ok=True)
        Path(f'{root}/visualization_files/%s/(npy_loader)sample_%d/image_series'%(dataset_name, sample_idx)).mkdir(parents=True, exist_ok=True)
        img = rearrange(img_tensor[:, :, :, slice_idx], 'c h w -> h w c') # [H, W, C]
        cv2.imwrite(f'{root}/visualization_files/%s/(npy_loader)sample_%d/slice_%d/img.jpg'%(dataset_name, sample_idx, slice_idx), img*255.0)
        cv2.imwrite(f'{root}/visualization_files/%s/(npy_loader)sample_%d/image_series/slice_%d.jpg'%(dataset_name, sample_idx, slice_idx), img*255.0)
        for label_idx, text in tqdm(enumerate(text_ls)):
            msk = mc_mask[label_idx, :, :, slice_idx]
            if np.sum(msk) > 0:
                """
                # the bbox
                non_zero_coordinates = np.nonzero(msk) # ([...], [...])
                y1, x1 = np.min(non_zero_coordinates[0]).item(), np.min(non_zero_coordinates[1]).item()
                y2, x2 = np.max(non_zero_coordinates[0]).item(), np.max(non_zero_coordinates[1]).item()
                print('slice no.%d, label no.%d : %s, [x1, y1, x2, y2] : [%d, %d, %d, %d]'%(slice_idx, label_idx, text, x1, y1, x2, y2))
                """
                print('slice no.%d, label no.%d : %s'%(slice_idx, label_idx, text))
                cv2.imwrite(f'{root}/visualization_files/%s/(npy_loader)sample_%d/slice_%d/%d_%s_msk.jpg'%(dataset_name, sample_idx,  slice_idx, label_idx, text), msk*255.0)
                if img.shape[2] == 1:
                    img = repeat(img, 'h w c -> h w (c r)', r=3)
                overlap = repeat(msk, 'h w -> h w c', c=3) # colorful mask H, W, C
                
                img = np.float32(img)
                overlap = np.float32(overlap)
                overlap = cv2.add(img*255.0, overlap*255.0)
                cv2.imwrite(f'{root}/visualization_files/%s/(npy_loader)sample_%d/slice_%d/%d_%s_seg.jpg'%(dataset_name, sample_idx,  slice_idx, label_idx, text), overlap)

def checkdataset(root, path2jsonl):
    """
    root: dir to save visualization files
    path2jsonl: path to the jsonl file
    sample_idx: choose a sample from the jsonl file
    """
    import traceback
    
    # 模拟读取
    with open(path2jsonl, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    dataset_with_error = set()
    
    for sample in tqdm(data, desc=f'checking each sample ... ...'):
        try:
            batch = npy_loader(sample)
            img_tensor, mc_mask, text_ls, modality, mask_type, image_path, mask_path = batch
            assert torch.sum(torch.where(mc_mask==0, 1, 0)).item() + torch.sum(torch.where(mc_mask==1, 1, 0)).item() == mc_mask.shape[0]*mc_mask.shape[1]*mc_mask.shape[2]*mc_mask.shape[3]
            assert img_tensor.shape[1] == mc_mask.shape[1] and img_tensor.shape[2] == mc_mask.shape[2] and img_tensor.shape[3] == mc_mask.shape[3], f'image {img_tensor.shape} != mask {mc_mask.shape} in {sample["image"]}'
            assert mc_mask.shape[0] == len(text_ls), f'mask {mc_mask.shape} != {len(text_ls)} labels in {sample["image"]}'
        except:
            if sample["dataset"] not in dataset_with_error:
                print(f'Meet Error in {sample["dataset"]}')
                dataset_with_error.add(sample["dataset"])
            
            info = traceback.format_exc()
            Path(f'{root}/visualization_files/{sample["dataset"]}').mkdir(exist_ok=True, parents=True)
            with open(f'{root}/visualization_files/{sample["dataset"]}/(npy_loader)load_error.text', 'a') as f:
                f.write(f'** {sample["dataset"]} ** {sample["patient_id"]} **\n')
                f.write(info)
                f.write('\n')
                f.write('\n')
                
    
if __name__ == '__main__':
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
               
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2jsonl', type=str)
    parser.add_argument('--i', type=int)
    config = parser.parse_args()

    root = 'NEED A PATH HERE'
    # path2jsonl = 'datasets/%s/%s.jsonl'%(config.dataset_name, config.dataset_name)
    if config.i is not None:
        checksample(root, config.path2jsonl, config.i)
    else:
        checkdataset(root, config.path2jsonl)