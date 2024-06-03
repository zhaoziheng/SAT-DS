import pandas as pd
from PIL import Image
from torchvision import transforms
import json
import numpy as np
import torch
import monai
from einops import repeat, rearrange, reduce
import os
import SimpleITK as sitk
from pathlib import Path
import argparse
from tqdm import tqdm
import nibabel as nib
import cv2

class Loader_Wrapper():
    def __init__(self):
        pass
    
    def ACDC(self, datum:dict) -> tuple:
        """
        'left ventricle cavity', 
        'right ventricle cavity', 
        'myocardium'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # merge labels
        mc_mask = [torch.where(mask==3, 1.0, 0.0), torch.where(mask==1, 1.0, 0.0), torch.where(mask==2, 1.0, 0.0)]
        
        # add heart ventricle
        labels = datum['label'][:3] + ['heart ventricle']
        ventricle_cavity = mc_mask[0] + mc_mask[1] # 'left heart ventricle' + 'right heart ventricle'
        mc_mask.append(ventricle_cavity)
        
        mask = torch.concat(mc_mask, dim=0) # [1, H, W, D] --> [C, H, W, D]
        mask = torch.where(mask>0.5, 1.0, 0.0)
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def HAN_Seg(self, datum:dict) -> tuple:
        """
        descriptive_labels = [
            'Arytenoid',    # 1
            'Brainstem',
            'Buccal Mucosa',
            'Left Carotid artery', 4
            'Right Carotid artery', 5
            'Cervical esophagus',
            'Left Cochlea', 7
            'Right Cochlea', 8
            'Cricopharyngeal inlet',
            'Left Anterior eyeball', 10
            'Right Anterior eyeball', 11
            'Left Posterior eyeball', 12
            'Right Posterior eyeball', 13
            'Left Lacrimal gland', 14
            'Right Lacrimal gland', 15
            'Larynx - glottis',
            'Larynx - supraglottic',
            'Lips',
            'Mandible',
            'Optic chiasm',
            'Left Optic nerve', 21
            'Right Optic nerve', 22
            'Oral cavity',
            'Left Parotid gland', 24
            'Right Parotid gland', 25
            'Pituitary gland',
            'Spinal cord',
            'Left Submandibular gland', 28 
            'Right Submandibular gland', 29
            'Thyroid'   # 30
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),#, ensure_channel_first=True),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:30]
        
        merge_mask = []
        for i in range(30):
            if 'Left' in datum['label'][i]: # Left xxx
                if datum['label'][i].replace('Left', 'Right') == datum['label'][i+1]:   # Right xxx
                    merge_mask.append(mask[i]+mask[i+1])
                    merged_label = datum['label'][i].replace('Left ', '')
                    labels.append(merged_label)
                        
        left_eye = mask[9] + mask[11]
        merge_mask.append(left_eye)
        labels.append('Left Eyeball')           
                                  
        right_eye = mask[10] + mask[12]
        merge_mask.append(right_eye)    
        labels.append('Right Eyeball')       

        eye = mask[9] + mask[10] + mask[11] + mask[12]
        merge_mask.append(eye)           
        labels.append('Eyeball')
        
        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
    
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def CHAOS_CT(self, datum:dict) -> tuple:
        """
        liver
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, datum['label'], datum['modality'], datum['image'], datum['mask']
    
    def CHAOS_MRI(self, datum:dict) -> tuple:
        """
        'liver', 
        'right kidney', 
        'left kidney', 
        'spleen'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:4]
        
        # NOTE: merge label
        kidney = mask[1] + mask[2]
        mask = torch.cat((mask, kidney.unsqueeze(0)), dim=0)
        labels.append("kidney")
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, datum['label'], datum['modality'], datum['image'], datum['mask']

    def AbdomenCT1K(self, datum:dict) -> tuple:
        """
        'liver', 
        'kidney', 
        'spleen', 
        'pancreas'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:4]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def ISLES2022(self, datum:dict) -> tuple:
        """
        'stroke'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label']

        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MRSpineSeg(self, datum:dict) -> tuple:
        """
        'sacrum',
        'lumbar vertebrae 5 (L5)',
        'lumbar vertebrae 4 (L4)', 
        'lumbar vertebrae 3 (L3)', 
        'lumbar vertebrae 2 (L2)', 
        'lumbar vertebrae 1 (L1)',
        'thoracic vertebrae 12 (T12)',
        'thoracic vertebrae 11 (T11)', 
        'thoracic vertebrae 10 (T10)', 
        'thoracic vertebrae 9 (T9)',
        'intervertebral discs between lumbar vertebrae 5 (L5) and sacrum',
        'intervertebral discs between lumbar vertebrae 4 (L4) and lumbar vertebrae 5 (L5)', 
        'intervertebral discs between lumbar vertebrae 3 (L3) and lumbar vertebrae 4 (L4)',
        'intervertebral discs between lumbar vertebrae 2 (L2) and lumbar vertebrae 3 (L3)', 
        'intervertebral discs between lumbar vertebrae 1 (L1) and lumbar vertebrae 2 (L2)', 
        'intervertebral discs between thoracic vertebrae 12 (T12) and lumbar vertebrae 1 (L1)',
        'intervertebral discs between thoracic vertebrae 11 (T11) and thoracic vertebrae 12 (T12)', 
        'intervertebral discs between thoracic vertebrae 10 (T10) and thoracic vertebrae 11 (T11)', 
        'intervertebral discs between thoracic vertebrae 9 (T9) and thoracic vertebrae 10 (T10)'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="ASR", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        mc_masks = []
        labels = datum['label'][:19]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
            
        # merge new labels
        lumbar = torch.zeros_like(mc_masks[0])
        for i in range(1, 6):   # lumbar vertebrae 5 (L5) ~ lumbar vertebrae 5 (L1)
            lumbar += mc_masks[i]
        mc_masks.append(lumbar)
        labels.append('lumbar vertebrae')
            
        thoracic = torch.zeros_like(mc_masks[0])
        for i in range(6, 10):  # thoracic vertebrae 12 (T12) ~ thoracic vertebrae 12 (T9)
            thoracic += mc_masks[i]
        mc_masks.append(thoracic)
        labels.append('thoracic vertebrae')
            
        intervertebral = torch.zeros_like(mc_masks[0])
        for i in range(10, 19): # intervertebral discs between xxx and xxx
            intervertebral += mc_masks[i]
        mc_masks.append(intervertebral)
        labels.append('intervertebral discs')
        
        vertebrae = torch.zeros_like(mc_masks[0])
        for i in range(0, 10):
            vertebrae += mc_masks[i]
        mc_masks.append(vertebrae)
        labels.append('vertebrae')
 
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def LUNA16(self, datum:dict) -> tuple:
        """
        'left lung', 
        'right lung', 
        'trachea'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:3]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+3), 1.0, 0.0)    # 3 left lung, 4 right lung, 5 trachea
            mc_masks.append(binary_mask)    # [1, H, W, D]
            
        mc_masks.append(mc_masks[0]+mc_masks[1])
        labels.append('lung')
        
        mask = torch.concat(mc_masks, dim=0)

        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Cardiac(self, datum:dict) -> tuple:
        """
        left atrium
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]

        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Liver(self, datum:dict) -> tuple:\
        """
        'liver', 
        'liver tumor'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        # 0 is liver, 1 is liver tumor, should be included in liver
        mask[0] += mask[1]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Hippocampus(self, datum:dict) -> tuple:
        """
        'Anterior Hippocampus', 
        'Posterior Hippocampus'
        """
        
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        # Merge label
        mc_masks.append(mc_masks[0] + mc_masks[1])
        labels.append('Hippocampus')
        
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Prostate(self, datum:dict) -> tuple:
        """
        'transition zone of prostate', 
        'peripheral zone of prostate'
        """
        mod2channel = {"T2":0, "ADC":1}
        tmp = datum['image'].split('/')
        mod = tmp[-1]
        channel = mod2channel[mod]
        img_path = '/'.join(tmp[:-1])
        
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
                #monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':img_path, 'label':datum['mask']})
        img = dictionary['image'][channel, :, :, :] # [H, W, D]
        mask = dictionary['label'] # [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        mc_masks.append(mc_masks[0]+mc_masks[1]) 
        labels.append('prostate')
        
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        
        img = repeat(img, 'h w d -> c h w d', c=1)  # [C, H, W, D]
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Lung(self, datum:dict) -> tuple:
        """
        lung tumor
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label']
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Pancreas(self, datum:dict) -> tuple:
        """
        'pancreas', 
        'pancreas tumor'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        mask[0] += mask[1]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_HepaticVessel(self, datum:dict) -> tuple:
        """
        'liver vessel', 
        'liver tumor'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Spleen(self, datum:dict) -> tuple:
        """
        'spleen'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Colon(self, datum:dict) -> tuple:
        """
        'colon cancer'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def SKI10(self, datum:dict) -> tuple:
        """
        'femur bone', 
        'femur cartilage', 
        'tibia bone', 
        'tibia cartilage'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']   # 1 h w d
        mask = dictionary['label']  # 1 h w d
        
        labels = datum['label'][:4]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def SLIVER07(self, datum:dict) -> tuple:
        """
        liver
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']   # 1 h w d
        mask = dictionary['label']  # 1 h w d
        
        labels = datum['label'][:1]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def PROMISE12(self, datum:dict) -> tuple:
        """
        prostate
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label']
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def BrainPTM(self, datum:dict) -> tuple:
        """
        'Left Optic Radiation', 
        'Right Optic Radiation', 
        'Left Corticospinal Tract', 
        'Right Corticospinal Tract', 
        'Brain'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        labels = datum['label'][:5]
        
        # 'Left Optic Radiation' + 'Right Optic Radiation'
        # 'Left Corticospinal Tract' + 'Right Corticospinal Tract'
        optic_radiation = mask[0] + mask[1]
        corticospinal_tract = mask[2] + mask[3]
        optic_radiation = repeat(optic_radiation, 'h w d -> c h w d', c=1)
        corticospinal_tract = repeat(corticospinal_tract, 'h w d -> c h w d', c=1)
        mask = torch.concat((mask, optic_radiation, corticospinal_tract), dim=0)
        
        labels.append('Optic Radiation')
        labels.append('Corticospinal Tract')
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def WMH_Segmentation_Challenge(self, datum:dict) -> tuple:
        """
        'white matter hyperintensities'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label']
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def WORD(self, datum:dict) -> tuple:
        """
        labels = [
            "liver",
            "spleen",
            "left kidney",  # 3
            "right kidney", # 4
            "stomach",
            "gallbladder",
            "esophagus",
            "pancreas", # 8
            "duodenum",
            "colon",    # 10
            "intestine",
            "adrenal",  # 12
            "rectum",
            "urinary bladder",
            "head of left femur", # 15
            "head of right femur" # 16
            "kidney" = "left kidney"+"right kidney"
            "head of femur" = "head of left femur"+"head of right femur" # 18
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:16]
        
        mc_masks = []
        for i, label in enumerate(datum['label'][:16]):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
            
        # merge label
        mc_masks.append(mc_masks[2]+mc_masks[3])
        labels.append("kidney")
        
        mc_masks.append(mc_masks[14]+mc_masks[15])
        labels.append("head of femur")  
          
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def TotalSegmentator_Organs(self, datum:dict) -> tuple:
        """
        'spleen',
        'right kidney',
        'left kidney',
        'gallbladder',
        'liver',
        'stomach',
        'pancreas',
        'right adrenal gland',
        'left adrenal gland',
        'left lung upper lobe',
        'left lung lower lobe',
        'right lung upper lobe',
        'right lung middle lobe',
        'right lung lower lobe',
        'esophagus',
        'trachea',
        'thyroid gland',
        'small bowel',
        'duodenum',
        'colon',
        'urinary bladder',
        'prostate',
        'left kidney cyst',
        'right kidney cyst',
        'kidney',
        'lung upper lobe',
        'lung lower lobe',
        'adrenal gland',
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:20]
        
        # Merge Label
        merge_mask = []
        
        for i in range(len(labels)):
            if 'left' in labels[i]: # xxx left
                if labels[i].replace('left', 'right') == labels[i+1]:   # xxx right
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('left ', '')
                    labels.append(merged_label)   # xxx
            if 'right' in labels[i]: # right xxx
                if labels[i].replace('right', 'left') == labels[i+1]:   # left xxx
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('right ', '')
                    labels.append(merged_label)   # xxx

        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def TotalSegmentator_Vertebrae(self, datum:dict) -> tuple:
        """
        'sacrum',
        'vertebrae S1',
        'lumbar vertebrae 5 (L5)',
        'lumbar vertebrae 4 (L4)',
        'lumbar vertebrae 3 (L3)',
        'lumbar vertebrae 2 (L2)',
        'lumbar vertebrae 1 (L1)',
        'thoracic vertebrae 12 (T12)',
        'thoracic vertebrae 11 (T11)',
        'thoracic vertebrae 10 (T10)',
        'thoracic vertebrae 9 (T9)',
        'thoracic vertebrae 8 (T8)',
        'thoracic vertebrae 7 (T7)',
        'thoracic vertebrae 6 (T6)',
        'thoracic vertebrae 5 (T5)',
        'thoracic vertebrae 4 (T4)',
        'thoracic vertebrae 3 (T3)',
        'thoracic vertebrae 2 (T2)',
        'thoracic vertebrae 1 (T1)',
        'cervical vertebrae 7 (C7)',
        'cervical vertebrae 6 (C6)',
        'cervical vertebrae 5 (C5)',
        'cervical vertebrae 4 (C4)',
        'cervical vertebrae 3 (C3)',
        'cervical vertebrae 2 (C2)',
        'cervical vertebrae 1 (C1)',
        'lumbar vertebrae',
        'cervical vertebrae',
        'thoracic vertebrae'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:25]
        
        # NOTE: Merge Label
        merge_mask = []
        
        cervical_vertebrae = torch.zeros((mask.shape[1], mask.shape[2], mask.shape[3]))
        lumbar_vertebrae = torch.zeros((mask.shape[1], mask.shape[2], mask.shape[3]))
        thoracic_vertebrae = torch.zeros((mask.shape[1], mask.shape[2], mask.shape[3]))
        vertebrae = torch.zeros((mask.shape[1], mask.shape[2], mask.shape[3]))
        
        for i in range(len(labels)):
            
            if 'cervical vertebrae' in labels[i]:
                cervical_vertebrae += mask[i]

            if 'lumbar vertebrae' in labels[i]:
                lumbar_vertebrae += mask[i]
                
            if 'thoracic vertebrae' in labels[i]:
                thoracic_vertebrae += mask[i]
                
            vertebrae += mask[i]
            
        merge_mask.append(cervical_vertebrae)        
        labels.append('cervical vertebrae')
        
        merge_mask.append(lumbar_vertebrae)    
        labels.append('lumbar vertebrae')
        
        merge_mask.append(thoracic_vertebrae)    
        labels.append('thoracic vertebrae')
        
        merge_mask.append(vertebrae)    
        labels.append('vertebrae')

        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def TotalSegmentator_Cardiac(self, datum:dict) -> tuple:
        """
        'heart',
        'aorta',
        'pulmonary artery',
        'brachiocephalic trunk',
        'right subclavian artery',
        'left subclavian artery',
        'right common carotid artery',
        'left common carotid artery',
        'left brachiocephalic vein',
        'right brachiocephalic vein',
        'left atrial appendage',
        'superior vena cava',
        'inferior vena cava',
        'portal vein and splenic vein',
        'left iliac artery',
        'right iliac artery',
        'left iliac vena',
        'right iliac vena',
        'iliac artery',
        'iliac vena',
        'left heart atrium',
        'right heart atrium',
        'heart myocardium',
        'left heart ventricle',
        'right heart ventricle',
        'heart atrium',
        'heart ventricle'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:13]
        
        # Merge Label
        merge_mask = []
        
        for i in range(len(labels)):
            if 'left' in labels[i]: # left xxx
                if labels[i].replace('left', 'right') == labels[i+1]:   # right xxx
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('left ', '')
                    labels.append(merged_label)   # xxx
            if 'right' in labels[i]: # right xxx
                if labels[i].replace('right', 'left') == labels[i+1]:   # left xxx
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('right ', '')
                    labels.append(merged_label)   # xxx

        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def TotalSegmentator_Muscles(self, datum:dict) -> tuple:
        """
        'left humerus',
        'right humerus',
        'left scapula',
        'right scapula',
        'left clavicula',
        'right clavicula',
        'left femur',
        'right femur',
        'left hip',
        'right hip',
        'spinal cord',
        'left gluteus maximus',
        'right gluteus maximus',
        'left gluteus medius',
        'right gluteus medius',
        'left gluteus minimus',
        'right gluteus minimus',
        'left autochthon',
        'right autochthon',
        'left iliopsoas',
        'right iliopsoas',
        'brain',
        'skull',
        'clavicula',
        'femur',
        'gluteus maximus',
        'gluteus medius',
        'gluteus minimus',
        'hip',
        'humerus',
        'iliopsoas',
        'scapula'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:21]
        
        # Merge Label
        merge_mask = []
        
        for i in range(len(labels)):
            if 'left' in labels[i]:
                if labels[i].replace('left', 'right') == labels[i+1]:
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('left ', '')
                    labels.append(merged_label)   # xxx
            if 'right' in labels[i]: # right xxx
                if labels[i].replace('right', 'left') == labels[i+1]:   # left xxx
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('right ', '')
                    labels.append(merged_label)   # xxx

        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def TotalSegmentator_Ribs(self, datum:dict) -> tuple:
        """
        'left rib 1',
        'left rib 2',
        'left rib 3',
        'left rib 4',
        'left rib 5',
        'left rib 6',
        'left rib 7',
        'left rib 8',
        'left rib 9',
        'left rib 10',
        'left rib 11',
        'left rib 12',
        'right rib 1',
        'right rib 2',
        'right rib 3',
        'right rib 4',
        'right rib 5',
        'right rib 6',
        'right rib 7',
        'right rib 8',
        'right rib 9',
        'right rib 10',
        'right rib 11',
        'right rib 12',
        'sternum',
        'costal cartilages',
        'rib',
        'left rib',
        'right rib',
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:24]
        
        # Merge Label
        merge_mask = []
        
        # rib 1~10
        for i, lab in enumerate(labels):
            if 'left ' in lab and lab.replace('left ', 'right ') in labels:
               right_idx = labels.index(lab.replace('left', 'right'))
               merge_mask.append(mask[i]+mask[right_idx])
               labels.append(lab.replace('left ', ''))
            if 'right' in labels[i]: # right xxx
                if labels[i].replace('right', 'left') == labels[i+1]:   # left xxx
                    merge_mask.append(mask[i]+mask[i+1]) 
                    merged_label = labels[i].replace('right ', '')
                    labels.append(merged_label)   # xxx
        
        # left rib, right rib, rib
        
        rib_right = torch.zeros((mask.shape[1], mask.shape[2], mask.shape[3]))
        rib_left = torch.zeros((mask.shape[1], mask.shape[2], mask.shape[3]))
        
        for i in range(len(labels)):
            if 'right rib' in labels[i]:
                rib_right += mask[i]
            if 'left rib' in labels[i]:
                rib_left += mask[i] 
                
        merge_mask.append(rib_right)
        labels.append("right rib")
        
        merge_mask.append(rib_left)
        labels.append("left rib")
        
        rib = rib_left + rib_right
        merge_mask.append(rib)
        labels.append("rib")

        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def TotalSegmentator_v2(self, datum:dict) -> tuple:
        """
        'left auricle of heart',    # 0
        'brachiocephalic trunk',
        'left brachiocephalic vein',
        'right brachiocephalic vein',
        'left common carotid artery',
        'right common carotid artery',
        'costal cartilage',
        'heart',
        'left kidney cyst',
        'right kidney cyst',
        'prostate',
        'pulmonary vein',
        'skull',
        'spinal cord',
        'sternum',
        'left subclavian artery',
        'right subclavian artery',
        'superior vena cava',
        'thyroid gland',
        'sacral vertebrae 1 (S1)'   # 19
        """
        
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:20]
        
        # NOTE: Merge Label
        merge_mask = []
        
        for i in range(len(labels)):
            
            if 'left' in labels[i]: # Left xxx
                if labels[i].replace('left', 'right') == labels[i+1]:   # Right xxx
                    merge_mask.append(mask[i]+mask[i+1])
                    merged_label = labels[i].replace('left ', '').strip()
                    labels.append(merged_label)
            elif 'right' in labels[i]: # Right xxx
                if labels[i].replace('right', 'left') == labels[i+1]:   # Right xxx
                    merge_mask.append(mask[i]+mask[i+1])
                    merged_label = labels[i].replace('right ', '').strip()
                    labels.append(merged_label)

        merge_mask = torch.stack(merge_mask, dim=0) # NHWD
        mask = torch.concat((mask, merge_mask), dim=0)
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def FLARE22(self, datum:dict) -> tuple:
        """
        'Liver',    # 1
        'Right kidney', # 2
        'Spleen',
        'Pancreas',
        'Aorta',
        'Inferior Vena Cava',
        'Right Adrenal Gland', # 7
        'Left Adrenal Gland', # 8
        'Gallbladder',
        'Esophagus',
        'Stomach',
        'Duodenum',
        'Left kidney'   # 13
        'Kidney' = 'Left kidney' + 'Right kidney'
        'Adrenal Gland' = 'Right Adrenal Gland' + 'Left Adrenal Gland'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:13]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[12])
        labels.append("Kidney")
        mc_masks.append(mc_masks[6]+mc_masks[7])
        labels.append("Adrenal Gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def NSCLC(self, datum:dict) -> tuple:
        """
        'thoracic cavity', 
        'effusion'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"])
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        
        labels = datum['label'][:2]
        
        # dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [2, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def COVID19(self, datum:dict) -> tuple:
        """
        'left lung', 
        'right lung', 
        'COVID-19 infection',
        'lung' = 'left lung' + 'right lung' + 'COVID-19 infection'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"])
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        
        # dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:3]
        
        mc_masks = []   # 'left lung', 'right lung', 'COVID-19 infection' --> 'lung'
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append('lung')
            
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def Brain_Atlas(self, datum:dict) -> tuple:
        """
        'left hippocampus',    # 1
        'right hippocampus',
        'left amygdala',
        'right amygdala',
        'left anterior temporal lobe medial part',
        'right anterior temporal lobe medial part',
        'left anterior temporal lobe lateral part',
        'right anterior temporal lobe lateral part',
        'left parahippocampal and ambient gyrus',
        'right parahippocampal and ambient gyrus',  # 10
        'left superior temporal gyrus middle part',
        'right superior temporal gyrus middle part',
        'left middle and inferior temporal gyrus',
        'right middle and inferior temporal gyrus',
        'left fusiform gyrus',
        'right fusiform gyrus',
        'left cerebellum',
        'right cerebellum',
        'brainstem excluding substantia nigra',
        'right insula posterior long gyrus',    # 20
        'left insula posterior long gyrus',
        'right lateral remainder occipital lobe',
        'left lateral remainder occipital lobe',
        'right anterior cingulate gyrus',
        'left anterior cingulate gyrus',
        'right posterior cingulate gyrus',
        'left posterior cingulate gyrus',
        'right middle frontal gyrus',   # 28
        'left middle frontal gyrus',
        'right posterior temporal lobe',
        'left posterior temporal lobe',
        'right angular gyrus',  # 31
        'left angular gyrus',
        'right caudate nucleus',
        'left caudate nucleus',
        'right nucleus accumbens',
        'left nucleus accumbens',
        'right putamen',
        'left putamen',
        'right thalamus',
        'left thalamus',
        'right pallidum',
        'left pallidum',
        'corpus callosum',
        'left Lateral ventricle excluding temporal horn',   # 45
        'right Lateral ventricle excluding temporal horn',
        'left Lateral ventricle temporal horn',
        'right Lateral ventricle temporal horn',
        'Third ventricle',
        'right precentral gyrus',
        'left precentral gyrus',
        'right straight gyrus',
        'left straight gyrus',
        'right anterior orbital gyrus',
        'left anterior orbital gyrus',
        'right inferior frontal gyrus',
        'left inferior frontal gyrus',
        'right superior frontal gyrus',
        'left superior frontal gyrus',
        'right postcentral gyrus',
        'left postcentral gyrus',
        'right superior parietal gyrus',
        'left superior parietal gyrus',
        'right lingual gyrus',
        'left lingual gyrus',
        'right cuneus',
        'left cuneus',
        'right medial orbital gyrus',
        'left medial orbital gyrus',
        'right lateral orbital gyrus',
        'left lateral orbital gyrus',
        'right posterior orbital gyrus',
        'left posterior orbital gyrus',
        'right substantia nigra',
        'left substantia nigra',
        'right subgenual frontal cortex',
        'left subgenual frontal cortex',
        'right subcallosal area',
        'left subcallosal area',
        'right pre-subgenual frontal cortex',
        'left pre-subgenual frontal cortex',
        'right superior temporal gyrus anterior part',
        'left superior temporal gyrus anterior part',
        'right supramarginal gyrus',
        'left supramarginal gyrus',
        'right insula anterior short gyrus',
        'left insula anterior short gyrus',
        'right insula middle short gyrus',
        'left insula middle short gyrus',
        'right insula posterior short gyrus',
        'left insula posterior short gyrus',
        'right insula anterior inferior cortex',
        'left insula anterior inferior cortex',
        'right insula anterior long gyrus',
        'left insula anterior long gyrus',      # 95
        'lateral ventricle' = 'left lateral ventricle excluding temporal horn'+...+'right lateral ventricle temporal horn',
        'insula' = 'right insula posterior long gyrus'+...+'left insula anterior long gyrus',
        'parietal lobe' = 'angular gyrus'+...
        'frontal lobe' = 'right middle frontal gyrus'+...
        'basal ganglia' = 'right caudate nucleus'+...
        'cingulate gyrus' = ...
        'brainstem' = ...
        'temporal lobe' = ...
        'thalamus' = ...
        'cerebellum' = ...
        'occipital lobe' = ...
        'hippocampus' = ...
        'amygdala' = ...
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"])
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        
        # dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:95]
        
        mc_masks = [] # 95 --> 108
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        lateral_ventricle = mc_masks[44] + mc_masks[45] + mc_masks[46] + mc_masks[47]
        mc_masks.append(lateral_ventricle)
        labels.append('lateral ventricle')
        
        insula = mc_masks[19] + mc_masks[20] + mc_masks[85] + mc_masks[86] + mc_masks[87] + mc_masks[88] + mc_masks[89] + mc_masks[90] + mc_masks[91] + mc_masks[92] + mc_masks[93] + mc_masks[94]
        mc_masks.append(insula)
        labels.append('insula')
        
        parietal_lobe = mc_masks[31] + mc_masks[32] + mc_masks[59] + mc_masks[60] + mc_masks[61] + mc_masks[62] + mc_masks[83] + mc_masks[84]
        mc_masks.append(parietal_lobe)
        labels.append('parietal lobe')
        
        frontal_lobe = mc_masks[27] + mc_masks[28] + mc_masks[49] + mc_masks[50] + mc_masks[51] + mc_masks[52] + mc_masks[53] + mc_masks[54] + mc_masks[55] + mc_masks[56] + mc_masks[57] + mc_masks[58] + mc_masks[67] + mc_masks[68] + mc_masks[69] + mc_masks[70] + mc_masks[71] + mc_masks[72] + mc_masks[75] + mc_masks[76] + mc_masks[79] + mc_masks[80]
        mc_masks.append(frontal_lobe)
        labels.append('frontal lobe')
        
        basal_ganglia = mc_masks[33] + mc_masks[34] + mc_masks[35] + mc_masks[36] + mc_masks[37] + mc_masks[38] + mc_masks[41] + mc_masks[42]
        mc_masks.append(basal_ganglia)
        labels.append('basal ganglia')
        
        cingulate_gyrus = mc_masks[23] + mc_masks[24] + mc_masks[25] + mc_masks[26] + mc_masks[77] + mc_masks[78]
        mc_masks.append(cingulate_gyrus)
        labels.append('cingulate gyrus')
        
        brainstem = mc_masks[18] + mc_masks[73] + mc_masks[74]
        mc_masks.append(brainstem)
        labels.append('brainstem')
        
        temporal_lobe = mc_masks[4] + mc_masks[5] + mc_masks[6] + mc_masks[7] + mc_masks[8] + mc_masks[9] + mc_masks[10] + mc_masks[11] + mc_masks[12] + mc_masks[13] + mc_masks[14] + mc_masks[15] + mc_masks[29] + mc_masks[30] + mc_masks[81] + mc_masks[82]
        mc_masks.append(temporal_lobe)
        labels.append('temporal lobe')
        
        thalamus = mc_masks[39] + mc_masks[40]
        mc_masks.append(thalamus)
        labels.append('thalamus')
        
        cerebellum = mc_masks[16] + mc_masks[17]
        mc_masks.append(cerebellum)
        labels.append('cerebellum')
        
        occipital_lobe = mc_masks[21] + mc_masks[22] + mc_masks[63] + mc_masks[64] + mc_masks[65] + mc_masks[66]
        mc_masks.append(occipital_lobe)
        labels.append('occipital lobe')
        
        hippocampus = mc_masks[1] + mc_masks[0]
        mc_masks.append(hippocampus)
        labels.append('hippocampus')
        
        amygdala = mc_masks[3] + mc_masks[2]
        mc_masks.append(amygdala)
        labels.append('amygdala')
            
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def Couinaud_Liver(self, datum:dict) -> tuple:
        """
        'caudate lobe',     0
        'left lateral superior segment of liver',   1
        'Left lateral inferior segment of liver',   2
        'left medial segment of liver', 3
        'right anterior inferior segment of liver', 4
        'right posterior inferior segment of liver',    5
        'right posterior superior segment of liver',    6
        'right anterior superior segment of liver'  7
        'left lobe of liver' = 1 + 2 + 3
        'right lobe of liver' = 4 + 5 + 6 + 7
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:8]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[1]+mc_masks[2]+mc_masks[3])
        labels.append('left lobe of liver')
        
        mc_masks.append(mc_masks[4]+mc_masks[5]+mc_masks[6]+mc_masks[7])
        labels.append('right lobe of liver')
        
        mask = torch.cat(mc_masks, dim=0) # [11, H, W, D]

        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_CT(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("kidney")
        mc_masks.append(mc_masks[10]+mc_masks[11])
        labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_MRI(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("kidney")
        mc_masks.append(mc_masks[10]+mc_masks[11])
        labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def BTCV(self, datum:dict) -> tuple:
        """
        labels = [
            "spleen",
            "right kidney",
            "left kidney",
            "gallbladder",
            "esophagus",
            "liver",
            "stomach",
            "aorta",
            "inferior vena cava",
            "portal vein and splenic vein",
            "pancreas",
            "right adrenal gland",
            "left adrenal gland",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:13]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("kidney")
        mc_masks.append(mc_masks[11]+mc_masks[12])
        labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def CT_ORG(self, datum:dict) -> tuple:
        """
        labels = [
            "liver",
            "urinary bladder",
            "lung",
            "kidney",
            "bone",
            "brain",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:6]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def FeTA2022(self, datum:dict) -> tuple:
        """
        labels = [
            "external cerebrospinal fluid",
            "grey matter",
            "white matter",
            "brain ventricle",
            "cerebellum",
            "deep grey matter",
            "brainstem",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:7]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def ToothFairy(self, datum:dict) -> tuple:
        """
        labels = [
            "inferior alveolar nerve",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="IPL", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def Hecktor2022(self, datum:dict) -> tuple:
        """
        labels = [
            "head and neck tumor",
            "lymph node",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        mc_masks = []
        labels = datum['label'][:2]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'PET' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def PARSE2022(self, datum:dict) -> tuple:
        """
        labels = [
            "pulmonary artery",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:1]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def SegTHOR(self, datum:dict) -> tuple:
        """
        labels = [
            "esophagus",
            "heart",
            "trachea",
            "aorta",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:4]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MM_WHS_CT(self, datum:dict) -> tuple:
        """
        labels = [
            "myocardium",
            "left heart atrium",
            "left heart ventricle",
            "right heart atrium",
            "right heart ventricle",
            "heart ascending aorta",
            "pulmonary artery",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:7]
        intensity = [205, 420, 500, 550, 600, 820, 850]
        
        mc_masks = []
        for label, value in zip(labels, intensity):
            binary_mask = torch.where(mask==value, 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[3])
        labels.append("heart atrium")
        mc_masks.append(mc_masks[2]+mc_masks[4])
        labels.append("heart ventricle")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MM_WHS_MRI(self, datum:dict) -> tuple:
        """
        labels = [
            "myocardium",
            "left heart atrium",
            "left heart ventricle",
            "right heart atrium",
            "right heart ventricle",
            "heart ascending aorta",
            "pulmonary artery",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:7]
        intensity = [205, 420, 500, 550, 600, 820, 850]
        
        mc_masks = []
        for label, value in zip(labels, intensity):
            binary_mask = torch.where(mask==value, 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        # merge label
        mc_masks.append(mc_masks[1]+mc_masks[3])
        labels.append("heart atrium")
        mc_masks.append(mc_masks[2]+mc_masks[4])
        labels.append("heart ventricle")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def CMRxMotion(self, datum:dict) -> tuple:
        """
        labels = [
            "left heart ventricle",
            "myocardium",
            "right heart ventricle",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        # merge label
        mc_masks.append(mc_masks[0]+mc_masks[2])
        labels.append("heart ventricle")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def LAScarQS22_Task1(self, datum:dict) -> tuple:
        """
        labels = [
            "left heart atrium",
            "left heart atrium scar"
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        mc_masks = []
        labels = datum['label'][:2]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[0] += mc_masks[1]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def LAScarQS22_Task2(self, datum:dict) -> tuple:
        """
        labels = [
            "left heart atrium",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def ATLASR2(self, datum:dict) -> tuple:
        """
        labels = [
            "stroke",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def CrossMoDA2021(self, datum:dict) -> tuple:
        """
        labels = [
            "vestibular schwannoma",
            "cochlea",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        labels = datum['label'][:2]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MyoPS2020(self, datum:dict) -> tuple:
        """
        labels = [
            "right heart ventricle",
            "myocardial scar",
            "myocardial edema",
            "myocardium",
            "left heart ventricle",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                # monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 

        labels = datum['label'][:5]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[3] += mc_masks[1] + mc_masks[2]

        mc_masks.append(mc_masks[0]+mc_masks[4])
        labels.append("heart ventricle")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]

        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def Instance22(self, datum:dict) -> tuple:
        """
        labels = [
            "intracranial hemorrhage",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        labels = datum['label'][:1]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def KiTS23(self, datum:dict) -> tuple:
        """
        labels = [
            "kidney",
            "kidney tumor",
            "kidney cyst",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[0] += mc_masks[1]
        mc_masks[0] += mc_masks[2]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def ATLAS(self, datum:dict) -> tuple:
        """
        labels = [
            "liver",
            "liver tumor",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:2]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[0] += mc_masks[1]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def KiPA22(self, datum:dict) -> tuple:
        """
        labels = [
            "renal vein",
            "kidney",
            "renal artery",
            "kidney tumor",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:4]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[1] += mc_masks[3]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def BraTS2023_GLI(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_MEN(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_MET(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_PED(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BraTS2023_SSA(self, datum:dict) -> tuple:
        '''
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:3]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mc_masks.append(mc_masks[0]+mc_masks[1]+mc_masks[2])
        labels.append("brain tumor")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def BTCV_Cervix(self, datum:dict) -> tuple:
        '''
        labels = [
            "urinary bladder",
            "uterus",
            "rectum",
            "small bowel",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="LPS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:4]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def SEGA(self, datum:dict) -> tuple:
        '''
        labels = [
            "aorta",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def Pancreas_CT(self, datum:dict) -> tuple:
        '''
        labels = [
            "pancreas",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def FUMPE(self, datum:dict) -> tuple:
        '''
        labels = [
            "pulmonary embolism",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def VerSe(self, datum:dict) -> tuple:
        '''
        labels = [
            "cervical vertebrae 1 (c1)",
            "cervical vertebrae 2 (c2)",
            "cervical vertebrae 3 (c3)",
            "cervical vertebrae 4 (c4)",
            "cervical vertebrae 5 (c5)",
            "cervical vertebrae 6 (c6)",
            "cervical vertebrae 7 (c7)", # 6
            "thoracic vertebrae 1 (t1)",
            "thoracic vertebrae 2 (t2)",
            "thoracic vertebrae 3 (t3)",
            "thoracic vertebrae 4 (t4)",
            "thoracic vertebrae 5 (t5)",
            "thoracic vertebrae 6 (t6)",
            "thoracic vertebrae 7 (t7)",
            "thoracic vertebrae 8 (t8)",
            "thoracic vertebrae 9 (t9)",
            "thoracic vertebrae 10 (t10)",
            "thoracic vertebrae 11 (t11)",
            "thoracic vertebrae 12 (t12)", # 18
            "lumbar vertebrae 1 (l1)",
            "lumbar vertebrae 2 (l2)",
            "lumbar vertebrae 3 (l3)",
            "lumbar vertebrae 4 (l4)",
            "lumbar vertebrae 5 (l5)",
            "lumbar vertebrae 6 (l6)", # 24
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="ASR", keys=['image', 'label']),  # IPR
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:26]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        cervical = torch.zeros_like(mc_masks[0])
        for i in range(7):
            cervical += mc_masks[i]
        mc_masks.append(cervical)
        labels.append('cervical vertebrae')

        thoracic = torch.zeros_like(mc_masks[0])
        for i in range(7, 19):
            thoracic += mc_masks[i]
        thoracic += mc_masks[25]
        mc_masks.append(thoracic)
        labels.append('thoracic vertebrae')

        lumbar = torch.zeros_like(mc_masks[0])
        for i in range(19, 25):
            lumbar += mc_masks[i]
        mc_masks.append(lumbar)
        labels.append('lumbar vertebrae')

        vertebrae = torch.zeros_like(mc_masks[0])
        for i in range(26):
            vertebrae += mc_masks[i]
        mc_masks.append(vertebrae)
        labels.append('vertebrae')

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def PDDCA(self, datum:dict) -> tuple:
        '''
        labels = [
            "brainstem",
            "optic chiasm",
            "mandible",
            "left optic nerve",
            "right optic nerve",
            "left parotid gland",
            "right parotid gland",
            "left submandibular gland",
            "right submandibular gland",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:9]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mc_masks.append(mc_masks[3]+mc_masks[4])
        labels.append("optic nerve")
        
        mc_masks.append(mc_masks[5]+mc_masks[6])
        labels.append("parotid gland")
        
        mc_masks.append(mc_masks[7]+mc_masks[8])
        labels.append("submandibular gland")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def LNDb(self, datum:dict) -> tuple:
        '''
        labels = [
            "lung nodule",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def SegRap2023_Task1(self, datum:dict) -> tuple:
        '''
        labels = [
            "brain", 
            "brainstem", 
            "optic chiasm", 
            "left temporal lobe", 
            "right temporal lobe", 
            "left temporal lobe hippocampus overlap", 
            "right temporal lobe hippocampus overlap", 
            "left hippocampus", 
            "right hippocampus", 
            "left eyeball", 
            "right eyeball", 
            "left lens", 
            "right lens", 
            "left optic nerve", 
            "right optic nerve", 
            "left middle ear", 
            "right middle ear", 
            "left internal auditory canal", 
            "right internal auditory canal", 
            "left middle ear tympanic cavity overlap", 
            "right middle ear tympanic cavity overlap", 
            "left tympanic cavity", 
            "right tympanic cavity", 
            "left middle ear vestibule semicircular canal overlap", 
            "right middle ear vestibule semicircular canal overlap", 
            "left vestibule semicircular canal", 
            "right vestibule semicircular canal", 
            "left cochlea", 
            "right cochlea", 
            "left middle ear eustachian tube bone overlap", 
            "right middle ear eustachian tube bone overlap", 
            "left eustachian tube bone", 
            "right eustachian tube bone", 
            "pituitary gland", 
            "oral cavity", 
            "left mandible", 
            "right mandible", 
            "left submandibular gland", 
            "right submandibular gland", 
            "left parotid gland", 
            "right parotid gland", 
            "left mastoid process", 
            "right mastoid process", 
            "left temporomandibular joint", 
            "right temporomandibular joint", 
            "spinal cord", 
            "esophagus", 
            "larynx", 
            "larynx glottis", 
            "larynx supraglottis", 
            "larynx pharynx constrictor muscle overlap", 
            "pharynx constrictor muscle", 
            "thyroid", 
            "trachea", 
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:54]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        mc_masks[0] = mc_masks[0] + mc_masks[1] + mc_masks[2] + mc_masks[3] + mc_masks[4] + mc_masks[5] + mc_masks[6] + mc_masks[7] + mc_masks[8]
        mc_masks[3] = mc_masks[3] + mc_masks[5]
        mc_masks[4] = mc_masks[4] + mc_masks[6]
        mc_masks[7] = mc_masks[7] + mc_masks[5]
        mc_masks[8] = mc_masks[8] + mc_masks[6]
        mc_masks[9] = mc_masks[9] + mc_masks[11]
        mc_masks[10] = mc_masks[10] + mc_masks[12]
        mc_masks[15] = mc_masks[17] + mc_masks[15] + mc_masks[19] + mc_masks[23] + mc_masks[27] + mc_masks[29]
        mc_masks[16] = mc_masks[18] + mc_masks[16] + mc_masks[20] + mc_masks[24] + mc_masks[28] + mc_masks[30]
        mc_masks[21] = mc_masks[21] + mc_masks[19]
        mc_masks[22] = mc_masks[22] + mc_masks[20]
        mc_masks[25] = mc_masks[25] + mc_masks[23]
        mc_masks[26] = mc_masks[26] + mc_masks[24]
        mc_masks[31] = mc_masks[31] + mc_masks[29]
        mc_masks[32] = mc_masks[32] + mc_masks[30]
        mc_masks[47] = mc_masks[47] + mc_masks[48] + mc_masks[49] + mc_masks[50]
        mc_masks[51] = mc_masks[50] + mc_masks[51]

        mc_masks.append(mc_masks[3]+mc_masks[4])
        labels.append("temporal lobe")
        
        mc_masks.append(mc_masks[7]+mc_masks[8])
        labels.append("hippocampus")
        
        mc_masks.append(mc_masks[9]+mc_masks[10])
        labels.append("eyeball")
        
        mc_masks.append(mc_masks[11]+mc_masks[12])
        labels.append("lens")
        
        mc_masks.append(mc_masks[13]+mc_masks[14])
        labels.append("optic nerve")
        
        mc_masks.append(mc_masks[15]+mc_masks[16])
        labels.append("middle ear")
        
        mc_masks.append(mc_masks[17]+mc_masks[18])
        labels.append("internal auditory canal")
        
        mc_masks.append(mc_masks[21]+mc_masks[22])
        labels.append("tympanic cavity")
        
        mc_masks.append(mc_masks[25]+mc_masks[26])
        labels.append("vestibule semicircular canal")
        
        mc_masks.append(mc_masks[27]+mc_masks[28])
        labels.append("cochlea")
        
        mc_masks.append(mc_masks[31]+mc_masks[32])
        labels.append("eustachian tube bone")
        
        mc_masks.append(mc_masks[35]+mc_masks[36])
        labels.append("mandible")
        
        mc_masks.append(mc_masks[37]+mc_masks[38])
        labels.append("submandibular gland")
        
        mc_masks.append(mc_masks[39]+mc_masks[40])
        labels.append("parotid gland")
        
        mc_masks.append(mc_masks[41]+mc_masks[42])
        labels.append("mastoid process")
        
        mc_masks.append(mc_masks[43]+mc_masks[44])
        labels.append("temporomandibular joint")
        
        labels_to_del = [
            'left temporal lobe hippocampus overlap',
            'right temporal lobe hippocampus overlap',
            "left middle ear tympanic cavity overlap",
            "right middle ear tympanic cavity overlap",
            "left middle ear vestibule semicircular canal overlap",
            "right middle ear vestibule semicircular canal overlap",
            "left middle ear eustachian tube bone overlap",
            "right middle ear eustachian tube bone overlap",
            "larynx pharynx constrictor muscle overlap",
        ]
        
        for label in labels_to_del:
            i = labels.index(label)
            del mc_masks[i]
            del labels[i]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def SegRap2023_Task2(self, datum:dict) -> tuple:
        '''
        labels = [
            "nasopharyngeal tumor",
            "nasopharyngeal lymph node",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:2]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def CTPelvic1K(self, datum:dict) -> tuple:
        '''
        labels = [
            "sacrum",
            "left hip",
            "right hip",
            "lumbar vertebrae",
            ]
        '''
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        # 'cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor' + 'brain tumor'
        mc_masks = []
        labels = datum['label'][:4]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        mc_masks.append(mc_masks[1]+mc_masks[2])
        labels.append("hip")

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def autoPET(self, datum:dict) -> tuple:
        """
        labels = [
            "tumor",
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        mc_masks = []
        labels = datum['label'][:1]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'PET')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def DAP_Atlas(self, datum:dict) -> tuple:
        """
        labels = [
            "muscle",   # 0 -> 1
            "fat",      # 1 -> 2
            "abdominal tissue",
            "mediastinal tissue",
            "esophagus",
            "stomach",
            "small bowel",
            "duodenum",
            "colon",    # 8 --> 9
            # "rectum", NOTE: Skip
            "gallbladder",  # 9 --> 11
            "liver",
            "pancreas",
            "left kidney",
            "right kidney",
            "urinary bladder",
            "gonad",
            "prostate",
            "uterocervix",
            "uterus",
            "left breast",
            "right breast",
            "spinal canal",
            "brain",
            "spleen",
            "left adrenal gland",
            "right adrenal gland",
            "left thyroid",
            "right thyroid",
            "thymus",
            "left gluteus maximus",
            "right gluteus maximus",
            "left gluteus medius",
            "right gluteus medius",
            "left gluteus minimus",
            "right gluteus minimus",
            "left iliopsoas",
            "right iliopsoas",
            "left autochthon",
            "right autochthon",
            "skin",
            "cervical vertebrae 1 (c1)",
            "cervical vertebrae 2 (c2)",
            "cervical vertebrae 3 (c3)",
            "cervical vertebrae 4 (c4)",
            "cervical vertebrae 5 (c5)",
            "cervical vertebrae 6 (c6)",
            "cervical vertebrae 7 (c7)",
            "thoracic vertebrae 1 (t1)",
            "thoracic vertebrae 2 (t2)",
            "thoracic vertebrae 3 (t3)",
            "thoracic vertebrae 4 (t4)",
            "thoracic vertebrae 5 (t5)",
            "thoracic vertebrae 6 (t6)",
            "thoracic vertebrae 7 (t7)",
            "thoracic vertebrae 8 (t8)",
            "thoracic vertebrae 9 (t9)",
            "thoracic vertebrae 10 (t10)",
            "thoracic vertebrae 11 (t11)",
            "thoracic vertebrae 12 (t12)",
            "lumbar vertebrae 1 (l1)",
            "lumbar vertebrae 2 (l2)",
            "lumbar vertebrae 3 (l3)",
            "lumbar vertebrae 4 (l4)",
            "lumbar vertebrae 5 (l5)",
            "left rib 1",
            "right rib 1",
            "left rib 2",
            "right rib 2",
            "left rib 3",
            "right rib 3",
            "left rib 4",
            "right rib 4",
            "left rib 5",
            "right rib 5",
            "left rib 6",
            "right rib 6",
            "left rib 7",
            "right rib 7",
            "left rib 8",
            "right rib 8",
            "left rib 9",
            "right rib 9",
            "left rib 10",
            "right rib 10",
            "left rib 11",
            "right rib 11",
            "left rib 12",
            "right rib 12",
            "rib cartilage",
            "sternum",
            "left clavicle",
            "right clavicle",
            "left scapula",
            "right scapula",
            "left humerus",
            "right humerus",
            "skull",
            "left hip",
            "right hip",
            "sacrum",
            "left femur",
            "right femur",
            "heart",
            "left heart atrium",
            "heart tissue",
            "right heart atrium",
            "myocardium",
            "left heart ventricle",
            "right heart ventricle",
            "left iliac artery",
            "right iliac artery",
            "aorta",
            "left iliac vena",
            "right iliac vena",
            "inferior vena cava",
            "portal vein and splenic vein",
            "celiac trunk",
            "left lung lower lobe",
            "left lung upper lobe",
            "right lung lower lobe",
            "right lung middle lobe",
            "right lung upper lobe",
            "bronchie",
            "trachea",
            "pulmonary artery",
            "left cheek",
            "right cheek",
            "left eyeball",
            "right eyeball",
            "nasal cavity",
            "right common carotid artery",  # 132
            "left common carotid artery",
            "manubrium of sternum",
            "right internal carotid artery",    # 135
            "left internal carotid artery",
            "right internal jugular vein",  # 137
            "left internal jugular vein",
            "brachiocephalic trunk",    # 139
            "right brachiocephalic vein",   # 140
            "left brachiocephalic vein",
            "right subclavian artery",  # 142
            "left subclavian artery"   
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        mc_masks = []
        labels = datum['label'][:143]
        for i, label in enumerate(labels):
            if i < 9:
                binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, 1 is unknown tissue, so plus 2 is the correct label
            else:
                binary_mask = torch.where(mask==(i+2), 1.0, 0.0)    # 0 is background, 1 is unknown tissue, so plus 2 is the correct label
            mc_masks.append(binary_mask)

        for i in range(len(labels)):
            if 'left' in labels[i]: # Left xxx
                if labels[i].replace('left', 'right') == labels[i+1]:   # Right xxx
                    mc_masks.append(mc_masks[i]+mc_masks[i+1])
                    merged_label = labels[i].replace('left ', '').strip()
                    labels.append(merged_label)
            elif 'right' in labels[i]: # Right xxx
                if labels[i].replace('right', 'left') == labels[i+1]:   # Right xxx
                    mc_masks.append(mc_masks[i]+mc_masks[i+1])
                    merged_label = labels[i].replace('right ', '').strip()
                    labels.append(merged_label)

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        #mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

def Normalization(torch_image, image_type):
    # rgb_list = ['rgb', 'photograph', 'laparoscopy', 'colonoscopy', 'microscopy', 'dermoscopy', 'fundus', 'fundus image']
    np_image = torch_image.numpy()
    if image_type.lower() == 'ct':
        lower_bound, upper_bound = -500, 1000
        np_image = np.clip(np_image, lower_bound, upper_bound)
        np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    else:
        lower_bound, upper_bound = np.percentile(np_image, 0.5), np.percentile(np_image, 99.5)
        np_image = np.clip(np_image, lower_bound, upper_bound)
        np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    return torch.tensor(np_image)


def checksample(visualization_dir, path2jsonl, sample_idx=0):
    """
    visualization_dir: dir to save visualization of data samples
    path2jsonl: path to the jsonl file
    sample_idx: choose a sample from the jsonl file
    """
    
    loader = Loader_Wrapper()
    
    # load
    with open(path2jsonl, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    func_name = data[sample_idx]['dataset']
    batch = getattr(loader, func_name)(data[sample_idx])
    img_tensor, mc_mask, text_ls, modality, image_path, mask_path = batch
    
    # check
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
            Path(f'{visualization_dir}/{dataset_name}/(loader_v4)sample_{sample_idx}/segmentations').mkdir(exist_ok=True, parents=True)
            # 每个label单独一个nii.gz
            segobj = nib.nifti2.Nifti1Image(mc_mask[j, :, :, :], np.eye(4))
            nib.save(segobj, f'{visualization_dir}/{dataset_name}/(loader_v4)sample_{sample_idx}/segmentations/{label}.nii.gz')
        segobj = nib.nifti2.Nifti1Image(results, np.eye(4))
        nib.save(segobj, f'{visualization_dir}/{dataset_name}/(loader_v4)sample_{sample_idx}/seg.nii.gz')
        
        imgobj = nib.nifti2.Nifti1Image(img_tensor[0], np.eye(4))   # hwd
        nib.save(imgobj, f'{visualization_dir}/{dataset_name}/(loader_v4)sample_{sample_idx}/img.nii.gz')

    # 按slice存
    for slice_idx in tqdm(range(mc_mask.shape[-1])):
        Path(f'{visualization_dir}/%s/(loader_v4)sample_%d/slice_%d'%(dataset_name, sample_idx, slice_idx)).mkdir(parents=True, exist_ok=True)
        Path(f'{visualization_dir}/%s/(loader_v4)sample_%d/image_series'%(dataset_name, sample_idx)).mkdir(parents=True, exist_ok=True)
        img = rearrange(img_tensor[:, :, :, slice_idx], 'c h w -> h w c') # [H, W, C]
        cv2.imwrite(f'{visualization_dir}/%s/(loader_v4)sample_%d/slice_%d/img.jpg'%(dataset_name, sample_idx, slice_idx), img*255.0)
        cv2.imwrite(f'{visualization_dir}/%s/(loader_v4)sample_%d/image_series/slice_%d.jpg'%(dataset_name, sample_idx, slice_idx), img*255.0)
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
                cv2.imwrite(f'{visualization_dir}/%s/(loader_v4)sample_%d/slice_%d/%d_%s_msk.jpg'%(dataset_name, sample_idx,  slice_idx, label_idx, text), msk*255.0)
                if img.shape[2] == 1:
                    img = repeat(img, 'h w c -> h w (c r)', r=3)
                overlap = repeat(msk, 'h w -> h w c', c=3) # colorful mask H, W, C
                img = np.float32(img)
                overlap = np.float32(overlap)
                overlap = cv2.add(img*255.0, overlap*255.0)
                cv2.imwrite(f'{visualization_dir}/%s/(loader_v4)sample_%d/slice_%d/%d_%s_seg.jpg'%(dataset_name, sample_idx,  slice_idx, label_idx, text), overlap)

def checkdataset(error_output, path2jsonl):
    """
    path2jsonl: path to the jsonl file
    error_output: file to record the error message
    """
    import traceback

    loader = Loader_Wrapper()
    
    # 模拟读取
    with open(path2jsonl, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    dataset_with_error = set()

    for sample in tqdm(data, desc=f'checking each sample ... ...'):
        func_name = sample['dataset']
        try:
            batch = getattr(loader, func_name)(sample)
            img_tensor, mc_mask, text_ls, modality, image_path, mask_path = batch
            assert torch.sum(torch.where(mc_mask==0, 1, 0)).item() + torch.sum(torch.where(mc_mask==1, 1, 0)).item() == mc_mask.shape[0]*mc_mask.shape[1]*mc_mask.shape[2]*mc_mask.shape[3]
            assert img_tensor.shape[1] == mc_mask.shape[1] and img_tensor.shape[2] == mc_mask.shape[2] and img_tensor.shape[3] == mc_mask.shape[3], f'image {img_tensor.shape} != mask {mc_mask.shape} in {sample["image"]}'
            assert mc_mask.shape[0] == len(text_ls), f'mask {mc_mask.shape} != {len(text_ls)} labels in {sample["image"]}'
        except:
            if sample["dataset"] not in dataset_with_error:
                print(f'Meet Error in {sample["dataset"]}')
                dataset_with_error.add(sample["dataset"])
            
            info = traceback.format_exc()
            with open(error_output, 'a') as f:
                f.write(f'** {sample["dataset"]} ** {sample["patient_id"]} **\n')
                f.write(info)
                f.write('\n')
                f.write('\n')

if __name__ == '__main__':               
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualization_dir', type=str)
    parser.add_argument('--error_output', type=str)
    parser.add_argument('--path2jsonl', type=str)
    parser.add_argument('--i', type=int)
    config = parser.parse_args()

    if config.i is not None:
        checksample(config.path2jsonl, config.i)
    else:
        checkdataset(config.error_output, config.path2jsonl)