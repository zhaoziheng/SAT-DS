from email.mime import image
import os
from turtle import left
from PIL import Image
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import torch
import cv2
import monai
import math
import SimpleITK as sitk
import shutil
import argparse
import dicom2nifti
import nibabel as nib
import nrrd
from einops import rearrange, repeat, reduce
import scipy.io

class Process_Wrapper():
    def __init__(self, jsonl_dir):
        self.jsonl_dir = jsonl_dir

    def preprocess_ACDC(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ACDC/database'):
        """
        https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
        
        masks_path: root_path/training/patientxxx/patientxxx_framexx_gt.nii.gz
        images_path: root_path/training/patientxxx/patientxxx_framexx.nii.gz
        """
        
        dataset = 'ACDC'
        labels = ['left ventricle cavity', 'right ventricle cavity', 'myocardium']
        modality = 'MRI'
        data = []
        mask_ls = Path(root_path).glob('**/*_gt.nii.gz')
        for mask in mask_ls:
            image = str(mask).replace('_gt.nii.gz', '.nii.gz')
            split = 'train' if 'train' in image else 'test'
            patent_id = image.split('/')[-1]    # patientxxx_framexx.nii.gz
            data.append({
                        'image':image,
                        'mask':str(mask),
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split':split,
                        'patent_id':patent_id
                        })
            
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_HAN_Seg(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/HAN_Seg/HaN-Seg/set_1'):
        """
        https://han-seg2023.grand-challenge.org
        
        images_path: root_path/case_xx/case_xx_IMG_CT.nrrd
        masks_path: root_path/case_xx/case_xx_OAR_xxx.seg.nrrd
        
        3D. segmentation on CT scans of 42 patients, annotating 30 organs-at-risks, labels are translated to OAR description
        each OAR annotation is stored in an individual file, we preprocess them to dervie masks with multi-channels
        
        nrrd -> nii.gz
        
        NOTE: The dataset also contains MRI-T1 scans, but the shape of them DO NOT match that of the seg annotation
        """
        dataset = 'HAN_Seg'
        descriptive_labels = [
            'Arytenoid',
            'Brainstem',
            'Buccal Mucosa',
            'Left Carotid artery',
            'Right Carotid artery',
            'Cervical esophagus',
            'Left Cochlea',
            'Right Cochlea',
            'Cricopharyngeal inlet',
            'Left Anterior eyeball',
            'Right Anterior eyeball',
            'Left Posterior eyeball',
            'Right Posterior eyeball',
            'Left Lacrimal gland',
            'Right Lacrimal gland',
            'Larynx - glottis',
            'Larynx - supraglottic',
            'Lips',
            'Mandible',
            'Optic chiasm',
            'Left Optic nerve',
            'Right Optic nerve',
            'Oral cavity',
            'Left Parotid gland',
            'Right Parotid gland',
            'Pituitary gland',
            'Spinal cord',
            'Left Submandibular gland',
            'Right Submandibular gland',
            'Thyroid'
            ]
        
        labels =[
                'Arytenoid',
                'Brainstem',
                'BuccalMucosa',
                'A_Carotid_L',
                'A_Carotid_R',
                'Esophagus_S',
                'Cochlea_L',
                'Cochlea_R',
                'Cricopharyngeus',
                'Eye_AL',
                'Eye_AR',
                'Eye_PL',
                'Eye_PR',
                'Glnd_Lacrimal_L',
                'Glnd_Lacrimal_R',
                'Glottis',
                'Larynx_SG',
                'Lips',
                'Bone_Mandible',
                'OpticChiasm',
                'OpticNrv_L',
                'OpticNrv_R',
                'Cavity_Oral',
                'Parotid_L',
                'Parotid_R',
                'Pituitary',
                'SpinalCord',
                'Glnd_Submand_L',
                'Glnd_Submand_R',
                'Glnd_Thyroid'
                ]

        modality = 'CT'
        data = []
        
        for dir in tqdm(os.listdir(root_path)):
            
            if dir == 'case_19':    # case_19 unmatched image and mask
                continue
            
            if 'case' in dir:
                mask_paths = {}
                for f in os.listdir(os.path.join(root_path, dir)):
                    if f.endswith('IMG_CT.nii.gz'):
                        image_path = os.path.join(root_path, dir, f)
                    elif 'OAR' in f:
                        label = f.split('.seg.nrrd')[0].split('OAR_')[-1]
                        mask_paths[label] = os.path.join(root_path, dir, f)  
                
                # convert nrrd to nii.gz       
                # img = sitk.ReadImage(image_path)
                # sitk.WriteImage(img, image_path.replace('nrrd', 'nii.gz'))    
                image_nib = nib.load(image_path.replace('nrrd', 'nii.gz'))
                image_np = image_nib.get_fdata()

                mask = []
                for label in tqdm(labels):
                    if label not in mask_paths:
                        mask.append(np.zeros_like(image_np))
                    else:
                        np_mask, _ = nrrd.read(mask_paths[label])
                        mask.append(np_mask)    # [1, 1024, 1024, 197]
                mask = np.stack(mask, axis=0) # NHWD
                
                nii_mask = nib.Nifti1Image(mask, image_nib.affine, image_nib.header)
                nib.save(nii_mask, os.path.join(root_path, dir, 'label.nii.gz'))
                print(f"save to {os.path.join(root_path, dir, 'label.nii.gz')}")

                data.append({
                            'image':image_path.replace('nrrd', 'nii.gz'),
                            'mask':os.path.join(root_path, dir, 'label.nii.gz'),
                            'label':descriptive_labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split':'unknown',
                            'patient_id':dir,
                            })
                
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_CHAOS_CT(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/CHAOS/Train_Sets/CT'):
        """
        https://chaos.grand-challenge.org
        
        Segmentation of liver from computed tomography (CT) data sets.
        
        .dcm -> nii.gz; 
        
        20 training scans.
        
        images_path: root_path/xx/DICOM_anon/xxxx.dcm
        masks_path: root_path/xx/Ground/liver_GT_xxxx.png
        """
        dataset = 'CHAOS_CT'
        labels = ['liver']
        data = []
        monai_loader = monai.transforms.LoadImage(image_only=True)
        
        for case in tqdm(os.listdir(root_path)):  # 1
            # convert image to nifti
            dicom2nifti.convert_directory(os.path.join(root_path, case, 'DICOM_anon'), os.path.join(root_path, case), compression=True)
            print(f"save {os.path.join(root_path, case, 'DICOM_anon')} to {os.path.join(root_path, case)}")
            for file_name in os.listdir(os.path.join(root_path, case)):
                if '.nii.gz' in file_name and file_name != 'label.nii.gz':
                    image_path = os.path.join(root_path, case, file_name)
                    
            image_nib = nib.load(image_path)

            masks = []
            mask_paths = os.listdir(os.path.join(root_path, case, 'Ground'))
            mask_paths = sorted(mask_paths)
            for p in mask_paths:
                mask = monai_loader(os.path.join(root_path, case, 'Ground', p))
                masks.append(mask)
            masks = np.stack(masks, axis=0) # (D, 256, 256)
            masks = np.flip(np.flip(masks, axis=0), axis=-1)
            masks = repeat(masks, 'd h w -> c h w d', c=1)
            
            # convert mask to nifti as well
            concat_mask_nib = nib.Nifti1Image(masks, image_nib.affine, image_nib.header)
            nib.save(concat_mask_nib, os.path.join(root_path, case, 'label.nii.gz'))
            print(f"save to {os.path.join(root_path, case, 'label.nii.gz')}")
            
            data.append({
                        'image':image_path,
                        'mask':os.path.join(root_path, case, 'label.nii.gz'),
                        'label':labels,
                        'modality':'CT',
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id':case,
                        })
           
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_CHAOS_MRI(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/CHAOS/Train_Sets/MR'):
        """
        https://chaos.grand-challenge.org
        
        .dcm -> nii.gz; 
        
        40 MRI scans.
        
        T1_in_images_path: root_path/xx/T1DUAL/DICOM_anon/OutPhase/xxx.dcm
        T1_out_images_path: root_path/xx/T1DUAL/DICOM_anon/InPhase/xxx.dcm
        T1_masks_path: root_path/xx/T1DUAL/Ground/xxx.png
        
        T2_images_path: root_path/xx/T2SPIR/DICOM_anon/xxx.dcm
        T2_masks_path: root_path/xx/T2SPIR/Ground/xxx.png
        
        Segmentation of four abdominal organs (i.e. liver, spleen, right and left kidneys) 

        from T1-DUAL(inPhase and outPhase are registered. Therefore, their ground truth is the same.) and T2-SPIR MRI data.
        
        --Labeles of the four abdomen organs in the ground data are represented by four different pixel values ranges:
        Liver: 63 (55<<<70)
        Right kidney: 126 (110<<<135)
        Left kidney: 189 (175<<<200)
        Spleen: 252 (240<<<255)
        """
        
        def mri_relabel(mask):
            mask = mask.squeeze().float()   # [D, H, W]
            spleen = torch.where(mask>240, 1.0, 0.0).float()
            l_kidney = torch.where(mask>175, mask, torch.tensor(1000.)).float()
            l_kidney = torch.where(l_kidney<=200, 1.0, 0.0).float()
            r_kidney = torch.where(mask>110, mask, torch.tensor(1000.)).float()
            r_kidney = torch.where(r_kidney<=135, 1.0, 0.0).float()
            liver = torch.where(mask>55, mask, torch.tensor(1000.)).float()
            liver = torch.where(liver<=70, 1.0, 0.0).float()
            return torch.stack([liver, r_kidney, l_kidney, spleen], dim=1)  # [D, 4, H, W]
        
        dataset = 'CHAOS_MRI'
        labels = ['liver', 'right kidney', 'left kidney', 'spleen']
        data = []
        monai_loader = monai.transforms.LoadImage(image_only=True)
        
        for case in tqdm(os.listdir(root_path)):  # 1
            dicom2nifti.convert_directory(os.path.join(root_path, case, 'T1DUAL/DICOM_anon/OutPhase'), os.path.join(root_path, case, 'T1DUAL/DICOM_anon/OutPhase'), compression=True)
            print(f"save {os.path.join(root_path, case, 'T1DUAL/DICOM_anon/OutPhase')} to {os.path.join(root_path, case, 'T1DUAL/DICOM_anon/OutPhase')}")
            for file_name in os.listdir(os.path.join(root_path, case, 'T1DUAL/DICOM_anon/OutPhase')):
                if '.nii.gz' in file_name:
                    out_image_path = os.path.join(root_path, case, 'T1DUAL/DICOM_anon/OutPhase', file_name)
                    
            dicom2nifti.convert_directory(os.path.join(root_path, case, 'T1DUAL/DICOM_anon/InPhase'), os.path.join(root_path, case, 'T1DUAL/DICOM_anon/InPhase'), compression=True)
            print(f"save {os.path.join(root_path, case, 'T1DUAL/DICOM_anon/InPhase')} to {os.path.join(root_path, case, 'T1DUAL/DICOM_anon/InPhase')}")
            for file_name in os.listdir(os.path.join(root_path, case, 'T1DUAL/DICOM_anon/InPhase')):
                if '.nii.gz' in file_name:
                    in_image_path = os.path.join(root_path, case, 'T1DUAL/DICOM_anon/InPhase', file_name)
                    
            image_nib = nib.load(in_image_path)

            mask_paths = os.listdir(os.path.join(root_path, case, 'T1DUAL/Ground'))
            mask_paths = sorted(mask_paths)
            masks = []
            for p in mask_paths:
                mask = monai_loader(os.path.join(root_path, case, 'T1DUAL/Ground', p))
                masks.append(mask)
            masks = np.stack(masks, axis=0) # (D, 256, 256)
            masks = mri_relabel(torch.tensor(masks)).numpy() # (D, 4, 256, 256)
            masks = rearrange(masks, 'd c h w -> c h w d')
            masks = np.flip(masks, axis=2)
            
            concat_mask_nib = nib.Nifti1Image(masks, image_nib.affine, image_nib.header)
            nib.save(concat_mask_nib, os.path.join(root_path, case, 't1_label.nii.gz'))
            print(f"save to {os.path.join(root_path, case, 't1_label.nii.gz')}")

            data.append({
                        'image':out_image_path,
                        'mask':os.path.join(root_path, case, 't1_label.nii.gz'),
                        'label':labels,
                        'modality':'MRI T1',
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id':case,
                        })
            data.append({
                        'image':in_image_path,
                        'mask':os.path.join(root_path, case, 't1_label.nii.gz'),
                        'label':labels,
                        'modality':'MRI T1',
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id':case,
                        })
            
            dicom2nifti.convert_directory(os.path.join(root_path, case, 'T2SPIR/DICOM_anon'), os.path.join(root_path, case, 'T2SPIR/DICOM_anon'), compression=True)
            print(f"save {os.path.join(root_path, case, 'T2SPIR/DICOM_anon')} to {os.path.join(root_path, case, 'T2SPIR/DICOM_anon')}")
            for file_name in os.listdir(os.path.join(root_path, case, 'T2SPIR/DICOM_anon')):
                if '.nii.gz' in file_name:
                    t2_image_path = os.path.join(root_path, case, 'T2SPIR/DICOM_anon', file_name)
                    
            image_nib = nib.load(t2_image_path)

            mask_paths = os.listdir(os.path.join(root_path, case, 'T2SPIR/Ground'))
            mask_paths = sorted(mask_paths)
            masks = []
            for p in mask_paths:
                mask = monai_loader(os.path.join(root_path, case, 'T2SPIR/Ground', p))
                masks.append(mask)
            masks = np.stack(masks, axis=0) # (D, 256, 256)
            masks = mri_relabel(torch.tensor(masks)).numpy() # (D, 4, 256, 256)
            masks = rearrange(masks, 'd c h w -> c h w d')
            masks = np.flip(masks, axis=2)
            
            concat_mask_nib = nib.Nifti1Image(masks, image_nib.affine, image_nib.header)
            nib.save(concat_mask_nib, os.path.join(root_path, case, 't2_label.nii.gz'))
            print(f"save to {os.path.join(root_path, case, 't2_label.nii.gz')}")

            data.append({
                        'image':t2_image_path,
                        'mask':os.path.join(root_path, case, 't2_label.nii.gz'),
                        'label':labels,
                        'modality':'MRI T2',
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
    
    def preprocess_AbdomenCT1K(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/AbdomenCT-1K'):
        """
        https://github.com/JunMa11/AbdomenCT-1K#50-cases-with-13-annotated-organs-download-zenodo
        
        .
        ├── Images
        │   ├── Case_00001_0000.nii.gz
        ... ...
        │   └── Case_01062_0000.nii.gz
        ├── Masks
        │   ├── Case_00001.nii.gz
        ... ...
        │   └── Case_01062.nii.gz
        """
        dataset = 'AbdomenCT1K'
        labels = ['liver', 'kidney', 'spleen', 'pancreas']
        modality = 'CT'
        data = []
        for case in tqdm(os.listdir(os.path.join(root_path, 'Masks'))):  # Case_00001.nii.gz
            mask_path = os.path.join(root_path, 'Masks', case)
            image_path = os.path.join(root_path, 'Images', case[:-7]+'_0000.nii.gz')
            data.append({
                        'image':image_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split':'unknown',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_ISLES2022(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ISLES22/ISLES'):
        """
        https://isles22.grand-challenge.org
        
        ISLES/dataset
        ├── derivatives
        |   ├── sub-strokecase0001
        |   |   └──ses-0001
        |   |       └── sub-strokecase0001_ses-0001_msk.nii.gz  (seg path)
        |   |
        |   ├── sub-strokecase0002
        |   |   └──ses-0002
        |   |       └── sub-strokecase0002_ses-0001_msk.nii.gz
        |   └── ... ...
        |
        └── rawdata
            ├── sub-strokecase0001
            |   └──ses-0001
            |       ├── sub-strokecase0001_ses-0001_adc.nii.gz  (img path)
            |       ├── sub-strokecase0001_ses-0001_dwi.nii.gz  (img path)
            |       └── sub-strokecase0001_ses-0001_flair.nii.gz    (not consistent with segmentation annotation)
            |
            ├── sub-strokecase0002
            |   └──ses-0001
            |       ├── sub-strokecase0001_ses-0001_adc.nii.gz
            |       ├── sub-strokecase0001_ses-0001_dwi.nii.gz
            |       └── sub-strokecase0001_ses-0001_flair.nii.gz
            └── ... ...
        """
        
        dataset = 'ISLES2022'
        labels = ['stroke']
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'dataset/rawdata')):
            if 'sub-strokecase' not in case:    # sub-strokecasexxx_ses-xxxx_adc.nii.gz
                continue
            
            for m, modality in zip(['adc', 'dwi'], ['MRI ADC', 'MRI DWI']):
                img_path = os.path.join(root_path, 'dataset/rawdata', case, 'ses-0001', case+'_ses-0001_'+m+'.nii.gz')
                msk_path = os.path.join(root_path, 'dataset/derivatives', case, 'ses-0001', case+'_ses-0001_msk.nii.gz')
                data.append({
                        'image':img_path,
                        'mask':msk_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split':'unknown',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')    
    
    def preprocess_MRSpineSeg(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MRSpineSeg_Challenge_SMU/train'):
        """
        https://www.spinesegmentation-challenge.com/?page_id=1162
        
        multi-class segmentation of vertebrae and intervertebral discs
        
        img paths: root_path/MR/Casexxx.nii.gz
        mask paths: root_path/Mask/mask_casexxx.nii.gz
        """
        
        dataset = 'MRSpineSeg'
        labels = [  # label value 1 to 19
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
                ]
        modality = 'MRI'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'MR')):
            case_id = case.split('Case')[-1][:-7]   # Case113.nii.gz -> 113
            mask_path = os.path.join(root_path, 'Mask', 'mask_case'+case_id+'.nii.gz')
            data.append({
                        'image':os.path.join(root_path, 'MR', case),
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id': case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_LUNA16(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/LUNA16'):
        """
        https://luna16.grand-challenge.org/Data/

        subset0.zip to subset9.zip: 10 zip files which contain all CT images
        (not included) annotations.csv: csv file that contains the annotations used as reference standard for the 'nodule detection' track
        lung segmentation: a directory that contains the lung segmentation for CT images computed using automatic algorithms

        LUNA16
        ├── part1
        |   ├── annotation.csv    (annotation file)
        ├── subset0
        |   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd    (img paths)
        |   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.raw    
        |   └── ... ... 
        └── ... ...
        """

        dataset = 'LUNA16'
        labels = ['left lung', 'right lung', 'trachea']
        modality = 'CT'
        data = []

        # iter over scans and generate masks
        for part in tqdm(range(10)):
            for f in os.listdir(os.path.join(root_path, 'subset%d'%part)):

                if not f.endswith('.mhd'):  # 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd
                    continue

                lung_seg_path = f'{root_path}/part1/seg-lungs-LUNA16/{f}'
                if not os.path.exists(lung_seg_path):
                    continue

                # optional) trans to nii.gz
                # img = sitk.ReadImage(join(root_path, 'subset%d'%part, f))
                # sitk.WriteImage(img, join(root_path, 'subset%d'%part, f.replace('.mhd', '.nii.gz')))
                # mask = sitk.ReadImage(lung_seg_path)
                # sitk.WriteImage(mask, lung_seg_path.replace('.mhd', '.nii.gz'))

                data.append({
                        # 'image':join(root_path, 'subset%d'%part, f.replace('.mhd', '.nii.gz')),
                        # 'mask':lung_seg_path.replace('.mhd', '.nii.gz'),
                        'image':join(root_path, 'subset%d'%part, f),
                        'mask':lung_seg_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'unknown',
                        'patient_id': f.split('.mhd')[0]
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                    
    def preprocess_MSD_Cardiac(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task02_Heart'):
        """
        http://medicaldecathlon.com/#tasks
        
        Mono-modal MRI
        20 3D volumes 
        
        "modality": { 
        "0": "MRI"
        }, 
        "labels": { 
            "0": "background", 
            "1": "left atrium"
        }, 
        
        img paths: root_path/imagesTr/la_001.nii.gz
        mask paths: root_path/labelsTr/la_001.nii.gz
        """
        
        dataset = 'MSD_Cardiac'
        labels = ['left atrium']
        modality = 'MRI'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_MSD_Liver(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task03_Liver'):
        """
        http://medicaldecathlon.com/#tasks
        
        Portal venous phase CT
        131 3D volumes
        
        "modality": { 
            "0": "CT"
        }, 
    
        "labels": { 
            "0": "background", 
            "1": "liver", 
            "2": "cancer"
        }
        
        img paths: root_path/imagesTr/liver_0.nii.gz
        mask paths: root_path/labelsTr/liver_0.nii.gz
        """
        
        dataset = 'MSD_Liver'
        labels = ['liver', 'liver tumor']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_MSD_Hippocampus(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task04_Hippocampus'):
        """
        http://medicaldecathlon.com/#tasks
        
        Mono-modal MRI 
        394 3D volumes
        
        "modality": { 
            "0": "MRI"
        }, 
        "labels": { 
            "0": "background", 
            "1": "Anterior", 
            "2": "Posterior"
        }, 
        
        img paths: root_path/imagesTr/hippocampus_001.nii.gz
        mask paths: root_path/labelsTr/hippocampus_001.nii.gz
        """
        
        dataset = 'MSD_Hippocampus'
        labels = ['Anterior Hippocampus', 'Posterior Hippocampus']
        modality = 'MRI'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_MSD_Prostate(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task05_Prostate'):
        """
        http://medicaldecathlon.com/#tasks
        
        Multimodal MR (T2, ADC)
        32 4D volumes
        
        "modality": { 
            "0": "T2", 
            "1": "ADC"
        }, 
        "labels": { 
            "0": "background", 
            "1": "TZ", 
            "2": "PZ"
        }, 
        
        img paths: root_path/imagesTr/prostate_00.nii.gz
        mask paths: root_path/labelsTr/prostate_00.nii.gz
        """
        
        dataset = 'MSD_Prostate'
        labels = ['transition zone of prostate', 'peripheral zone of prostate']
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            for mod, modality in zip(["T2", "ADC"], ["MRI T2", "MRI ADC"]):
                img_path = os.path.join(root_path, 'imagesTr', case, mod)
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                     
    def preprocess_MSD_Lung(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task06_Lung'):
        """
        http://medicaldecathlon.com/#tasks
        
        64 3D volumes
        
        "modality": { 
            "0": "CT"
        }, 
        "labels": { 
            "0": "background", 
            "1": "cancer"
        },
        img paths: root_path/imagesTr/lung_00.nii.gz
        mask paths: root_path/labelsTr/lung_00.nii.gz
        """
        
        dataset = 'MSD_Lung'
        labels = ['lung tumor']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
          
    def preprocess_MSD_Pancreas(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task07_Pancreas'):
        """
        http://medicaldecathlon.com/#tasks
        
        282 3D volumes
        
        "modality": { 
            "0": "CT"
        }, 
        "labels": { 
            "0": "background", 
            "1": "pancreas", 
            "2": "cancer"
        },  
    
        img paths: root_path/imagesTr/pancreas_001.nii.gz
        mask paths: root_path/labelsTr/pancreas_001.nii.gz
        """
        
        dataset = 'MSD_Pancreas'
        labels = ['pancreas', 'pancreas tumor']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')      
                
    def preprocess_MSD_HepaticVessel(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task08_HepaticVessel'):
        """
        http://medicaldecathlon.com/#tasks
        
        303 3D volumes
        
        "modality": { 
            "0": "CT"
        }, 
        "labels": { 
            "0": "background", 
            "1": "Vessel", 
            "2": "Tumour"
        }, 
    
        img paths: root_path/imagesTr/hepaticvessel_001.nii.gz
        mask paths: root_path/labelsTr/hepaticvessel_001.nii.gz
        """
        
        dataset = 'MSD_HepaticVessel'
        labels = ['liver vessel', 'liver tumor']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_MSD_Spleen(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task09_Spleen'):
        """
        http://medicaldecathlon.com/#tasks
        
        41 3D volumes
        
        "modality": { 
            "0": "CT"
        }, 
        "labels": { 
            "0": "background", 
            "1": "spleen"
        }, 
    
        img paths: root_path/imagesTr/spleen_2.nii.gz
        mask paths: root_path/labelsTr/spleen_2.nii.gz
        """
        
        dataset = 'MSD_Spleen'
        labels = ['spleen']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_MSD_Colon(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task10_Colon'):
        """
        http://medicaldecathlon.com/#tasks
        
        126 3D volumes
        
        "modality": { 
            "0": "CT"
        }, 
        "labels": { 
            "0": "background", 
            "1": "colon cancer primaries"
        }, 
    
        img paths: root_path/imagesTr/colon_001.nii.gz
        mask paths: root_path/labelsTr/colon_001.nii.gz
        """
        
        dataset = 'MSD_Colon'
        labels = ['colon cancer']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'imagesTr')):  # xxx_001.nii.gz
            if case[0] == '.':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz
                continue
            mask_path = os.path.join(root_path, 'labelsTr', case)
            img_path = os.path.join(root_path, 'imagesTr', case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                            
    def preprocess_SKI10(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/SKI10Data'):
        """
        https://ski10.grand-challenge.org
        
        The goal of SKI10 was to compare different algorithms for cartilage and bone segmentation from knee MRI data. 
        Knee cartilage segmentation is a clinically relevant segmentation problem that has gained considerable importance in recent years. 
        Among others, it is used to quantify cartilage deterioration for the diagnosis of Osteoarthritis and to optimize surgical planning of knee implants.
        
        The last training data set (images 61-100) includes corresponding ROI images; these specify regions of interest where cartilage segmentations will be evaluated.
    
        Segmentations are multi-label images with the following codes: 
        0=background, 1=femur bone, 2=femur cartilage, 3=tibia bone, 4=tibia cartilage.
    
        img paths: root_path/training(validation)/image-xxx.mhd
        mask paths: root_path/training(validation)/labels-xxx.mhd
        """
        
        dataset = 'SKI10'
        labels = ['femur bone', 'femur cartilage', 'tibia bone', 'tibia cartilage']
        modality = 'MRI'
        data = []
        
        for split in ['training', 'validation']:
            for case in os.listdir(os.path.join(root_path, split)):  # xxx.png
                if 'image' in case and '.mhd' in case and 'roi' not in case: # images-xxx.mhd
                    patient_id = case
                    
                    img_path = os.path.join(root_path, split, case)
                    img = sitk.ReadImage(img_path)
                    sitk.WriteImage(img, img_path.replace('.mhd', '.nii.gz'))
                    
                    mask_path = img_path.replace('image-', 'labels-')
                    mask = sitk.ReadImage(mask_path)
                    sitk.WriteImage(mask, mask_path.replace('.mhd', '.nii.gz'))
                    
                    is_train = 'train' if split == 'training' else 'valid'
                    data.append({
                                'image':img_path.replace('.mhd', '.nii.gz'),
                                'mask':mask_path.replace('.mhd', '.nii.gz'),
                                'label':labels,
                                'modality':modality,
                                'dataset':dataset,
                                'official_split':is_train,
                                'patient_id':patient_id.split('.mhd')[0].split('image-')[-1]
                                })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
                
    def preprocess_SLIVER07(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/SLIVER07'):
        """
        https://sliver07.grand-challenge.org/Home/
        
        The goal of Liver Competition 2007 (SLIVER07) is to compare different algorithms to segment the liver from clinical 3D computed tomography (CT) scans.
    
        img paths: root_path/scan/liver-origxxx.mhd
        mask paths: root_path/label/liver-segxxx.mhd
        """
        
        dataset = 'SLIVER07'
        labels = ['liver']
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'scan')):  # xxx.png
            if '.mhd' in case:
                patient_id = case
                mask_path = os.path.join(root_path, 'label', case).replace('liver-orig', 'liver-seg')
                img_path = os.path.join(root_path, 'scan', case)
                
                # convert to nii.gz (unsupported mhd type for itk-python)
                img = sitk.ReadImage(img_path)
                sitk.WriteImage(img, img_path.replace('.mhd', '.nii.gz'))
                msk = sitk.ReadImage(mask_path)
                sitk.WriteImage(msk, mask_path.replace('.mhd', '.nii.gz'))

                data.append({
                            'image':img_path.replace('.mhd', '.nii.gz'),
                            'mask':mask_path.replace('.mhd', '.nii.gz'),
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':patient_id
                            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_PROMISE12(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/PROMISE12'):
        """
        https://liuquande.github.io/SAML/
        
        This is a well-organized multi-site dataset for prostate MRI segmentation, 
        which contains prostate T2-weighted MRI data (with segmentation mask) collected from six different data sources out of three public datasets.
        
        img paths: root_path/TrainingData_PartX/Casexx.mhd
        mask paths: root_path/TrainingData_PartX/Casexx_segmentation.mhd
        """
        
        dataset = 'PROMISE12'
        labels = ['prostate']
        modality = 'MRI T2'
        data = []
        
        for part in ['1', '2', '3']:
            part_path = os.path.join(root_path, 'TrainingData_Part%s'%part)
            for case in os.listdir(part_path):  # Case00.mhd
                if 'Case' in case and '.mhd' in case and 'segmentation' not in case:
                    img_path = os.path.join(root_path, 'TrainingData_Part%s'%part, case)
                    mask = case[:-4]+'_segmentation.mhd'
                    mask_path = os.path.join(root_path, 'TrainingData_Part%s'%part, mask)
                    
                    img = sitk.ReadImage(img_path)
                    sitk.WriteImage(img, img_path[:-4]+'.nii.gz')
                    msk = sitk.ReadImage(mask_path)
                    sitk.WriteImage(msk, mask_path[:-4]+'.nii.gz')
                    
                    data.append({
                                'image':img_path[:-4]+'.nii.gz',
                                'mask':mask_path[:-4]+'.nii.gz',
                                'label':labels,
                                'modality':modality,
                                'dataset':dataset,
                                'official_split': 'train',
                                'patient_id':img_path[:-4]+'.nii.gz',
                                })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_BrainPTM(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/sheba75_data'):
        """
        https://brainptm-2021.grand-challenge.org/Participation/
        
        The dataset comprise 75 clinical cases with T1w Structural and Diffusion Weighted (DW) modalities.
        The DW protocol had 64 gradient directions.
        Each T1w and DW scan is stored in a separate nifti file with 128x144x128 spatial dimension.
        White matter tracts mapping annotations were acquired in a semi-manual process for each scan.
        
        sheba75_data
        ├── sheba75_data_train
        |   ├── case_1    
        |   |   ├── T1.nii.gz   (img paths)
        |   |   └── brain_mask.nii.gz   (not mask paths! this is the mask of the whole brain)
        |   └── ... ... 
        ├── sheba75_tracts_train
        |   ├── case_1    
        |   |   ├── OR_left.nii.gz   (official mask paths)
        |   |   └── OR_right.nii.gz   (official mask paths)
        |   └── ... ... 
        └── sheba75_data_test   (annotation not provided)
            ├── case_61
            |   ├── T1.nii.gz   
            |   └── brain_mask.nii.gz   
            └── ... ... 
        """
        
        dataset = 'BrainPTM'
        labels = ['Left Optic Radiation', 'Right Optic Radiation', 'Left Corticospinal Tract', 'Right Corticospinal Tract', 'Brain']
        data = []
        
        for case in os.listdir(os.path.join(root_path, 'sheba75_data_train')):
            t1_img_path = os.path.join(root_path, 'sheba75_data_train', case, 'T1.nii.gz')

            # dw_img_path = os.path.join(root_path, 'sheba75_data_train', case, 'Diffusion.nii.gz')
            left_or_mask_path = os.path.join(root_path, 'sheba75_tracts_train', case, 'OR_left.nii.gz')
            right_or_mask_path = os.path.join(root_path, 'sheba75_tracts_train', case, 'OR_right.nii.gz')
            left_cst_mask_path = os.path.join(root_path, 'sheba75_tracts_train', case, 'CST_left.nii.gz')
            right_cst_mask_path = os.path.join(root_path, 'sheba75_tracts_train', case, 'CST_right.nii.gz')
            brain_mask_path = os.path.join(root_path, 'sheba75_data_train', case, 'brain_mask.nii.gz')
            
            monai_loader = monai.transforms.LoadImage(image_only=True)
            t1_img = monai_loader(t1_img_path)
            h, w, d = t1_img.shape
            
            mc_mask = []
            for path in [left_or_mask_path, right_or_mask_path, left_cst_mask_path, right_cst_mask_path]:
                if os.path.exists(path):
                    mask = monai_loader(path)
                    mc_mask.append(mask)  # [H, W, D]
                else:
                    mc_mask.append(np.zeros((h, w, d)))
                    
            brain_mask = monai_loader(brain_mask_path)
            brain_mask = np.flip(brain_mask, 0)
            mc_mask.append(brain_mask)
                    
            mc_mask = np.stack(mc_mask, axis=0) # CHWD

            mc_mask_nib = nib.Nifti1Image(mc_mask, mask.affine)
            nib.save(mc_mask_nib, os.path.join(root_path, 'sheba75_tracts_train', case, 'label.nii.gz'))
            
            data.append({
                        'image':t1_img_path,
                        'mask':os.path.join(root_path, 'sheba75_tracts_train', case, 'label.nii.gz'),
                        'label':labels,
                        'modality':'MRI T1',
                        'dataset':dataset,
                        'official_split':'train',
                        'patient_id':case
                        })
            
            """data.append({
                        'image':dw_img_path,
                        'mask':os.path.join(root_path, 'sheba75_tracts_train', case, 'label.nii.gz'),
                        'label':labels,
                        'modality':'MRI DWI',
                        'dataset':dataset,
                        'official_split': 'train',
                        })"""
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_WMH_Segmentation_Challenge(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/WMH_Segmentation_Challenge/dataverse_files'):            
        """
        https://wmh.isi.uu.nl
        
        Image data used in this challenge were acquired from five different scanners from three different vendors in three different hospitals in the Netherlands and Singapore. 
        For each subject, a 3D T1weighted image and a 2D multi-slice FLAIR image are provided.

        The original FLAIR image is used to manually delineate the WMH and participants should provide the results in this image space.
        NOTE: Only the FLAIR scan is annotated
        
        dataverse_files
        └── test/train
            └── Amsterdam
            |   └── GE1T5/GE3T/...
            |       └── 100/101/...
            |           ├── orig
            |           |   └── FLAIR.nii.gz    (img paths)
            |           └── wmh.nii.gz   (mask paths)
            └── Singapore/Utrecht
                └── 70/71/...
                    ├── orig
                    |   └── FLAIR.nii.gz    (img paths)
                    └── wmh.nii.gz   (mask paths)
        """
        
        dataset = 'WMH_Segmentation_Challenge'
        labels = ['white matter hyperintensities']
        data = []
        
        for split in ['training', 'test']:
            for area in ['Singapore', 'Utrecht']:
                for case in os.listdir(os.path.join(root_path, split, area)):
                    img_path = os.path.join(root_path, split, area, case, 'orig/FLAIR.nii.gz')
                    mask_path = os.path.join(root_path, split, area, case, 'wmh.nii.gz')
                    official_split = 'train' if split == 'training' else 'test'
                    data.append({
                                'image':img_path,
                                'mask':mask_path,
                                'label':labels,
                                'modality':'MRI FLAIR',
                                'dataset':dataset,
                                'official_split': official_split,
                                'patient_id':case,
                                })
            area = 'Amsterdam'
            for device in os.listdir(os.path.join(root_path, split, area)):
                for case in os.listdir(os.path.join(root_path, split, area, device)):
                        img_path = os.path.join(root_path, split, area, device, case, 'orig/FLAIR.nii.gz')
                        mask_path = os.path.join(root_path, split, area, device, case, 'wmh.nii.gz')
                        official_split = 'train' if split == 'training' else 'test'
                        data.append({
                                    'image':img_path,
                                    'mask':mask_path,
                                    'label':labels,
                                    'modality':'MRI FLAIR',
                                    'dataset':dataset,
                                    'official_split': official_split,
                                    'patient_id':case,
                                    })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_WORD(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/WORD-V0.1.0'):            
        """
        https://github.com/HiLab-git/WORD
        
        WORD is a dataset for organ semantic segmentation 
        that contains 150 abdominal CT volumes (30,495 slices) 
        and each volume has 16 organs with fine pixel-level annotations and scribble-based sparse annotation, 
        which may be the largest dataset with whole abdominal organs annotation.
        
        "modality": {
            "0": "CT"
        },
        
        WORD-V0.1.0
        ├── imagesTr
        |   ├── word_0002.nii.gz    (img paths)
        |   └── ... ... 
        ├── labelsTr
        |   ├── word_0002.nii.gz    (mask paths)
        |   └── ... ... 
        ├── imagesVal
        |   ├── word_0014.nii.gz
        |   └── ... ... 
        ├── labelsVal
        |   ├── word_0014.nii.gz
        |   └── ... ... 
        ├── imagesTs
        |   ├── word_0001.nii.gz
        |   └── ... ... 
        └── labelsTs
            ├── word_0001.nii.gz
            └── ... ... 
        """
        
        dataset = 'WORD'
        labels = [
                "liver",    # 1
                "spleen",
                "left kidney",
                "right kidney",
                "stomach",  # 5
                "gallbladder",
                "esophagus",
                "pancreas",
                "duodenum",
                "colon",    # 10
                "intestine",
                "adrenal gland",
                "rectum",   # 13
                "urinary bladder",
                "head of left femur", # 15
                "head of right femur" # 16
                ]
        modality = 'CT'
        data = []
        
        for suffix, split in zip(['Tr', 'Ts', 'Val'], ['train', 'test', 'validation']):
            for case in os.listdir(os.path.join(root_path, "images"+suffix)):
                patient_id = case
                img_path = os.path.join(root_path, "images"+suffix, case)
                mask_path = os.path.join(root_path, "labels"+suffix, case)
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': split,
                            'patient_id': patient_id
                            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
    
    def preprocess_TotalSegmentator(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Totalsegmentator_dataset'):            
        """
        In 1204 CT images we segmented 104 anatomical structures (27 organs, 59 bones, 10 muscles, 8 vessels) 
        covering a majority of relevant classes for most use cases.
        
        Totalsegmentator_dataset
        ├── s0000
        |   ├── ct.nii.gz    (img paths)
        |   ├── ribs_label.nii.gz    (processed mask paths)
        |   ├── muscles_label.nii.gz
        |   ├── cardiac_label.nii.gz
        |   ├── vertebrae_label.nii.gz
        |   ├── organs_label.nii.gz
        |   └── segmentations
        |       ├── adrenal_gland_left.nii.gz   (official mask paths)
        |       ├── adrenal_gland_right.nii.gz
        |       └── ... ... 
        ... ...
        └── meta.csv
        """
        modality = 'CT'
        data = []
        
        # The ORDER of labels is the same with 'os.listdir('Totalsegmentator_dataset/case/segmentations').sort()'
        labels = [
            'left adrenal gland',
            'right adrenal gland',
            'aorta',
            'left autochthon',
            'right autochthon',
            'brain',
            'left clavicle',
            'right clavicle',
            'colon',
            'duodenum',
            'esophagus',
            'face',
            'left femur',
            'right femur',
            'gallbladder',
            'left gluteus maximus',
            'right gluteus maximus',
            'left gluteus medius',
            'right gluteus medius',
            'left gluteus minimus',
            'right gluteus minimus',
            'left heart atrium',
            'right heart atrium',
            'heart myocardium',
            'left heart ventricle',
            'right heart ventricle',
            'left hip',
            'right hip',
            'left humerus',
            'right humerus',
            'left iliac artery',
            'right iliac artery',
            'left iliac vena',
            'right iliac vena',
            'left iliopsoas',
            'right iliopsoas',
            'inferior vena cava',
            'left kidney',
            'right kidney',
            'liver',
            'left lung lower lobe',
            'right lung lower lobe',
            'right lung middle lobe',
            'left lung upper lobe',
            'right lung upper lobe',
            'pancreas',
            'portal vein and splenic vein',
            'pulmonary artery',
            'left rib 1',
            'left rib 10',                                                                                                                                     
            'left rib 11',                                                                                                                                       
            'left rib 12',
            'left rib 2',
            'left rib 3',
            'left rib 4',
            'left rib 5',
            'left rib 6',
            'left rib 7',
            'left rib 8',
            'left rib 9',
            'right rib 1',
            'right rib 10',
            'right rib 11',
            'right rib 12',
            'right rib 2',
            'right rib 3',
            'right rib 4',
            'right rib 5',
            'right rib 6',
            'right rib 7',
            'right rib 8',
            'right rib 9',
            'sacrum',                                                                                                                                            
            'left scapula',                                                                                                                                      
            'right scapula',                                                                                                                                     
            'small bowel',                                                                                                                                       
            'spleen',
            'stomach',
            'trachea',
            'urinary bladder',
            'cervical vertebrae 1 (C1)',
            'cervical vertebrae 2 (C2)',
            'cervical vertebrae 3 (C3)',
            'cervical vertebrae 4 (C4)',
            'cervical vertebrae 5 (C5)',
            'cervical vertebrae 6 (C6)',
            'cervical vertebrae 7 (C7)',
            'lumbar vertebrae 1 (L1)',
            'lumbar vertebrae 2 (L2)',
            'lumbar vertebrae 3 (L3)',
            'lumbar vertebrae 4 (L4)',
            'lumbar vertebrae 5 (L5)',
            'thoracic vertebrae 1 (T1)',
            'thoracic vertebrae 10 (T10)',
            'thoracic vertebrae 11 (T11)',
            'thoracic vertebrae 12 (T12)',
            'thoracic vertebrae 2 (T2)',
            'thoracic vertebrae 3 (T3)',
            'thoracic vertebrae 4 (T4)',
            'thoracic vertebrae 5 (T5)',
            'thoracic vertebrae 6 (T6)',
            'thoracic vertebrae 7 (T7)',
            'thoracic vertebrae 8 (T8)',
            'thoracic vertebrae 9 (T9)',
            ]

        TotalSegmentator_Organs = []
        TotalSegmentator_Vertebrae = []
        TotalSegmentator_Cardiac = []
        TotalSegmentator_Muscles = []
        TotalSegmentator_Ribs = []
        
        organs = [
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
        ]
        vertebrae = [
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
        ]
        cardiac = [
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
        ]
        muscles = [
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
        ]
        ribs = [
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
        ]

        for case in tqdm(os.listdir(os.path.join(root_path))):
            
            avoid_read_and_save = False
            
            if case == 'meta.csv' or len(case)!=5 or case[0]!= 's':
                continue
            img_path = os.path.join(root_path, case, "ct.nii.gz")
            
            # 避免重复处理mask
            if os.path.exists(os.path.join(root_path, case, 'organs_label.nii.gz')) and \
                os.path.exists(os.path.join(root_path, case, 'vertebrae_label.nii.gz')) and \
                    os.path.exists(os.path.join(root_path, case, 'cardiac_label.nii.gz')) and \
                        os.path.exists(os.path.join(root_path, case, 'muscles_label.nii.gz')) and \
                            os.path.exists(os.path.join(root_path, case, 'ribs_label.nii.gz')):
                                
                avoid_read_and_save = True
                
            sorted_labels = os.listdir(os.path.join(root_path, case, 'segmentations'))
            sorted_labels.sort()

            organ_mask = []
            vertebrae_mask = []
            cardiac_mask = []
            muscles_mask = []
            ribs_mask = []
            
            organ_label = []
            vertebrae_label = []
            cardiac_label = []
            muscles_label = []
            ribs_label = []
            
            i = 0
            for mask_file in sorted_labels:
                
                if not avoid_read_and_save:
                    mask = nib.load(os.path.join(root_path, case, 'segmentations', mask_file)) # [H, W, D]
                    mask_array = mask.get_fdata()   # hwd
                    
                label = labels[i]
                # concat_mask_array += mask_array * (i+1)
                if label in organs:
                    if not avoid_read_and_save:
                        organ_mask.append(mask_array)
                    organ_label.append(label)
                elif label in vertebrae:
                    if not avoid_read_and_save:
                        vertebrae_mask.append(mask_array)
                    vertebrae_label.append(label)
                elif label in cardiac:
                    if not avoid_read_and_save:
                        cardiac_mask.append(mask_array)
                    cardiac_label.append(label)
                elif label in muscles:
                    if not avoid_read_and_save:
                        muscles_mask.append(mask_array)
                    muscles_label.append(label)
                elif label in ribs:
                    if not avoid_read_and_save:
                        ribs_mask.append(mask_array)
                    ribs_label.append(label)
                elif label != 'face':
                    print(label)
                i += 1
            
            organ_mask_path = os.path.join(root_path, case, 'organs_label.nii.gz')        
            if not avoid_read_and_save:
                organ_mask = np.stack(organ_mask, axis=0)
                organ_mask_nib = nib.Nifti1Image(organ_mask, mask.affine, mask.header)
                nib.save(organ_mask_nib, organ_mask_path)
            
            vertebrae_mask_path = os.path.join(root_path, case, 'vertebrae_label.nii.gz')
            if not avoid_read_and_save:
                vertebrae_mask = np.stack(vertebrae_mask, axis=0)
                vertebrae_mask_nib = nib.Nifti1Image(vertebrae_mask, mask.affine, mask.header)
                nib.save(vertebrae_mask_nib, vertebrae_mask_path)
            
            cardiac_mask_path = os.path.join(root_path, case, 'cardiac_label.nii.gz')
            if not avoid_read_and_save:
                cardiac_mask = np.stack(cardiac_mask, axis=0)
                cardiac_mask_nib = nib.Nifti1Image(cardiac_mask, mask.affine, mask.header)
                nib.save(cardiac_mask_nib, cardiac_mask_path)
            
            muscles_mask_path = os.path.join(root_path, case, 'muscles_label.nii.gz')
            if not avoid_read_and_save:
                muscles_mask = np.stack(muscles_mask, axis=0)
                muscles_mask_nib = nib.Nifti1Image(muscles_mask, mask.affine, mask.header)
                nib.save(muscles_mask_nib, muscles_mask_path)
            
            ribs_mask_path = os.path.join(root_path, case, 'ribs_label.nii.gz')
            if not avoid_read_and_save:
                ribs_mask = np.stack(ribs_mask, axis=0)
                ribs_mask_nib = nib.Nifti1Image(ribs_mask, mask.affine, mask.header)
                nib.save(ribs_mask_nib, ribs_mask_path)
            
            TotalSegmentator_Organs.append({
                                            'image':img_path,
                                            'mask':organ_mask_path,
                                            'label':organ_label,
                                            'modality':modality,
                                            'dataset':'TotalSegmentator_Organs',
                                            'official_split': 'train',
                                            'patient_id':case,
                                            })
            
            TotalSegmentator_Vertebrae.append({
                                            'image':img_path,
                                            'mask':vertebrae_mask_path,
                                            'label':vertebrae_label,
                                            'modality':modality,
                                            'dataset':'TotalSegmentator_Vertebrae',
                                            'official_split': 'train',
                                            'patient_id':case,
                                            })
            
            TotalSegmentator_Cardiac.append({
                                            'image':img_path,
                                            'mask':cardiac_mask_path,
                                            'label':cardiac_label,
                                            'modality':modality,
                                            'dataset':'TotalSegmentator_Cardiac',
                                            'official_split': 'train',
                                            'patient_id':case,
                                            })
            
            TotalSegmentator_Muscles.append({
                                            'image':img_path,
                                            'mask':muscles_mask_path,
                                            'label':muscles_label,
                                            'modality':modality,
                                            'dataset':'TotalSegmentator_Muscles',
                                            'official_split': 'train',
                                            'patient_id':case,
                                            })
            TotalSegmentator_Ribs.append({
                                            'image':img_path,
                                            'mask':ribs_mask_path,
                                            'label':ribs_label,
                                            'modality':modality,
                                            'dataset':'TotalSegmentator_Ribs',
                                            'official_split': 'train',
                                            'patient_id':case,
                                            })
            
        for dataset, data in zip(['TotalSegmentator_Organs', 'TotalSegmentator_Vertebrae', 'TotalSegmentator_Cardiac', 'TotalSegmentator_Muscles', 'TotalSegmentator_Ribs'], [TotalSegmentator_Organs, TotalSegmentator_Vertebrae, TotalSegmentator_Cardiac, TotalSegmentator_Muscles, TotalSegmentator_Ribs]):
            Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
            with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
                for datum in data:
                    f.write(json.dumps(datum)+'\n')
    
    def preprocess_TotalSegmentator_v2(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Totalsegmentator_version2'):            
        """
        The same images as v1, but add 19 new labels and a merged one (heart)
        
        image_root should be the same as v1
        root_path -> mask_path
        """
        # NOTE: set image_root as the root_path of TotalSegmentator 
        image_root = '/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Totalsegmentator_dataset'
            
        dataset = 'TotalSegmentator_v2'
        modality = 'CT'
        data = []
        
        file_names = [
            'atrial_appendage_left.nii.gz',
            'brachiocephalic_trunk.nii.gz',
            'brachiocephalic_vein_left.nii.gz',
            'brachiocephalic_vein_right.nii.gz',
            'common_carotid_artery_left.nii.gz',
            'common_carotid_artery_right.nii.gz',
            'costal_cartilages.nii.gz',
            'heart.nii.gz',
            'kidney_cyst_left.nii.gz',
            'kidney_cyst_right.nii.gz',
            'prostate.nii.gz',
            'pulmonary_vein.nii.gz',
            'skull.nii.gz',
            'spinal_cord.nii.gz',
            'sternum.nii.gz',
            'subclavian_artery_left.nii.gz',
            'subclavian_artery_right.nii.gz',
            'superior_vena_cava.nii.gz',
            'thyroid_gland.nii.gz',
            'vertebrae_S1.nii.gz'
        ]
        
        labels = [
            'left auricle of heart',
            'brachiocephalic trunk',
            'left brachiocephalic vein',
            'right brachiocephalic vein',
            'left common carotid artery',
            'right common carotid artery',
            'costal cartilage',
            'heart',    #
            'left kidney cyst',
            'right kidney cyst',
            'prostate', #
            'pulmonary vein',
            'skull',    #
            'spinal cord',  #
            'sternum',  #
            'left subclavian artery',
            'right subclavian artery',
            'superior vena cava',
            'thyroid gland',
            'sacral vertebrae 1 (S1)'
        ]
        
        for case in tqdm(os.listdir(os.path.join(image_root))):  # s1000
            if len(case) != 5 or not case.startswith('s'):
                continue
            
            new_mask = []
            for file_name, label_name in zip(file_names, labels):
                mask_path = os.path.join(root_path, case, 'segmentations', file_name)
                
                # empty anno
                if not os.path.exists(mask_path):
                    tmp = os.path.join(image_root, case, 'organs_label.nii.gz')
                    tmp = nib.load(tmp).get_fdata()
                    new_mask.append(np.zeros_like(tmp))
                    print(mask_path)
                else:
                    mask = nib.load(os.path.join(root_path, case, 'segmentations', file_name)) # [H, W, D]
                    mask_array = mask.get_fdata()   # hwd
                    new_mask.append(mask_array)
                    
            new_mask = np.stack(new_mask, axis=0)
            assert new_mask.shape[0] == len(labels)
            
            new_mask_nib = nib.Nifti1Image(new_mask, mask.affine, mask.header)
            new_mask_path = os.path.join(image_root, case, 'v2_label.nii.gz')
            nib.save(new_mask_nib, new_mask_path)
            print(f'save v2 new labels to {new_mask_path}')
            
            img_path = os.path.join(image_root, case, 'ct.nii.gz')
            data.append({
                        'image':img_path,
                        'mask':new_mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':'TotalSegmentator_v2',
                        'official_split': 'unknown',
                        'patient_id':case,
                        })
            
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_FLARE22(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/FLARE22'):
        """
        https://flare22.grand-challenge.org
        
        50 CT cases with 13 annotated organs
        
        FLARE22
        ├── images
        |   ├── FLARE22_Tr_0001_0000.nii.gz     # img path
        |   |
        |   ... ...
        |   |
        |   └── FLARE22_Tr_0050_0000.nii.gz
        |
        └── labels
            ├── FLARE22_Tr_0001.nii.gz      # mask path
            |
            ... ...
            |
            └── FLARE22_Tr_0050.nii.gz
        """
        
        dataset = 'FLARE22'
        labels = [
            'Liver',    # 1
            'Right kidney',
            'Spleen',
            'Pancreas',
            'Aorta',
            'Inferior Vena Cava',
            'Right Adrenal Gland',
            'Left Adrenal Gland',
            'Gallbladder',
            'Esophagus',
            'Stomach',
            'Duodenum',
            'Left kidney'   # 13
            ]
        data = []
        modality = 'CT'

        for case in os.listdir(os.path.join(root_path, 'labels')):
            msk_path = os.path.join(root_path, 'labels', case)
            img_path = os.path.join(root_path, 'images', case[:-7]+'_0000.nii.gz')
            data.append({
                    'image':img_path,
                    'mask':msk_path,
                    'label':labels,
                    'modality':modality,
                    'dataset':dataset,
                    'official_split': 'unknown',
                    'patient_id': case[:-7]+'_0000.nii.gz'
                    })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_NSCLC(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/NSCLC'):
        """
        https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68551327#685513274328f8386ccc42dcb282f6a42d8beffd
        
        a dataset of pleural effusion (only 78 cases) and thoracic cavity segmentations in subjects with diseased lungs

        NOTE: These segmentations use the RPI orientation, 
        but the original DICOM files are oriented using the RAI convention.  
        As a result, some tools such as ITK-SNAP will not render the segmentations in the correct orientation when visualized.  
        The authors of these data suggest using software like Mango (http://ric.uthscsa.edu/mango/) 
        or to convert to DICOM files to NIfTI with software like dcm2niix (https://github.com/rordenlab/dcm2niix) to address this issue.
        
        NSCLC/
        ├── Effusions/
        |   ├── LUNG1-001/
        |   |   ├── seg.npy (processed mask path)
        |   |   ├── LUNG1-001_effusion_first_reviewer.nii.gz    (official mask path)
        |   |   └── LUNG1-001_effusion_second_reviewer.nii.gz
        |   |
        |   |   ... ...
        |   |
        |   └── LUNG1-420/
        |       ├── LUNG1-420_effusion_first_reviewer.nii.gz
        |       └── LUNG1-420_effusion_second_reviewer.nii.gz
        |
        ├── manifest-1586193031612/
        |   ├── metadata.csv
        |   |
        |   └── NSCLC-Radiomics/
        |       ├── LUNG1-001/
        |       |   └── 09-18-2008-StudyID-NA-69331/
        |       |       └── 0.000000-NA-82046/
        |       |           ├── 0.000000-NA-82046_RCCTPET_THORAX_CONTRAST_dag0_20080918110916_4268307088.nii (processed image path)
        |       |           ├── 1-001.dcm   (official image path)
        |       |           |
        |       |           ... ...
        |       |           |
        |       |           └── 1-134.dcm
        |       |
        |       ... ...
        |       |
        |       └── LUNG1-422/
        └── Thoracic_Cavities/
            ├── LUNG1-001/
            |   └── LUNG1-001_thor_cav_primary_reviewer.nii.gz  (official mask path)
            ... ...
            |
            └── LUNG1-420/
                └── LUNG1-420_thor_cav_primary_reviewer.nii.gz
        """
        
        dataset = 'NSCLC'
        labels = ['thoracic cavity', 'effusion']    # 2, 1, 1
        data = []
        modality = 'CT'

        # convert dicom to NifTi：a demo
        # for folder in tqdm(os.listdir('/mnt/hwfile/medai/zhaoziheng/SAM/SAM/NSCLC/manifest-1586193031612/NSCLC-Radiomics')):
        #     if 'LUNG1-' not in folder:
        #         continue
        #     for tmp1 in os.listdir(os.path.join('/mnt/hwfile/medai/zhaoziheng/SAM/SAM/NSCLC/manifest-1586193031612/NSCLC-Radiomics', folder)):
        #         for tmp2 in os.listdir(os.path.join('/mnt/hwfile/medai/zhaoziheng/SAM/SAM/NSCLC/manifest-1586193031612/NSCLC-Radiomics', folder, tmp1)):
        #             os.system('dcm2niix %s'%os.path.join('/mnt/hwfile/medai/zhaoziheng/SAM/SAM/NSCLC/manifest-1586193031612/NSCLC-Radiomics', folder, tmp1, tmp2))
        #             print('dcm2niix %s'%os.path.join('/mnt/hwfile/medai/zhaoziheng/SAM/SAM/NSCLC/manifest-1586193031612/NSCLC-Radiomics', folder, tmp1, tmp2))

        monai_loader = monai.transforms.LoadImage(image_only=True)
        for case in tqdm(os.listdir(root_path+'/manifest-1586193031612/NSCLC-Radiomics')):
            if 'LUNG1-' not in case:    # LUNG1-420
                continue
            for tmp1 in os.listdir(os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case)):    # 07-25-2010-NA-NA-06557
                for tmp2 in os.listdir(os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1)):  # 0.000000-NA-42364
                    # check if the dicom has been translated to nifti
                    translated = False
                    for f in os.listdir(os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2)):
                        if '.nii' in f and f != 'concat_label.nii.gz':
                            translated = True
                            break
                    if not translated:
                        os.system('dcm2niix %s'%os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2))
                        for f in os.listdir(os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2)):
                            if '.nii' in f and f != 'concat_label.nii.gz':
                                break
                    # so far, the dicom has been translated to nifti (i.e, f) anyway

                    img_path = os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2, f)
                    image_nib = nib.load(img_path)
                    image = image_nib.get_fdata()
                    
                    # load thoracic cavity mask
                    thor_cav_msk_path = os.path.join(root_path, 'Thoracic_Cavities', case, '%s_thor_cav_primary_reviewer.nii.gz'%case)
                    if os.path.exists(thor_cav_msk_path):
                        thor_cav_seg = monai_loader(thor_cav_msk_path)
                        thor_cav_seg = np.where(thor_cav_seg>0, 1.0, 0.0)
                    else:
                        print(thor_cav_msk_path)
                        thor_cav_seg = np.zeros_like(image)
                        continue
                    
                    # load effusion mask
                    effusion_msk_path = os.path.join(root_path, 'Effusions', case, '%s_effusion_first_reviewer.nii.gz'%case)
                    if os.path.exists(effusion_msk_path):
                        effusion = monai_loader(effusion_msk_path)
                    else:
                        print(effusion_msk_path)
                        effusion = np.zeros_like(image)
                        continue
                        
                    mc_mask = np.stack([thor_cav_seg, effusion], axis=0)  # [C, H, W, D]
                    mc_mask_nib = nib.Nifti1Image(mc_mask, image_nib.affine, image_nib.header)
                    nib.save(mc_mask_nib, os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2, 'concat_label.nii.gz'))
                    print(f"save to {os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2, 'concat_label.nii.gz')}")
                    
                    data.append({
                            'image':img_path,
                            'mask':os.path.join(root_path+'/manifest-1586193031612/NSCLC-Radiomics', case, tmp1, tmp2, 'concat_label.nii.gz'),
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'unknown',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')            

    def preprocess_COVID19(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/COVID-19-CT-Seg_20cases'):
        """
        https://zenodo.org/record/3757476#.Xpz8OcgzZPY
        
        This dataset contains 20 labeled COVID-19 CT scans. 
        Left lung, right lung, and infections are labeled by two radiologists and verified by an experienced radiologist. 
        In particular, we focus on learning to segment left lung, right lung, and infections using

        COVID-19-CT-Seg_20cases/
        ├── Lung_and_Infection_Mask/
        |   ├── coronacases_001.nii.gz  (official mask path)
        |   |
        |   ... ...
        |   |
        |   └── radiopaedia_40_86625_0.nii.gz   (exclude)
        |
        ├── coronacases_001.nii.gz  (official image path)
        |
        ... ...
        |
        └── radiopaedia_40_86625_0.nii.gz   (exclude)
        """
        
        dataset = 'COVID19'
        labels = ['left lung', 'right lung', 'COVID-19 infection']    # 1, 2, 3 
        data = []
        modality = 'CT'

        # monai_loader = monai.transforms.LoadImage(image_only=True)
        for case in tqdm(os.listdir(os.path.join(root_path, 'Lung_and_Infection_Mask'))):
            if 'radiopaedia' in case:
                continue 
            
            mask_path = os.path.join(root_path, 'Lung_and_Infection_Mask', case)
            img_path = os.path.join(root_path, case)
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'unknown',
                        'patient_id':case
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_Brain_Atlas(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Hammers_data'):
        """
        http://www.brain-development.org/brain-atlases/
        
        95 Regions
        
        immage: root_path/a06.nii.gz
        segmentation: root_path/a06-seg.nii.gz
        """
        
        dataset = 'Brain_Atlas'
        labels = [
            'left hippocampus',    # 1
            'right hippocampus',
            'left amygdala',
            'right amygdala',
            'left anterior temporal lobe medial part',
            'right anterior temporal lobe medial part',
            'left anterior temporal lobe lateral part',
            'right anterior temporal lobe lateral part',
            'left parahippocampal and ambient gyrus',
            'right parahippocampal and ambient gyrus',
            'left superior temporal gyrus middle part',
            'right superior temporal gyrus middle part',
            'left middle and inferior temporal gyrus',
            'right middle and inferior temporal gyrus',
            'left fusiform gyrus',
            'right fusiform gyrus',
            'left cerebellum',
            'right cerebellum',
            'brainstem excluding substantia nigra',
            'right insula posterior long gyrus',
            'left insula posterior long gyrus',
            'right lateral remainder occipital lobe',
            'left lateral remainder occipital lobe',
            'right anterior cingulate gyrus',
            'left anterior cingulate gyrus',
            'right posterior cingulate gyrus',
            'left posterior cingulate gyrus',
            'right middle frontal gyrus',
            'left middle frontal gyrus',
            'right posterior temporal lobe',
            'left posterior temporal lobe',
            'right angular gyrus',
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
            'left Lateral ventricle excluding temporal horn',
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
        ]
        data = []
        modality = 'MRI'

        for case in tqdm(os.listdir(root_path)):
            if '-seg.nii.gz' in case:
                mask_path = os.path.join(root_path, case)
                img_path = mask_path.replace('-seg.nii.gz', '.nii.gz')
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'unknown',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_Couinaud_Liver(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Couinaud_Liver'):
        """
        Couinaud Liver Segmentation (8 parts of liver)
        
        images are from MSD_Liver
        """
        
        # NOTE: set this well you place the image of MSD_HepaticVessel
        root_path_to_image = "/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MSD/Task08_HepaticVessel/imagesTr"
        
        dataset = 'Couinaud_Liver'
        labels = [
            'caudate lobe', 
            'left lateral superior segment of liver',
            'Left lateral inferior segment of liver',
            'left medial segment of liver',
            'right anterior inferior segment of liver',
            'right posterior inferior segment of liver',
            'right posterior superior segment of liver',
            'right anterior superior segment of liver'
            ]
        modality = 'CT'
        data = []
        
        for case in os.listdir(os.path.join(root_path)):  # xxx_063.nii.gz
            if case[:13] == 'hepaticvessel' and case[-7:] == '.nii.gz':  # after untar, seems that every xxx.nii.gz is companied with another .xxx.nii.gz

                mask_path = os.path.join(root_path, case)
                img_path = os.path.join(root_path_to_image, case)
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_AMOS22_CT(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/AMOS22'):
        """
        https://zenodo.org/records/7155725#.Y0OOCOxBztM.
        """
        
        dataset = 'AMOS22_CT'
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
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'amos22', 'imagesTr'))) if x.endswith('nii.gz') and int(x.split('.')[0].split('_')[1]) <= 500]
        for case in case_list:
            img_path = join(root_path, 'amos22', 'imagesTr', case)
            mask_path = join(root_path, 'amos22', 'labelsTr', case)
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'amos22', 'imagesVa'))) if x.endswith('nii.gz') and int(x.split('.')[0].split('_')[1]) <= 500]
        for case in case_list:
            img_path = join(root_path, 'amos22', 'imagesVa', case)
            mask_path = join(root_path, 'amos22', 'labelsVa', case)
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'valid',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_AMOS22_MRI(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/AMOS22'):
        """
        https://zenodo.org/records/7155725#.Y0OOCOxBztM.
        """
        
        dataset = 'AMOS22_MRI'
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
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'amos22', 'imagesTr'))) if x.endswith('nii.gz') and int(x.split('.')[0].split('_')[1]) > 500]
        for case in case_list:
            img_path = join(root_path, 'amos22', 'imagesTr', case)
            mask_path = join(root_path, 'amos22', 'labelsTr', case)
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'amos22', 'imagesVa'))) if x.endswith('nii.gz') and int(x.split('.')[0].split('_')[1]) > 500]
        for case in case_list:
            img_path = join(root_path, 'amos22', 'imagesVa', case)
            mask_path = join(root_path, 'amos22', 'labelsVa', case)
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'valid',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_BTCV(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BTCV'):
        """
        BTCV Abdomen
        
        https://www.synapse.org/#!Synapse:syn3193805/wiki/217789
        """
        
        dataset = 'BTCV'
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
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'RawData', 'Training', 'img'))) if x.endswith('nii.gz')]
        for case in case_list:
            img_path = join(root_path, 'RawData', 'Training', 'img', case)
            mask_path = join(root_path, 'RawData', 'Training', 'label', case.replace('img', 'label'))
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split':'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_CT_ORG(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/CT_ORG'):
        """
        https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890
        """
        
        dataset = 'CT_ORG'
        labels = [
            "liver",
            "urinary bladder",
            "lung",
            "kidney",
            "bone",
            "brain",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'volumes'))) if x.endswith('nii.gz')]
        for case in case_list:
            img_path = join(root_path, 'volumes', case)
            mask_path = join(root_path, 'labels', case.replace('volume', 'labels'))
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'unknown',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_FeTA2022(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/FeTA2022'):
        """
        https://feta.grand-challenge.org/data-download/
        """
        
        dataset = 'FeTA2022'
        labels = [
            "external cerebrospinal fluid",
            "grey matter",
            "white matter",
            "brain ventricle",
            "cerebellum",
            "deep grey matter",
            "brainstem",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'feta_2.2'))) if x.startswith('sub-')]
        for case in case_list:
            img_path = join(root_path, 'feta_2.2', case, 'anat', case+'_rec-mial_T2w.nii.gz')
            mask_path = join(root_path, 'feta_2.2', case, 'anat', case+'_rec-mial_dseg.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'feta22_Vienna'))) if x.startswith('sub-')]
        for case in case_list:
            img_path = join(root_path, 'feta22_Vienna', case, 'anat', case+'_rec-nmic_T2w.nii.gz')
            mask_path = join(root_path, 'feta22_Vienna', case, 'anat', case+'_rec-nmic_dseg.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_ToothFairy(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ToothFairy'):
        """
        https://ditto.ing.unimore.it/toothfairy/
        """
        
        dataset = 'ToothFairy'
        labels = [
            "inferior alveolar nerve",
            ]
        modality = 'CT'
        data = []
        
        with open(join(root_path, 'ToothFairy_Dataset', 'splits.json'), 'r') as f:
            splits = json.load(f)

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'ToothFairy_Dataset', 'Dataset')))]
        for case in case_list:
            if not os.path.exists(join(root_path, 'ToothFairy_Dataset', 'Dataset', case, 'gt_alpha.npy')):
                continue
            img_path = join(root_path, 'ToothFairy_Dataset', 'Dataset', case, 'data.npy')
            mask_path = join(root_path, 'ToothFairy_Dataset', 'Dataset', case, 'gt_alpha.npy')
            if case in splits['train']:
                split_set = 'train'
            if case in splits['val']:
                split_set = 'valid'
            if case in splits['test']:
                split_set = 'test'
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': split_set,
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_Hecktor2022(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Hecktor2022'):
        """
        https://hecktor.grand-challenge.org/Data/
        """
        
        dataset = 'Hecktor2022'
        labels = [
            "head and neck tumor",
            "lymph node",
            ]
        modality = 'PET'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'hecktor2022_training', 'hecktor2022', 'imagesTr_resampled'))) if x.endswith('PT.nii.gz')]
        for case in case_list:  # CHUM-001__PT.nii.gz
            img_path = join(root_path, 'hecktor2022_training', 'hecktor2022', 'imagesTr_resampled', case)
            mask_path = join(root_path, 'hecktor2022_training', 'hecktor2022', 'labelsTr_resampled', case.replace('__PT', ''))
            patient_id = case.split('__PT.nii.gz')[0] # CHUM-001    
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':patient_id,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_PARSE2022(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/PARSE2022'):
        """
        https://parse2022.grand-challenge.org/Dataset/
        """
        
        dataset = 'PARSE2022'
        labels = [
            "pulmonary artery",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'train', 'train')))]
        for case in case_list:
            img_path = join(root_path, 'train', 'train', case, 'image', case+'.nii.gz')
            mask_path = join(root_path, 'train', 'train', case, 'label', case+'.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_SegTHOR(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/SegTHOR'):
        """
        https://competitions.codalab.org/competitions/21145#participate-get_starting_kit
        """
        
        dataset = 'SegTHOR'
        labels = [
            "esophagus",
            "heart",
            "trachea",
            "aorta",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'train')))]
        for case in case_list:
            img_path = join(root_path, 'train', case, case+'.nii.gz')
            mask_path = join(root_path, 'train', case, 'GT.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_MM_WHS_CT(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MM_WHS'):
        """
        https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA
        """
        
        dataset = 'MM_WHS_CT'
        labels = [
            "myocardium",
            "left heart atrium",
            "left heart ventricle",
            "right heart atrium",
            "right heart ventricle",
            "heart ascending aorta",
            "pulmonary artery",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'MM-WHS', 'ct_train'))) if 'image' in x]
        for case in case_list:
            img_path = join(root_path, 'MM-WHS', 'ct_train', case)
            mask_path = join(root_path, 'MM-WHS', 'ct_train', case.replace('image', 'label'))
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_MM_WHS_MRI(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MM_WHS'):
        """
        https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA
        """
        
        dataset = 'MM_WHS_MRI'
        labels = [
            "myocardium",
            "left heart atrium",
            "left heart ventricle",
            "right heart atrium",
            "right heart ventricle",
            "heart ascending aorta",
            "pulmonary artery",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'MM-WHS', 'mr_train'))) if 'image' in x]
        for case in case_list:
            img_path = join(root_path, 'MM-WHS', 'mr_train', case)
            mask_path = join(root_path, 'MM-WHS', 'mr_train', case.replace('image', 'label'))
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_CMRxMotion(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/CMRxMotion'):
        """
        https://www.synapse.org/Synapse:syn28503327/files/
        """

        dataset = 'CMRxMotion'
        labels = [
            "left heart ventricle",
            "myocardium",
            "right heart ventricle",
            ]
        modality = 'MRI'
        data = []

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'segmentation', 'data')))]
        for case in case_list:
            if not (os.path.exists(join(root_path, 'segmentation', 'data', case, case+'-ED-label.nii.gz')) and os.path.exists(join(root_path, 'segmentation', 'data', case, case+'-ES-label.nii.gz'))):
                continue 
            Path(join(root_path, 'segmentation', 'processed_data', case)).mkdir(exist_ok=True, parents=True)   

            img_path = join(root_path, 'segmentation', 'data', case, case+'-ED.nii.gz')
            img = nib.load(img_path)
            img = nib.Nifti1Image(np.squeeze(img.get_fdata()), img.affine, img.header)
            nib.save(img, join(root_path, 'segmentation', 'processed_data', case, case+'-ED.nii.gz'))
            mask_path = join(root_path, 'segmentation', 'data', case, case+'-ED-label.nii.gz')
            mask = nib.load(mask_path)
            mask = nib.Nifti1Image(mask.get_fdata().astype(np.uint8), img.affine, img.header)
            nib.save(mask, join(root_path, 'segmentation', 'processed_data', case, case+'-ED-label.nii.gz'))

            img_path = join(root_path, 'segmentation', 'data', case, case+'-ES.nii.gz')
            img = nib.load(img_path)
            img = nib.Nifti1Image(np.squeeze(img.get_fdata()), img.affine, img.header)
            nib.save(img, join(root_path, 'segmentation', 'processed_data', case, case+'-ES.nii.gz'))
            mask_path = join(root_path, 'segmentation', 'data', case, case+'-ES-label.nii.gz')
            mask = nib.load(mask_path)
            mask = nib.Nifti1Image(mask.get_fdata().astype(np.uint8), img.affine, img.header)
            nib.save(mask, join(root_path, 'segmentation', 'processed_data', case, case+'-ES-label.nii.gz'))

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'segmentation', 'processed_data')))]
        for case in case_list:
            img_path = join(root_path, 'segmentation', 'processed_data', case, case+'-ED.nii.gz')
            mask_path = join(root_path, 'segmentation', 'processed_data', case, case+'-ED-label.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })

            img_path = join(root_path, 'segmentation', 'processed_data', case, case+'-ES.nii.gz')
            mask_path = join(root_path, 'segmentation', 'processed_data', case, case+'-ES-label.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_LAScarQS22_Task1(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/LAScarQS22'):
        """
        https://zmiclab.github.io/projects/lascarqs22/data.html
        """
        
        dataset = 'LAScarQS22_Task1'
        labels = [
            "left heart atrium",
            "left heart atrium scar"
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'train_data')))]
        for case in case_list:
            Path(join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'processed_train_data', case)).mkdir(exist_ok=True, parents=True) 

            img_path = join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'train_data', case, 'enhanced.nii.gz')
            img = nib.load(img_path)
            nib.save(img, join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'processed_train_data', case, 'enhanced.nii.gz'))

            la_mask_path = join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'train_data', case, 'atriumSegImgMO.nii.gz')
            la_mask = nib.load(la_mask_path).get_fdata()
            la_mask[la_mask>0] = 1

            las_mask_path = join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'train_data', case, 'scarSegImgM.nii.gz')
            las_mask = nib.load(las_mask_path).get_fdata()
            la_mask[las_mask>0] = 2

            la_mask = nib.Nifti1Image(la_mask.astype(np.uint8), img.affine, img.header)
            nib.save(la_mask, join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'processed_train_data', case, 'label.nii.gz'))

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'processed_train_data')))]
        for case in case_list:
            img_path = join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'processed_train_data', case, 'enhanced.nii.gz')
            mask_path = join(root_path, 'LAScarQS2022_released_training_data', 'task1', 'processed_train_data', case, 'label.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_LAScarQS22_Task2(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/LAScarQS22'):
        """
        https://zmiclab.github.io/projects/lascarqs22/data.html
        """
        
        dataset = 'LAScarQS22_Task2'
        labels = [
            "left heart atrium",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'train_data')))]
        for case in case_list:
            Path(join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'processed_train_data', case)).mkdir(exist_ok=True, parents=True) 

            img_path = join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'train_data', case, 'enhanced.nii.gz')
            img = nib.load(img_path)
            nib.save(img, join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'processed_train_data', case, 'enhanced.nii.gz'))

            la_mask_path = join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'train_data', case, 'atriumSegImgMO.nii.gz')
            la_mask = nib.load(la_mask_path)
            la_mask = nib.Nifti1Image(la_mask.get_fdata().astype(np.uint8)>0, la_mask.affine, la_mask.header)
            nib.save(la_mask, join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'processed_train_data', case, 'label.nii.gz'))

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'processed_train_data')))]
        for case in case_list:
            img_path = join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'processed_train_data', case, 'enhanced.nii.gz')
            mask_path = join(root_path, 'LAScarQS2022_released_training_data', 'task2', 'processed_train_data', case, 'label.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
    
    def preprocess_ATLASR2(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ATLASR2'):
        """
        http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html
        """
        
        dataset = 'ATLASR2'
        labels = [
            "stroke",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'ATLAS_2', 'Training')))]
        for case in case_list:
            seq_list = [x for x in np.sort(os.listdir(join(root_path, 'ATLAS_2', 'Training', case)))]
            for seq in seq_list:
                if seq == 'dataset_description.json':
                    continue
                
                img_path = join(root_path, 'ATLAS_2', 'Training', case, seq, 'ses-1', 'anat', seq+'_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz')
                mask_path = join(root_path, 'ATLAS_2', 'Training', case, seq, 'ses-1', 'anat', seq+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
                data.append({
                    'image':img_path,
                    'mask':mask_path,
                    'label':labels,
                    'modality':modality,
                    'dataset':dataset,
                    'official_split': 'train',
                    'patient_id':case,
                })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_CrossMoDA2021(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/CrossMoDA2021'):
        """
        https://crossmoda.grand-challenge.org/Data/
        """
        
        dataset = 'CrossMoDA2021'
        labels = [
            "vestibular schwannoma",
            "cochlea",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'source_training'))) if x.endswith('ceT1.nii.gz')]
        for case in case_list:
            img_path = join(root_path, 'source_training', case)
            mask_path = join(root_path, 'source_training', case.replace('ceT1', 'Label'))
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_MyoPS2020(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/MyoPS2020'):
        """
        https://mega.nz/folder/BRdnDISQ#FnCg9ykPlTWYe5hrRZxi-w
        """
        
        dataset = 'MyoPS2020'
        labels = [
            "right heart ventricle",
            "myocardial scar",
            "myocardial edema",
            "myocardium",
            "left heart ventricle",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'train25_myops_gd')))]
        for case in case_list:
            Path(join(root_path, 'processed_train25_myops_gd')).mkdir(exist_ok=True, parents=True) 

            mask_path = join(root_path, 'train25_myops_gd', case)
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata().astype(np.uint8)
            mask_data[mask_data==88] = 1
            mask_data[mask_data==173] = 2
            mask_data[mask_data==196] = 3
            mask_data[mask_data==200] = 4
            mask_data[mask_data==244] = 5
            mask = nib.Nifti1Image(mask_data, mask.affine, mask.header)
            nib.save(mask, join(root_path, 'processed_train25_myops_gd', case))
        
        # case_list = [x for x in np.sort(os.listdir(join(root_path, 'test_data_gd')))]
        # for case in case_list:
        #     Path(join(root_path, 'processed_test_data_gd')).mkdir(exist_ok=True, parents=True) 

        #     mask_path = join(root_path, 'test_data_gd', case)
        #     mask = nib.load(mask_path)
        #     mask_data = mask.get_fdata().astype(np.uint8)
        #     mask_data[mask_data==200] = 1
        #     mask_data[mask_data==500] = 2
        #     mask_data[mask_data==600] = 3
        #     mask_data[mask_data==1220] = 4
        #     mask_data[mask_data==2221] = 5
        #     mask = nib.Nifti1Image(mask_data, mask.affine, mask.header)
        #     nib.save(mask, join(root_path, 'processed_test_data_gd', case))

        case_list = [x for x in np.sort(os.listdir(join(root_path, 'train25')))]
        for case in case_list:
            img_path = join(root_path, 'train25', case)
            mask_path = join(root_path, 'processed_train25_myops_gd', case[:-10]+'_gd.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case.split('_')[2],
            })

        # case_list = [x for x in np.sort(os.listdir(join(root_path, 'test20')))]
        # for case in case_list:
        #     img_path = join(root_path, 'test20', case)
        #     mask_path = join(root_path, 'processed_test_data_gd', case[:-10]+'_gdencrypt.nii.gz')
        #     data.append({
        #         'image':img_path,
        #         'mask':mask_path,
        #         'label':labels,
        #         'modality':modality,
        #         'dataset':dataset,
        #         'official_split': 'test',
        #         'patient_id':case,
        #     })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_Instance22(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Instance22'):
        """
        https://instance.grand-challenge.org/Dataset/
        """
        
        dataset = 'Instance22'
        labels = [
            "intracranial hemorrhage",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'train_2', 'data')))]
        for case in case_list:
            img_path = join(root_path, 'train_2', 'data', case)
            mask_path = join(root_path, 'train_2', 'label', case)
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_KiTS23(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/KiTS23'):
        """
        https://github.com/neheller/kits23
        """
        
        dataset = 'KiTS23'
        labels = [
            "kidney",
            "kidney tumor",
            "kidney cyst",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'dataset')))]
        for case in case_list:
            img_path = join(root_path, 'dataset', case, 'imaging.nii.gz')
            mask_path = join(root_path, 'dataset', case, 'segmentation.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_ATLAS(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/ATLAS'):
        """
        https://atlas-challenge.u-bourgogne.fr/dataset
        """
        
        dataset = 'ATLAS'
        labels = [
            "liver",
            "liver tumor",
            ]
        modality = 'MRI'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'atlas-train-dataset-1.0.1', 'train', 'imagesTr')))]
        for case in case_list:
            img_path = join(root_path, 'atlas-train-dataset-1.0.1', 'train', 'imagesTr', case)
            mask_path = join(root_path, 'atlas-train-dataset-1.0.1', 'train', 'labelsTr', case.replace('im', 'lb'))
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_KiPA22(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/KiPA22'):
        """
        https://kipa22.grand-challenge.org/dataset/
        """
        
        dataset = 'KiPA22'
        labels = [
            "renal vein",
            "kidney",
            "renal artery",
            "kidney tumor",
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'train', 'image')))]
        for case in case_list:
            img_path = join(root_path, 'train', 'image', case)
            img = nib.load(img_path)
            affine = img.affine
            header = img.header
            img = img.get_fdata() - 1000
            img_path = img_path.replace('image', 'processed_image')
            if not os.path.exists(join(root_path, 'train', 'processed_image')):
                os.makedirs(join(root_path, 'train', 'processed_image'))
            nib.save(nib.Nifti1Image(img, affine, header), img_path)

            mask_path = join(root_path, 'train', 'label', case)
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_BraTS2023_GLI(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BraTS2023_GLI'):
        """
        https://www.synapse.org/Synapse:syn53708126
        """
        
        dataset = 'BraTS2023_GLI'
        # labels = ['cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor']
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        data = []
        modality = 'MRI'
        for case in np.sort(os.listdir(join(root_path, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'))):  # 
            for seq in ['t1n', 't1c', 't2w', 't2f']:
                img_path = join(root_path, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData', case, case+'-'+seq+'.nii.gz')
                mask_path = join(root_path, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData', case, case+'-seg.nii.gz')
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_BraTS2023_MEN(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BraTS2023_MEN'):
        """
        https://www.synapse.org/Synapse:syn53708126
        """
        
        dataset = 'BraTS2023_MEN'
        # labels = ['cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor']
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        data = []
        modality = 'MRI'
        fixed_case_list = np.sort(os.listdir(join(root_path, 'BraTS-MEN-Train-Fix')))
        for case in np.sort(os.listdir(join(root_path, 'BraTS-MEN-Train'))):  # 
            for seq in ['t1n', 't1c', 't2w', 't2f']:
                if case in fixed_case_list:
                    img_path = join(root_path, 'BraTS-MEN-Train-Fix', case, case+'-'+seq+'.nii.gz')
                    mask_path = join(root_path, 'BraTS-MEN-Train-Fix', case, case+'-seg.nii.gz')
                else:
                    img_path = join(root_path, 'BraTS-MEN-Train', case, case+'-'+seq+'.nii.gz')
                    mask_path = join(root_path, 'BraTS-MEN-Train', case, case+'-seg.nii.gz')

                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_BraTS2023_MET(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BraTS2023_MET'):
        """
        https://www.synapse.org/Synapse:syn53708126
        """
        
        dataset = 'BraTS2023_MET'
        # labels = ['cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor']
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        data = []
        modality = 'MRI'
        for case in np.sort(os.listdir(join(root_path, 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData'))):  # 
            for seq in ['t1n', 't1c', 't2w', 't2f']:
                img_path = join(root_path, 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData', case, case+'-'+seq+'.nii.gz')
                mask_path = join(root_path, 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData', case, case+'-seg.nii.gz')

                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })
        for case in np.sort(os.listdir(join(root_path, 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional'))):  # 
            for seq in ['t1n', 't1c', 't2w', 't2f']:
                img_path = join(root_path, 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional', case, case+'-'+seq+'.nii.gz')
                mask_path = join(root_path, 'ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData_Additional', case, case+'-seg.nii.gz')

                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_BraTS2023_PED(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BraTS2023_PED'):
        """
        https://www.synapse.org/Synapse:syn53708126
        """
        
        dataset = 'BraTS2023_PED'
        # labels = ['cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor']
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        data = []
        modality = 'MRI'
        for case in np.sort(os.listdir(join(root_path, 'ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData'))):  # 
            for seq in ['t1n', 't1c', 't2w', 't2f']:
                img_path = join(root_path, 'ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData', case, case+'-'+seq+'.nii.gz')
                mask_path = join(root_path, 'ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData', case, case+'-seg.nii.gz')
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_BraTS2023_SSA(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BraTS2023_SSA'):
        """
        https://www.synapse.org/Synapse:syn53708126
        """
        
        dataset = 'BraTS2023_SSA'
        # labels = ['cerebral edema', 'non-enhancing brain tumor', 'enhancing brain tumor']
        labels = [
            "necrotic brain tumor core",
            "brain edema",
            "enhancing brain tumor",
            ]
        data = []
        modality = 'MRI'
        for case in np.sort(os.listdir(join(root_path, 'ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'))):  # 
            for seq in ['t1n', 't1c', 't2w', 't2f']:
                img_path = join(root_path, 'ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2', case, case+'-'+seq+'.nii.gz')
                mask_path = join(root_path, 'ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2', case, case+'-seg.nii.gz')
                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_BTCV_Cervix(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/BTCV_Cervix'):
        """
        https://www.synapse.org/Synapse:syn3378972
        """
        
        dataset = 'BTCV_Cervix'
        labels = [
            "urinary bladder",
            "uterus",
            "rectum",
            "small bowel",
            ]
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'images'))) if x.endswith('.nii.gz')]:  # 
            img_path = join(root_path, 'images', case)
            mask_path = join(root_path, 'labels', case.replace('avg', 'avg_seg'))
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'unknown',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_SEGA(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/SEGA'):
        """
        https://multicenteraorta.grand-challenge.org/data/
        """
        
        dataset = 'SEGA'
        labels = [
            "aorta",
            ]
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'Dongyang')))]:  # 
            img_path = join(root_path, 'Dongyang', case, case+'.nrrd')
            img = sitk.ReadImage(img_path)
            img_path = img_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(img, img_path) 

            mask_path = join(root_path, 'Dongyang', case, case+'.seg.nrrd')
            mask = sitk.ReadImage(mask_path)
            mask_path = mask_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(mask, mask_path)

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })

        for case in [x for x in np.sort(os.listdir(join(root_path, 'KiTS')))]:  # 
            img_path = join(root_path, 'KiTS', case, case+'.nrrd')
            img = sitk.ReadImage(img_path)
            img_path = img_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(img, img_path) 
            img = nib.load(img_path)
            affine = img.affine
            header = img.header
            img = img.get_fdata() - 1000
            nib.save(nib.Nifti1Image(img, affine, header), img_path)

            mask_path = join(root_path, 'KiTS', case, case+'.seg.nrrd')
            mask = sitk.ReadImage(mask_path)
            mask_path = mask_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(mask, mask_path)

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })

        for case in [x for x in np.sort(os.listdir(join(root_path, 'Rider')))]:  # 
            img_path = join(root_path, 'Rider', case, case.split(' ')[0]+'.nrrd')
            img = sitk.ReadImage(img_path)
            img_path = img_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(img, img_path) 
            img = nib.load(img_path)
            affine = img.affine
            header = img.header
            img = img.get_fdata() - 1000
            nib.save(nib.Nifti1Image(img, affine, header), img_path)

            mask_path = join(root_path, 'Rider', case, case.split(' ')[0]+'.seg.nrrd')
            mask = sitk.ReadImage(mask_path)
            mask_path = mask_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(mask, mask_path)

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_Pancreas_CT(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/Pancreas_CT'):
        """
        https://wiki.cancerimagingarchive.net/display/public/pancreas-ct
        """
        
        dataset = 'Pancreas_CT'
        labels = [
            "pancreas",
            ]
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'images'))) if x.startswith('PAN')]:  # 
            for dcm_path in Path(join(root_path, 'images', case)).glob('*/*'):
                dicom2nifti.convert_directory(dcm_path, join(root_path, 'images', case), compression=True)

            img_path = join(root_path, 'images', case, 'none_pancreas.nii.gz')
            mask_path = join(root_path, 'labels', 'label'+case.split('_')[1]+'.nii.gz')

            affine = nib.load(mask_path).affine
            img = nib.load(img_path)
            img = nib.Nifti1Image(img.get_fdata()[:, :, ::-1], affine, img.header)
            nib.save(img, join(root_path, 'images', case, 'none_pancreas.nii.gz'))
            
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'unknown',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_FUMPE(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/FUMPE'):
        """
        https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images
        """
        
        dataset = 'FUMPE'
        labels = [
            "pulmonary embolism",
            ]
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'CT_scans'))) if x.startswith('PAT')]:  # 
            dicom2nifti.convert_directory(join(root_path, 'CT_scans', case), join(root_path, 'CT_scans', case), compression=True)

            img_path = join(root_path, 'CT_scans', case, [filename for filename in os.listdir(join(root_path, 'CT_scans', case)) if filename.endswith('.nii.gz')][0])

            mask_path = join(root_path, 'GroundTruth', case+'.mat')

            img = nib.load(img_path)
            mask = nib.Nifti1Image(np.rot90(scipy.io.loadmat(mask_path)['Mask'], -1, axes=(0, 1)), img.affine, img.header)
            nib.save(mask, join(root_path, 'GroundTruth', case+'.nii.gz'))
            mask_path = join(root_path, 'GroundTruth', case+'.nii.gz')
            
            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'unknown',
                        'patient_id':case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
      
    def preprocess_VerSe(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/VerSe'):
        """
        https://github.com/anjany/verse
        """
        
        dataset = 'VerSe'
        labels = [
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
            "lumbar vertebrae 6 (l6)",
            ]
        data = []
        modality = 'CT'
        verse19_case_list = [x for x in np.sort(os.listdir(join(root_path, 'dataset-verse19training', 'rawdata')))]
        for case in verse19_case_list:  # 
            img_path = join(root_path, 'dataset-verse19training', 'rawdata', case, case+'_ct.nii.gz')
            mask_path = join(root_path, 'dataset-verse19training', 'derivatives', case, case+'_seg-vert_msk.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })

        verse20_case_list = [x for x in np.sort(os.listdir(join(root_path, 'dataset-01training', 'rawdata')))]
        for case in verse20_case_list:  # 
            if case in verse19_case_list:
                continue
            img_path = join(root_path, 'dataset-01training', 'rawdata', case, case+'_dir-ax_ct.nii.gz')
            mask_path = join(root_path, 'dataset-01training', 'derivatives', case, case+'_dir-ax_seg-vert_msk.nii.gz')
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'train',
                'patient_id':case,
            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_PDDCA(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/PDDCA'):
        """
        https://www.imagenglab.com/newsite/pddca/
        """
        
        dataset = 'PDDCA'
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
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path))) if x.startswith('0522')]:  # 
            img_path = join(root_path, case, 'img.nrrd')
            img = sitk.ReadImage(img_path)
            img_path = img_path.replace('nrrd', 'nii.gz')
            sitk.WriteImage(img, img_path) 

            img = nib.load(img_path)
            mask = np.zeros(img.get_fdata().shape, dtype=np.uint8)

            for i, structure in enumerate(['BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R', 'Submandibular_L', 'Submandibular_R'], start=1):
                tmp_mask_path = join(root_path, case, 'structures', structure+'.nrrd')
                if os.path.exists(tmp_mask_path):
                    tmp_mask = sitk.ReadImage(tmp_mask_path)
                    tmp_mask_path = tmp_mask_path.replace('nrrd', 'nii.gz')
                    sitk.WriteImage(tmp_mask, tmp_mask_path)
                    tmp_mask = nib.load(tmp_mask_path).get_fdata().astype(np.uint8)
                    mask[tmp_mask==1] = i

            mask_path = img_path.replace('img.nii.gz', 'mask.nii.gz')
            mask = nib.Nifti1Image(mask, img.affine, img.header)
            nib.save(mask, mask_path)
            
            data.append({
                'image':img_path,
                'mask':mask_path,
                'label':labels,
                'modality':modality,
                'dataset':dataset,
                'official_split': 'unknown',
                'patient_id':case,
            })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 
                
    def preprocess_LNDb(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/LNDb'):
        """
        https://zenodo.org/record/7153205#.Yz_oVHbMJPZ
        """
        
        dataset = 'LNDb'
        labels = [
            "lung nodule",
            ]
        data = []
        modality = 'CT'
        for subset in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
            for case in [x for x in np.sort(os.listdir(join(root_path, subset))) if x.endswith('.mhd')]:  # 
                img_path = join(root_path, subset, case)
                img = sitk.ReadImage(img_path)
                img_path = img_path.replace('mhd', 'nii.gz')
                sitk.WriteImage(img, img_path) 

                mask_path_list = Path(join(root_path, 'masks')).glob(case.split('.')[0]+'*.mhd')
                for mask_path in mask_path_list:
                    mask = sitk.ReadImage(mask_path)
                    mask_path = str(mask_path).replace('mhd', 'nii.gz')
                    sitk.WriteImage(mask, mask_path)

                img = nib.load(img_path)
                mask = np.zeros(img.get_fdata().shape, dtype=np.uint8)
                mask_path_list = Path(join(root_path, 'masks')).glob(case.split('.')[0]+'_rad*.nii.gz')
                for mask_path in mask_path_list:
                    tmp_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
                    mask[tmp_mask>0] = 1
                mask_path = join(root_path, 'masks', case.replace('mhd', 'nii.gz'))
                nib.save(nib.Nifti1Image(mask, img.affine, img.header), mask_path)

                data.append({
                            'image':img_path,
                            'mask':mask_path,
                            'label':labels,
                            'modality':modality,
                            'dataset':dataset,
                            'official_split': 'train',
                            'patient_id':case,
                            })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')

    def preprocess_SegRap2023_Task1(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/SegRap2023'):
        """
        https://segrap2023.grand-challenge.org
        """
        
        dataset = 'SegRap2023_Task1'
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
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'SegRap2023_Training_Set_120cases'))) if x.startswith('segrap')]:  # 
            img_path = join(root_path, 'SegRap2023_Training_Set_120cases', case, 'image_contrast.nii.gz')
            mask_path = join(root_path, 'SegRap2023_Training_Set_120cases_OneHot_Labels', 'Task001', case+'.nii.gz')

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id': case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_SegRap2023_Task2(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/SegRap2023'):
        """
        https://segrap2023.grand-challenge.org
        """
        
        dataset = 'SegRap2023_Task2'
        labels = [
            "nasopharyngeal tumor",
            "nasopharyngeal lymph node",
            ]
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'SegRap2023_Training_Set_120cases'))) if x.startswith('segrap')]:  # 
            img_path = join(root_path, 'SegRap2023_Training_Set_120cases', case, 'image_contrast.nii.gz')
            mask_path = join(root_path, 'SegRap2023_Training_Set_120cases_OneHot_Labels', 'Task002', case+'.nii.gz')

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id': case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_CTPelvic1K(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/CTPelvic1K'):
        """
        https://zenodo.org/records/4588403#.YEyLq_0zaCo
        """
        
        dataset = 'CTPelvic1K'
        labels = [
            "sacrum",
            "left hip",
            "right hip",
            "lumbar vertebrae",
            ]
        data = []
        modality = 'CT'
        for case in [x for x in np.sort(os.listdir(join(root_path, 'CTPelvic1K_dataset6_data'))) if x.endswith('nii.gz')]:  # 
            img_path = join(root_path, 'CTPelvic1K_dataset6_data', case)
            mask_path = join(root_path, 'CTPelvic1K_dataset6_Anonymized_mask', case.replace('_data', '_mask_4label'))

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id': case,
                        })

        for case in [x for x in np.sort(os.listdir(join(root_path, 'CTPelvic1K_dataset7_mask'))) if x.endswith('nii.gz')]:  # 
            img_path = join(root_path, 'CTPelvic1K_dataset7_data', case.replace('CLINIC', 'dataset7_CLINIC').replace('_mask_4label', '_data'))
            mask_path = join(root_path, 'CTPelvic1K_dataset7_mask', case)

            data.append({
                        'image':img_path,
                        'mask':mask_path,
                        'label':labels,
                        'modality':modality,
                        'dataset':dataset,
                        'official_split': 'train',
                        'patient_id': case,
                        })

        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
                
    def preprocess_autoPET(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/autoPET'):
        """
        https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287
        
        To save space, crop the image by roi depth +-48
        """
        
        dataset = 'autoPET'
        labels = [
            "tumor",
            ]
        modality = 'PET'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path, 'autoPET_nii')))]
        for case in case_list:
            study_list = [x for x in np.sort(os.listdir(join(root_path, 'autoPET_nii', case)))]
            for study in study_list:
                img_path = join(root_path, 'autoPET_nii', case, study, 'SUV.nii.gz')
                img = nib.load(img_path).get_fdata()
                affine = nib.load(img_path).affine
                header = nib.load(img_path).header
                mask_path = join(root_path, 'autoPET_nii', case, study, 'SEG.nii.gz')
                mask = nib.load(mask_path).get_fdata().astype(np.uint8)
                if np.sum(mask) == 0:
                    continue

                z_min = np.min(np.where(mask>0)[2])
                z_max = np.max(np.where(mask>0)[2])
                z_min = z_min - 48
                z_max = z_max + 48
                z_min = np.max([0, z_min])
                z_max = np.min([mask.shape[2]-1, z_max])

                img = img[..., z_min:z_max+1]
                mask = mask[..., z_min:z_max+1]

                Path(join(root_path, 'autoPET_nii_crop', case, study)).mkdir(exist_ok=True, parents=True)   
                img_path = img_path.replace('autoPET_nii', 'autoPET_nii_crop')
                nib.save(nib.Nifti1Image(img, affine, header), img_path)
                mask_path = mask_path.replace('autoPET_nii', 'autoPET_nii_crop')
                nib.save(nib.Nifti1Image(mask, affine, header), mask_path)

                data.append({
                    'image':img_path,
                    'mask':mask_path,
                    'label':labels,
                    'modality':modality,
                    'dataset':dataset,
                    'official_split': 'train',
                    'patient_id': case,
                })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n') 

    def preprocess_DAP_Atlas(self, root_path='/mnt/hwfile/medai/zhaoziheng/SAM/SAM/DAP_Atlas'):
        """
        https://github.com/alexanderjaus/AtlasDataset
        """
        
        dataset = 'DAP_Atlas'
        labels = [
            "muscle",   # 2
            "fat",
            "abdominal tissue",
            "mediastinal tissue",
            "esophagus",
            "stomach",
            "small bowel",
            "duodenum",
            "colon",    # 10
            # "rectum",
            "gallbladder",
            "liver",
            "pancreas",
            "left kidney",
            "right kidney",
            "urinary bladder",
            "gonad",
            "prostate",
            "uterocervix",  # 20
            "uterus",
            "left breast",
            "right breast",
            "spinal canal",
            "brain",
            "spleen",
            "left adrenal gland",
            "right adrenal gland",
            "left thyroid",
            "right thyroid",    # 30
            "thymus",
            "left gluteus maximus",
            "right gluteus maximus",
            "left gluteus medius",
            "right gluteus medius",
            "left gluteus minimus",
            "right gluteus minimus",
            "left iliopsoas",
            "right iliopsoas",
            "left autochthon",  # 40
            "right autochthon",
            "skin",
            "cervical vertebrae 1 (c1)",
            "cervical vertebrae 2 (c2)",
            "cervical vertebrae 3 (c3)",
            "cervical vertebrae 4 (c4)",
            "cervical vertebrae 5 (c5)",
            "cervical vertebrae 6 (c6)",
            "cervical vertebrae 7 (c7)",
            "thoracic vertebrae 1 (t1)",    # 50
            "thoracic vertebrae 2 (t2)",
            "thoracic vertebrae 3 (t3)",
            "thoracic vertebrae 4 (t4)",
            "thoracic vertebrae 5 (t5)",
            "thoracic vertebrae 6 (t6)",
            "thoracic vertebrae 7 (t7)",
            "thoracic vertebrae 8 (t8)",
            "thoracic vertebrae 9 (t9)",
            "thoracic vertebrae 10 (t10)",
            "thoracic vertebrae 11 (t11)",  # 60
            "thoracic vertebrae 12 (t12)",
            "lumbar vertebrae 1 (l1)",
            "lumbar vertebrae 2 (l2)",
            "lumbar vertebrae 3 (l3)",
            "lumbar vertebrae 4 (l4)",
            "lumbar vertebrae 5 (l5)",
            "left rib 1",
            "right rib 1",
            "left rib 2",
            "right rib 2",  # 70
            "left rib 3",
            "right rib 3",
            "left rib 4",
            "right rib 4",
            "left rib 5",
            "right rib 5",
            "left rib 6",
            "right rib 6",
            "left rib 7",   
            "right rib 7",  # 80
            "left rib 8",
            "right rib 8",
            "left rib 9",
            "right rib 9",
            "left rib 10",
            "right rib 10",
            "left rib 11",
            "right rib 11",
            "left rib 12",
            "right rib 12", # 90
            "rib cartilage",
            "sternum",
            "left clavicle",
            "right clavicle",
            "left scapula",
            "right scapula",
            "left humerus",
            "right humerus",
            "skull",
            "left hip", # 100
            "right hip",
            "sacrum",
            "left femur",
            "right femur",
            "heart",
            "left heart atrium",
            "heart tissue",
            "right heart atrium",
            "myocardium",
            "left heart ventricle", # 110
            "right heart ventricle",
            "left iliac artery",
            "right iliac artery",
            "aorta",
            "left iliac vena",
            "right iliac vena",
            "inferior vena cava",
            "portal vein and splenic vein",
            "celiac trunk",
            "left lung lower lobe", # 120
            "left lung upper lobe",
            "right lung lower lobe",
            "right lung middle lobe",
            "right lung upper lobe",
            "bronchie",
            "trachea",
            "pulmonary artery",
            "left cheek",
            "right cheek",
            "left eyeball", # 130
            "right eyeball",
            "nasal cavity",
            "right common carotid artery",  # 133
            "left common carotid artery",
            "manubrium of sternum",
            "right internal carotid artery",    # 136
            "left internal carotid artery",
            "right internal jugular vein",  # 138
            "left internal jugular vein",
            "brachiocephalic trunk",    # 140
            "right brachiocephalic vein",   # 141
            "left brachiocephalic vein",
            "right subclavian artery",  # 143
            "left subclavian artery"
            ]
        modality = 'CT'
        data = []
        
        case_list = [x for x in np.sort(os.listdir(join(root_path.replace('DAP_Atlas', 'autoPET/autoPET_nii'))))]
        for case in case_list:
            study_list = [x for x in np.sort(os.listdir(join(root_path.replace('DAP_Atlas', 'autoPET/autoPET_nii'), case)))]
            for study in study_list:
                img_path = join(root_path.replace('DAP_Atlas', 'autoPET/autoPET_nii'), case, study, 'CT.nii.gz')
                mask_path = join(root_path, 'Atlas_final_dataset_V1_533', 'AutoPET_'+case.split('_')[1]+'_'+study.split('-')[-1]+'.nii.gz')
                data.append({
                    'image':img_path,
                    'mask':mask_path,
                    'label':labels,
                    'modality':modality,
                    'dataset':dataset,
                    'official_split': 'train',
                    'patient_id': case,
                })
        
        Path(self.jsonl_dir).mkdir(exist_ok=True, parents=True)        
        with open(f"{self.jsonl_dir}/{dataset}.jsonl", 'w') as f:
            for datum in data:
                f.write(json.dumps(datum)+'\n')
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name')
    parser.add_argument('--root_path')
    parser.add_argument('--jsonl_dir', default='/mnt/hwfile/medai/zhaoziheng/SAM/processed_files_v1_debug')
    config = parser.parse_args()

    loader = Process_Wrapper(config.jsonl_dir)
    if config.root_path:
        getattr(loader, 'preprocess_'+config.dataset_name)(root_path=config.root_path)
    else:
        getattr(loader, 'preprocess_'+config.dataset_name)()