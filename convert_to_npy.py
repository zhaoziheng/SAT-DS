import os
import math
import json
import traceback

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import shutil

def convert_to_npy(jsonl2load, jsonl2save, process_image=True, process_mask=True, overwrite=True): 
    with open(jsonl2load, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    for sample_idx, datum in tqdm(enumerate(data), desc=f"processing {data[0]['dataset']} ..."):
        
        dataset_name = datum['dataset']
                
        # if already reid
        if 'sample_id' in datum:
            sample_idx = datum['sample_id']
        elif 'renorm_segmentation_dir' in datum:
            sample_idx = datum['renorm_segmentation_dir'].split('/')[-1]
        elif 'renorm_image' in datum:
            sample_idx =  datum['renorm_image'].split('/')[-1][:-4]
            
        datum['sample_id'] = sample_idx
        
        try:
            tmp = getattr(loader, dataset_name)(datum)
            image = tmp[0]
            mask = tmp[1]
            label_ls = tmp[2]
            _, h, w, d = image.shape
            n, mh, mw, md = mask.shape
            assert h==mh and w==mw and d==md
            assert len(label_ls) == n
        except:
            info = traceback.format_exc()
            with open('preprocess_to_npy_errors.txt', 'a') as f:
                f.write(info + '\n\n')
            continue
        
        if process_image:
            if 'TotalSegmentator' not in dataset_name:
                image_root = f'preprocessed_npy/{dataset_name}/renorm_image'
            else:
                image_root = f'preprocessed_npy/TotalSegmentator/renorm_image'
            Path(image_root).mkdir(exist_ok=True, parents=True)
            image_path = f'{image_root}/{sample_idx}.npy'
            if not os.path.exists(image_path) or overwrite:
                np.save(image_path, image)
            datum['renorm_image'] = image_path
            c, h, w, d = image.shape
            datum['chwd'] = [c, h, w, d]
        
        if process_mask:
            segmentation_root = f'preprocessed_npy/{dataset_name}/renorm_segmentation'
            segmentation_dir = f'{segmentation_root}/{sample_idx}'
            Path(segmentation_dir).mkdir(exist_ok=True, parents=True)
            # crop and save mask of each single label
            y1x1z1_y2x2z2 = []
            for label_idx in range(mask.shape[0]):
                label_mask = mask[label_idx, :, :, :]   # h w d
                non_zero_coordinates = torch.nonzero(label_mask, as_tuple=True) # ([...], [...], [...])
                if non_zero_coordinates[0].numel()==0:
                    y1x1z1_y2x2z2.append(False)
                else:
                    y1, x1, z1 = torch.min(non_zero_coordinates[0]).item(), torch.min(non_zero_coordinates[1]).item(), torch.min(non_zero_coordinates[2]).item()
                    y2, x2, z2 = torch.max(non_zero_coordinates[0]).item(), torch.max(non_zero_coordinates[1]).item(), torch.max(non_zero_coordinates[2]).item()
                    y1x1z1_y2x2z2.append([y1, x1, z1, y2+1, x2+1, z2+1])
                    label = label_ls[label_idx]
                    non_empty_mask = label_mask[y1:y2+1, x1:x2+1, z1:z2+1].bool()
                    segmentation_path = f'{segmentation_dir}/{label}.npy'
                    if not os.path.exists(segmentation_path) or overwrite:
                        np.save(segmentation_path, non_empty_mask)
            mask = torch.sum(mask, dim=0)   # nhwd -> hwd
            non_zero_coordinates = torch.nonzero(mask, as_tuple=True) # ([...], [...], [...])
            if non_zero_coordinates[0].numel()==0:
                datum['roi_y1x1z1_y2x2z2'] = False
            else:
                # 否->[y1 x1 z1 y2 x2 z2]
                y1, x1, z1 = torch.min(non_zero_coordinates[0]).item(), torch.min(non_zero_coordinates[1]).item(), torch.min(non_zero_coordinates[2]).item()
                y2, x2, z2 = torch.max(non_zero_coordinates[0]).item(), torch.max(non_zero_coordinates[1]).item(), torch.max(non_zero_coordinates[2]).item()
                datum['roi_y1x1z1_y2x2z2'] = [y1, x1, z1, y2+1, x2+1, z2+1]
            
            datum['label'] = label_ls        
            datum['renorm_segmentation_dir'] = segmentation_dir
            datum['renorm_y1x1z1_y2x2z2'] = y1x1z1_y2x2z2
    
    # 保存
    parent_directory = os.path.dirname(jsonl2save)
    os.makedirs(parent_directory, exist_ok=True)
    with open(jsonl2save, 'w') as f:    
        for datum in data:
            f.write(json.dumps(datum)+'\n')
 
if __name__ == '__main__':
    import argparse
    from loader import Loader_Wrapper
    
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
    parser.add_argument('--jsonl2load', type=str)
    parser.add_argument('--jsonl2save', type=str)
    parser.add_argument('--image', type=str2bool, default=True)
    parser.add_argument('--mask', type=str2bool, default=True)
    parser.add_argument('--overwrite', type=str2bool, default=True)
    config = parser.parse_args()
    
    loader = Loader_Wrapper()
    
    # when generate new labels, only update masks (make sure data_loader_renorm robust to additional labels)
    convert_to_npy(config.jsonl2load,
                   config.jsonl2save,
                   config.image,
                   config.mask,
                   config.overwrite)    