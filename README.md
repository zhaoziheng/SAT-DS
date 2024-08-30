# SAT-DS

[![Dropbox](https://img.shields.io/badge/Dropbox-Data-blue)](https://www.dropbox.com/scl/fo/gsr7wqh9s5wc2rfsmg08j/AJ98Hfn-FbkroCEXDEIlgkw?rlkey=ubx2nkisroks3vbkopgm3jxyz&st=60l9ybda&dl=0)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2312.17183)
[![Model](https://img.shields.io/badge/GitHub-SAT-blue)](https://github.com/zhaoziheng/SAT)

This is the official repository to build **SAT-DS**, a medical data collection of **72** public segmentation datasets, contains over **22K** 3D images, **302K** segmentation masks and **497** classes from **3** different modalities (MRI, CT, PET) and **8** human body regions. 🚀

Based on this data collection, we build an universal segmentation model for 3D radiology scans driven by text prompts (check this [repo](https://github.com/zhaoziheng/SAT) and our [paper](https://arxiv.org/abs/2312.17183)).

The data collection will continuously growing, stay tuned!

### Hightlight
🎉 To save your time from downloading and preprocess so many datasets, we offer shortcut download links of 42/72 datasets in SAT-DS, which allow re-attribution with licenses such as CC BY-SA. Find them in [dropbox](https://www.dropbox.com/scl/fo/gsr7wqh9s5wc2rfsmg08j/AJ98Hfn-FbkroCEXDEIlgkw?rlkey=ubx2nkisroks3vbkopgm3jxyz&st=60l9ybda&dl=0). 

**All these datasets are preprocessed and packaged by us for your convenience, ready for immediate use upon download and extraction.** Download the datasets you need and unzip them in `data/nii`, these datasets can be used immediately with the paired jsonl files in `data/jsonl`, check Step 3 below for how to use them. Note that we respect and adhere to the licenses of all the datasets, if we incorrectly reattribute any of them, please contact us.

### What we have done in building SAT-DS:
  - Collect as many public datasets as possible for 3D medical segmentation, and compile their basic information;
  - Check and normalize image scans in each dataset, including orientation, spacing and intensity;
  - Check, standardize, and merge the label names for categories in each dataset;
  - Carefully split each dataset into train and test set by the patient id.

### What we offer in this repo:
  - (Step 1) Access to each dataset in SAT-DS.
  - (Step 2) Code to preprocess samples in each dataset.
  - (**Shortcut to skip Step 1 and 2**) [Access](https://www.dropbox.com/scl/fo/gsr7wqh9s5wc2rfsmg08j/AJ98Hfn-FbkroCEXDEIlgkw?rlkey=ubx2nkisroks3vbkopgm3jxyz&st=60l9ybda&dl=0) to preprocessed and packaged datasets that can be used immediately.
  - (Step 3) Code to load samples with normalized image, standardized class names from each dataset.
  - (Step 3) Code to visualize and check the samples.
  - (Step 4) Code to prepare the train and evaluation data for SAT in required format.
  - (Step 5) Code to split the dataset into train and test in consistent with SAT.

### This repo can be used to:
  - (Follow step 1~3) Preprocess and unfied a large-scale and comprehensive 3D medical segmentation data collection, suitable to train or finetune universal segmentation models like SAM2. 
  - (Follow step 1~6) Prepare the training and test data  in required format for [SAT](https://github.com/zhaoziheng/SAT).

Check our paper "One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompts" for more details.

[ArXiv](https://arxiv.org/abs/2312.17183)

[Website](https://zhaoziheng.github.io/SAT/)

![Example Figure](figures/wholebody_demonstration.png)

# Step 1: Download datasets
This is the detailed list of all the datasets and their official download links. Their citation information can be found in `citation.bib` . 

As a shortcut, we preprocess, package and re-attribute some of them for your convenient use. Download them [here](https://www.dropbox.com/scl/fo/gsr7wqh9s5wc2rfsmg08j/AJ98Hfn-FbkroCEXDEIlgkw?rlkey=ubx2nkisroks3vbkopgm3jxyz&st=60l9ybda&dl=0).
| Dataset Name              | Modality | Region        | Classes | Scans | Download link                                                                                      |
|---------------------------|----------|---------------|---------|-------|----------------------------------------------------------------------------------------------------|
| AbdomenCT1K               | CT       | Abdomen       | 4       | 988   | https://github.com/JunMa11/AbdomenCT-1K                                                            |
| ACDC                      | CT       | Thorax        | 4       | 300   | https://humanheart-project.creatis.insa-lyon.fr/database/                                          |
| AMOS CT                   | CT       | Abdomen       | 16      | 300   | https://zenodo.org/records/7262581                                                                 |
| AMOS MRI                  | MRI      | Thorax        | 16      | 60    | https://zenodo.org/records/7262581                                                                 |
| ATLASR2                   | MRI      | Brain         | 1       | 654   | http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html                                          |
| ATLAS                     | MRI      | Abdomen       | 2       | 60    | https://atlas-challenge.u-bourgogne.fr                                                             |
| autoPET                   | PET      | Whole Body    | 1       | 501   | https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287                        |
| Brain Atlas               | MRI      | Brain         | 108     | 30    | http://brain-development.org/                                                                      |
| BrainPTM                  | MRI      | Brain         | 7       | 60    | https://brainptm-2021.grand-challenge.org/                                                         |
| BraTS2023 GLI             | MRI      | Brain         | 4       | 5004  | https://www.synapse.org/#!Synapse:syn51514105                                                      |
| BraTS2023 MEN             | MRI      | Brain         | 4       | 4000  | https://www.synapse.org/#!Synapse:syn51514106                                                      |
| BraTS2023 MET             | MRI      | Brain         | 4       | 951   | https://www.synapse.org/#!Synapse:syn51514107                                                      |
| BraTS2023 PED             | MRI      | Brain         | 4       | 396   | https://www.synapse.org/#!Synapse:syn51514108                                                      |
| BraTS2023 SSA             | MRI      | Brain         | 4       | 240   | https://www.synapse.org/#!Synapse:syn51514109                                                      |
| BTCV Abdomen              | CT       | Abdomen       | 15      | 30    | https://www.synapse.org/#!Synapse:syn3193805/wiki/217789                                           |
| BTCV Cervix               | CT       | Abdomen       | 4       | 30    | https://www.synapse.org/Synapse:syn3378972                                                         |
| CHAOS CT                  | CT       | Abdomen       | 1       | 20    | https://chaos.grand-challenge.org/                                                                 |
| CHAOS MRI                 | MRI      | Abdomen       | 5       | 60    | https://chaos.grand-challenge.org/                                                                 |
| CMRxMotion                | MRI      | Thorax        | 4       | 138   | https://www.synapse.org/#!Synapse:syn28503327/files/                                               |
| Couinaud                  | CT       | Abdomen       | 10      | 161   | https://github.com/GLCUnet/dataset                                                                 |
| COVID-19 CT Seg           | CT       | Thorax        | 4       | 20    | https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark                                               |
| CrossMoDA2021             | MRI      | Head and Neck | 2       | 105   | https://crossmoda.grand-challenge.org/Data/                                                        |
| CT-ORG                    | CT       | Whole Body    | 6       | 140   | https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890                        |
| CTPelvic1K                | CT       | Lower Limb    | 5       | 117   | https://zenodo.org/record/4588403#.YEyLq_0zaCo                                                     |
| DAP Atlas                 | CT       | Whole Body    | 179     | 533   | https://github.com/alexanderjaus/AtlasDataset                                                      |
| FeTA2022                  | MRI      | Brain         | 7       | 80    | https://feta.grand-challenge.org/data-download/                                                    |
| FLARE22                   | CT       | Abdomen       | 15      | 50    | https://flare22.grand-challenge.org/                                                               |
| FUMPE                     | CT       | Thorax        | 1       | 35    | https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images                          |
| HAN Seg                   | CT       | Head and Neck | 41      | 41    | https://zenodo.org/record/                                                                         |
| HECKTOR2022               | PET      | Head and Neck | 2       | 524   | https://hecktor.grand-challenge.org/Data/                                                          |
| INSTANCE                  | CT       | Brain         | 1       | 100   | https://instance.grand-challenge.org/Dataset/                                                      |
| ISLES2022                 | MRI      | Brain         | 1       | 500   | http://www.isles-challenge.org/                                                                    |
| KiPA22                    | CT       | Abdomen       | 4       | 70    | https://kipa22.grand-challenge.org/dataset/                                                        |
| KiTS23                    | CT       | Abdomen       | 3       | 489   | https://github.com/neheller/kits23                                                                 |
| LAScarQS2022 Task 1       | MRI      | Thorax        | 2       | 60    | https://zmiclab.github.io/projects/lascarqs22/data.html                                            |
| LAScarQS2022 Task 2       | MRI      | Thorax        | 1       | 130   | https://zmiclab.github.io/projects/lascarqs22/data.html                                            |
| LNDb                      | CT       | Thorax        | 1       | 236   | https://zenodo.org/record/7153205#.Yz_oVHbMJPZ                                                     |
| LUNA16                    | CT       | Thorax        | 1       | 888   | https://luna16.grand-challenge.org/                                                                |
| MM-WHS CT                 | CT       | Thorax        | 9       | 40    | https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA                                             |
| MM-WHS MR                 | MRI      | Thorax        | 9       | 40    | https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA                                             |
| MRSpineSeg                | MRI      | Spine         | 23      | 91    | https://www.cg.informatik.uni-siegen.de/en/spine-segmentation-and-analysis                         |
| MSD Cardiac               | MRI      | Thorax        | 1       | 20    | http://medicaldecathlon.com/                                                                       |
| MSD Colon                 | CT       | Abdomen       | 1       | 126   | http://medicaldecathlon.com/                                                                       |
| MSD HepaticVessel         | CT       | Abdomen       | 2       | 303   | http://medicaldecathlon.com/                                                                       |
| MSD Hippocampus           | MRI      | Brain         | 3       | 260   | http://medicaldecathlon.com/                                                                       |
| MSD Liver                 | CT       | Abdomen       | 2       | 131   | http://medicaldecathlon.com/                                                                       |
| MSD Lung                  | CT       | Thorax        | 1       | 63    | http://medicaldecathlon.com/                                                                       |
| MSD Pancreas              | CT       | Abdomen       | 2       | 281   | http://medicaldecathlon.com/                                                                       |
| MSD Prostate              | MRI      | Pelvis        | 2       | 64    | http://medicaldecathlon.com/                                                                       |
| MSD Spleen                | CT       | Abdomen       | 1       | 41    | http://medicaldecathlon.com/                                                                       |
| MyoPS2020                 | MRI      | Thorax        | 6       | 135   | https://mega.nz/folder/BRdnDISQ#FnCg9ykPlTWYe5hrRZxi-w                                             |
| NSCLC                     | CT       | Thorax        | 2       | 85    | https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68551327                        |
| Pancreas CT               | CT       | Abdomen       | 1       | 80    | https://wiki.cancerimagingarchive.net/display/public/pancreas-ct                                   |
| Parse2022                 | CT       | Thorax        | 1       | 100   | https://parse2022.grand-challenge.org/Dataset/                                                     |
| PDDCA                     | CT       | Head and Neck | 12      | 48    | https://www.imagenglab.com/newsite/pddca/                                                          |
| PROMISE12                 | MRI      | Pelvis        | 1       | 50    | https://promise12.grand-challenge.org/Details/                                                     |
| SEGA                      | CT       | Whole Body    | 1       | 56    | https://multicenteraorta.grand-challenge.org/data/                                                 |
| SegRap2023 Task1          | CT       | Head and Neck | 61      | 120   | https://segrap2023.grand-challenge.org/                                                            |
| SegRap2023 Task2          | CT       | Thorax        | 2       | 120   | https://segrap2023.grand-challenge.org/                                                            |
| SegTHOR                   | CT       | Thorax        | 4       | 40    | https://competitions.codalab.org/competitions/21145#learn_the_details                              |
| SKI10                     | CT       | Upper Limb    | 4       | 99    | https://ambellan.de/sharing/QjrntLwah                                                              |
| SLIVER07                  | CT       | Abdomen       | 1       | 20    | https://sliver07.grand-challenge.org/                                                              |
| ToothFairy                | MRI      | Head and Neck | 4       | 153   | https://ditto.ing.unimore.it/toothfairy/                                                           |
| TotalSegmentator Cardiac  | CT       | Whole Body    | 17      | 1202  | https://zenodo.org/record/6802614                                                                  |
| TotalSegmentator Muscles  | CT       | Whole Body    | 31      | 1202  | https://zenodo.org/record/6802614                                                                  |
| TotalSegmentator Organs   | CT       | Whole Body    | 24      | 1202  | https://zenodo.org/record/6802614                                                                  |
| TotalSegmentator Ribs     | CT       | Whole Body    | 39      | 1202  | https://zenodo.org/record/6802614                                                                  |
| TotalSegmentator Vertebrae| CT       | Whole Body    | 29      | 1202  | https://zenodo.org/record/6802614                                                                  |
| TotalSegmentator V2       | CT       | Whole Body    | 24      | 1202  | https://zenodo.org/record/6802614                                                                  |
| VerSe                     | CT       | Whole Body    | 29      | 96    | https://github.com/anjany/verse                                                                    |
| WMH                       | MRI      | Brain         | 1       | 170   | https://wmh.isi.uu.nl/                                                                             |
| WORD                      | CT       | Abdomen       | 18      | 150   | https://github.com/HiLab-git/WORD                                                                  |

# Step 2: Preprocess datasets
For each dataset, we need to find all the image and mask pairs, and another 5 basic information: dataset name, modality, label name, patient ids (to split train-test set) and official split (if provided). \
In `processor.py`, we customize the process procedure for each dataset, to generate a jsonl file including these information for each sample. \
Take AbdomenCT1K for instance, you need to run the following command:
```
python processor.py \
--dataset_name AbdomenCT1K \
--root_path 'SAT-DS/data/nii/AbdomenCT-1K' \
--jsonl_dir 'SAT-DS/data/jsonl'
```
`root_path` should be where you download and place the data, `jsonl_dir` should be where you plan to place the jsonl files. \
⚠️ Note the `dataset_name` and the name in the table might not be exactly the same. For specific details, please refer to each process function in `processor.py`. \
After process, each sample in jsonl files would be like:
```
{
  'image' :"SAT-DS/data/nii/AbdomenCT-1K/Images/Case_00558_0000.nii.gz",
  'mask': "SAT-DS/data/nii/AbdomenCT-1K/Masks/Case_00558.nii.gz",
  'label': ["liver", "kidney", "spleen", "pancreas"],
  'modality': 'CT',
  'dataset': 'AbdomenCT1K,
  'official_split': 'unknown',
  'patient_id': 'Case_00558_0000.nii.gz',
}
```
Note that in this step, we may convert the image and mask into new nifiti files for some datasets, such as TotalSegmentator and so on. So it may take some time.

# Shortcut to skip Step 1 and 2: Download the preprocessed and packaged data for immediate use
We offer shortcut download links of 42 datasets in [dropbox](https://www.dropbox.com/scl/fo/gsr7wqh9s5wc2rfsmg08j/AJ98Hfn-FbkroCEXDEIlgkw?rlkey=ubx2nkisroks3vbkopgm3jxyz&st=60l9ybda&dl=0). All these datasets are preprocessed and packaged in advance. Download the datasets you need and unzip them in `data/nii`, each dataset is paired with a jsonl file in `data/jsonl`.

# Step 3: Load data with unified normalization
With the generated jsonl file, a dataset is now ready to be used. \
However, when mixing all the datasets to train a universal segmentation model, we need to **apply normalization on the image intensity, orientation, spacing across all the datasets, and adjust labels if necessary.** \
We realize this by customizing the load script for each dataset in `loader.py`, this is a simple demo how to use it in your code:
```
from loader import Loader_Wrapper

loader = Loader_Wrapper()
    
# load samples from jsonl
with open('SAT-DS/data/jsonl', 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

# load a sample
for sample in data:
    batch = getattr(loader, func_name)(sample)
    img_tensor, mc_mask, text_ls, modality, image_path, mask_path = batch
```
**For each sample, whatever the dataset it comes from, the loader will give output in a normalized format**:
```
img_tensor  # tensor with shape (1, H, W, D)
mc_mask  # binary tensor with shape (N, H, W, D), one channel for each class;
text_ls  # a list of N class name;
modality  # MRI, CT or PET;
image_path  # path to the loaded mask file;
mask_path  # path to the loaded imag file;
```
⚠️ Note that we may merge and adjust labels here in the loader. Therefore, the output `text_ls` may be different from the `label` you see in the input jsonl file. 
Here is an case where we merge `left kidney' and `right kidney` for a new label `kidney` when loading examples from CHAOS_MRI:
```
kidney = mask[1] + mask[2]
mask = torch.cat((mask, kidney.unsqueeze(0)), dim=0)
labels.append("kidney")
```
And here is another case where we adjust the annotation of `kidney` by integrating the annotation of `kidney tumor` and `kidney cyst`:
```
mc_masks[0] += mc_masks[1]
mc_masks[0] += mc_masks[2]
```

We also offer the shortcut to visualize and check any sample in any dataset after normalization. For example, to visualize the first sample in AbdomenCT1K.jsonl, just run the following command:
```
python loader.py \
--visualization_dir 'SAT-DS/data/visualization' \
--path2jsonl 'SAT-DS/data/jsonl/AbdomenCT1K.jsonl' \
--i 0
```

# (Optional) Step 4: Convert to npy files
For convenience, before training SAT, we normalize all the data according to step 3, and convert the images and segmentation masks to npy files. If you try to use our training code, run this command for each dataset:
```
python convert_to_npy.py \
--jsonl2load 'SAT-DS/data/jsonl/AbdomenCT1K.jsonl' \
--jsonl2save 'SAT-DS/data/jsonl/AbdomenCT1K.jsonl'
```
The converted npy files will be saved in `preprocessed_npy/dataset_name`, and some new information will be added to the jsonl file for connivence to load the npy files.

# (Optional) Step 5: Split train and test set
We offer the train-test split used in our paper for each dataset in json files. To follow our split and benchmark your method, simply run this command:
```
python train_test_split.py \
--jsonl2split 'SAT-DS/data/jsonl/AbdomenCT1K.jsonl' \
--train_jsonl 'SAT-DS/data/trainset_jsonl/AbdomenCT1K.jsonl' \
--test_jsonl 'SAT-DS/data/testset_jsonl/AbdomenCT1K.jsonl' \
--split_json 'SAT-DS/data/split_json/AbdomenCT1K.json'
```
This will split the jsonl file into train and test. 

Or, if you want to re-split them, just customize your split by identifying the `patient_id` in the json file (``patient_id`` of each sample can be found in jsonl file of each dataset):
```
{'train':['train_patient_id1', ...], 'test':['test_patient_id1', ...]}
```

# (Optional) Step 6: DIY your data collection
You may want to customize the dataset collection in training your model, simply merge the train jsonls of the data you want to involve. For example, merge the jsonls for all the 72 datasets into `train.jsonl`, and you can use them together to train SAT, using our training code in this [repo](https://github.com/zhaoziheng/SAT). 

Similarly, you can customize a benchmark with arbitrary datasets you want by merging the test jsonls.

# Citation
If you use this code for your research or project, please cite:
```
@arxiv{zhao2023model,
  title={One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompt}, 
  author={Ziheng Zhao and Yao Zhang and Chaoyi Wu and Xiaoman Zhang and Ya Zhang and Yanfeng Wang and Weidi Xie},
  year={2023},
  journal={arXiv preprint arXiv:2312.17183},
}
```
And if you use any of these datasets in SAT-DS, please cite the corresponding papers. A summerized citation information can be found in `citation.bib` .
