# A Large-scale Multi Domain Leukemia Dataset for the White Blood Cells Detection with Morphological Attributes for Explainability

![architecture_AttriDet](https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset/assets/155678287/e2004432-3411-4eea-bc27-cf2a6a6daab9)


**Authors:** Abdul Rehman, Talha Meraj, Aiman Mahmood Minhas, Ayisha Imran, Mohsen Ali, Waqas Sultani

**MICCAI 2024**

**Paper:** [ArXiv](https://arxiv.org/abs/2405.10803)

**Abstract:** _Earlier diagnosis of Leukemia can save thousands of lives annually. The prognosis of leukemia is challenging without the morphological information of White Blood Cells (WBC) and relies on the accessibility of expensive microscopes and the availability of hematologists to analyze Peripheral Blood Samples (PBS). Deep Learning based methods can be employed to assist hematologists. However, these algorithms require a large amount of labeled data, which is not readily available. To overcome this limitation, we have acquired a realistic, generalized, and large dataset. To collect this comprehensive dataset for real-world applications, two microscopes from two different cost spectrums (high-cost HCM and low-cost LCM) are used for dataset capturing at three magnifications (100x, 40x, 10x) through different sensors (high-end camera for HCM, middle-level camera for LCM and mobile-phone camera for both). The high-sensor camera is 47 times more expensive than the middle-level camera and HCM is 17 times more expensive than LCM. In this collection, using HCM at high resolution (100x), experienced hematologists annotated 10.3k WBC types (14) and artifacts, having 55k morphological labels (Cell Size, Nuclear Chromatin, Nuclear Shape, etc.) from 2.4k images of several PBS leukemia patients. Later on, these annotations are transferred to other 2 magnifications of HCM, and 3 magnifications of LCM, and on each camera captured images. Along with the LeukemiaAttri dataset, we provide baselines over multiple object detectors and Unsupervised Domain Adaptation (UDA) strategies, along with morphological information-based attribute prediction. The dataset will be publicly available after publication to facilitate the research in this direction._

# Installation

We recommend the use of a Linux machine equipped with CUDA compatible GPUs. The execution environment can be installed through Conda.

Clone repo:
```
git clone https://github.com/AttriDet/AttriDet
cd AttriDet
```
 
Conda
Install requirements.txt in a Python>=3.7.16 environment, require PyTorch version 1.13.1 with CUDA version 11.7 support. The environment can be installed and activated with:
```
conda create --name AttriDet python=3.7.16
conda activate AttriDet
pip install -r requirements.txt  # install
```

# Dataset 
LeukemiaAttri dataset can be downloaded from the given link:

[H_100x_C1](https://drive.google.com/drive/folders/1GTmefJJQyVaZ3qaCdfhvryWX9kNdKP80?usp=sharing)

[H_100x_C2](https://drive.google.com/drive/folders/1gdP_zikQ8Bo52-nQXBky6LdVCqJmzErN?usp=sharing)

[L_100x_C1](https://drive.google.com/drive/folders/1W1CJdvGoTuF9VkGnXURli0fVZLrTaBbD?usp=sharing)

[L_100x_C2](https://drive.google.com/drive/folders/11pMj05Fba6fj6CJnDSXOUEtMxj4zkc9R?usp=sharing)

[H_40x_C1](https://drive.google.com/drive/folders/1b0ow0uqp32WvRMalrJWfDswxVCjMvPps?usp=sharing)

[H_40x_C2](https://drive.google.com/drive/folders/1o0DiZcsoFI4mMpQMpvsDkU4gm_9ST4rY?usp=sharing)

[L_40x_C2](https://drive.google.com/drive/folders/1Xpbig7gVwfGi4mruBA8JGpddlO8njeeJ?usp=sharing)


# JSON COCO Format
```
|-COCO Dataset
      |---Annotations
                     |---train.json
                     |---test.json
      |---Images
                |---train
                |---test
```

# YOLO Format

We construct the training and testing set for the yolo format settings, dataset can be downloaded from:

labels prepared in YOLO format but with attributes information as: cls x y w h a1 a2 a3 a4 a5 a6 whereas standard yolo format of labels was cls x y w h 

data -> WBC_v1.yaml
```
train: ../images/train
test: ../images/test


# number of classes
nc: 14

# class names
names: ["None","Myeloblast","Lymphoblast", "Neutrophil","Atypical lymphocyte","Promonocyte","Monoblast","Lymphocyte","Myelocyte","Abnormal promyelocyte", "Monocyte","Metamyelocyte","Eosinophil","Basophil"]
```

# Training
To reproduce the experimental result, we recommend training the model with the following steps.

Before training, please check data/WBC_v1.yaml, and enter the correct data paths.

The model is trained in 2 successive phases:

Phase 1: Model pre-train # 100 Epochs

Phase 2: Pre-trained weights used for further training # 200 Epochs


# Phase 1: Model pre-train
The first phase of training consists in the pre-training of the model. Training can be performed by running the following bash script:

```
python train.py \
 --name AttriDet_Phase1 \
 --batch 8 \
 --imgsz 640 \
 --epochs 100 \
 --data data/WBC_v1.yaml \
 --hyp data/hyps/hyp.scratch-high.yaml
 --weights yolov5x.pt
```

# Phase 2: Pre-trained weights used for further training 
The Pre-trained weights used for further training. Training can be performed by running the following bash script:

```
python train.py \
 --name AttriDet_Phase2 \
 --batch 8 \
 --imgsz 640 \
 --epochs 200 \
 --data data/WBC_v1.yaml \
 --hyp data/hyps/hyp.scratch-high.yaml
 --weights runs/AttriDet_Phase1/weights/last.pt
```

# Testing phase
once model training will be done, an Attribute_model directory will be created, containing ground truth vs predicted attributes csv files, additionally it will contain the attribute model weights saved with f1 score as best weights whereas last.pt will also be saved. These files and weights will be saved based on validation of model. To get model testing, the last.pt of YOLO and last.pt of attribute model will be used to run the test.py file. In result, in Attribute_model directory, a test subdirectory will be created, containing test.csv of ground truth vs predicted attributes. The yolo weights and testing will be save correspondingly in runs/val/exp.

```
python test.py \
 --weights /runs/train/AttriDet/weights//last.pt,
 --data, data/WBC_v1.yaml, 
 --save-csv,
 --imgsz,640
```
