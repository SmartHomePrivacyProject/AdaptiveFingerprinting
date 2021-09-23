# Adaptive Fingerprinting
This repository contains the code and data for project "Adaptive Fingerprinting". The code can be used to perform website fingerprinting attacks when there are very limited training data (e.g., <= 20 traffic traces per class). Transfer learning, more specifically, adversarial domain adaption, is leveraged in this project to adress the challenges when training data is small. 

**The dataset and code are for research purposes only**. The results of this study are published in the following paper:

Chenggang Wang, Jimmy Dani, Xiang Li, Xiaodong Jia, Boyang Wang, "Adaptive Fingerprinting: Website Fingerprinting over Few Encrypted Traffic," ACM Conference on Data and Application Security and Privacy (**ACM CODASPY 2021**), April, 2021.


## Content
This repository contains separate directories for the attack, defense, website traffic datasets. A brief description of the contents of these directories is below. More detailed usage instrcutions are found in each subfolder's README.md file.

### Attack
THe attack directory contains the code for three transfer learning based attack methods: fine-tuning based method, triplet network based method and our proposed Adapative Fingerprinting method (adversarial domain adaption based method.). The data preparation and other supporting functions are in tools and utility folder.

### Defense
The defense directory contains the code for the WTF-PAD method.

### Dataset
All the four datasets we examined in the paper can be downloaded from the link below (**last modified Sep, 2021**): 

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/Ej4IxFX-ZlVOgH2Ipc-C4p8BK-i_5iCt6bhxMx0xf8RUBg?e=5VMbSy

**Note:** the above link needs to be updated every 6 months due to certain settings of OneDrive. If you find the link is expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will update the link as soon as we can. Thanks!   

### Additional information
The additional_info has information about the structures of the neural networks and its hyperparameters

## Requirements
This project is developed with Python 3.6, tensorflow 2.3 and Keras with Ubuntu 18.04

## Usage
See the details in each subfolder.

## Citation
When reporting results that use the dataset or code in this repository, please cite:

Chenggang Wang, Jimmy Dani, Xiang Li, Xiaodong Jia, Boyang Wang, "Adaptive Fingerprinting: Website Fingerprinting over Few Encrypted Traffic," ACM Conference on Data and Application Security and Privacy (**ACM CODASPY 2021**), April, 2021.

## Contacts
Chenggang Wang, wang2c9@mail.uc.edu, University of Cincinnati

Boyang Wang, boyang.wang@uc.edu, University of Cincinnati
