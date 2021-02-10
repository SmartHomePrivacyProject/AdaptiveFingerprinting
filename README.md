# AdaptiveFingerprinting
This repository contain the code and data for project "Adaptive Fingerprinting". The attack is a privacy leakage attack which utilize transfer learning techniques that can allows a passive adversary to perform website fingerprinting attack with very limited training data (<= 20 samples per class).

The dataset and code are for research purposes only. The results of this study are published in the following paper:


## Content
This repository contains separate directories for the attack, defense, website traffic datasets. A brief description of the contents of these directories is below. More detailed usage instrcutions are found in each subfolder's README.md file.

### Attack
THe attack directory contains the code for three transfer learning based attack methods: fine-tuning based method, triplet network based method and our proposed Adapative Fingerprinting method (adversarial domain adaption based method.). The data preparation and other supporting functions are in tools and utility folder.

### Defense
The defense directory contains the code for the WTF-PAD method.

### Dataset
The datasets has infomation about where you can find and download the datasets

### Additional information
The additional_info has information about the structures of the neural networks and its hyperparameters

## Requirements
This project is developed with Python 3.6, tensorflow 2.3 and Keras with Ubuntu 18.04

## Usage
See the details in each subfolder.

## Citation
When reporting results that use the dataset or code in this repository, please cite:

Chenggang Wang, Jimmy Dani, Xiang Li, Xiaodong Jia, Boyang Wang, "Adaptive Fingerprinting: Website Fingerprinting over Few Encrypted Traffic," ACM Conference on Data and Application Security and Privacy (ACM Codaspy 2021), July, 2020.

## Contacts
Chenggang Wang, wang2c9@mail.uc.edu, University of Cincinnati

Boyang Wang, boyang.wang@uc.edu, University of Cincinnati
