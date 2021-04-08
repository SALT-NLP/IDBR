# Information-Disentanglement-Based-Regularization(IDBR)

This repo contains codes for the following paper:

*Yufan Huang <sup>*</sup>, Yanzhe Zhang <sup>*</sup>, Jiaao Chen, Xuezhi Wang, Diyi Yang*: Continual Learning for Text Classification with Information Disentanglement Based Regularization, NAACL2021. 

If you would like to refer to it, please cite the paper mentioned above.

## Getting Started

These instructions will get you running the codes of IDBR.

### Requirements

- Python 3.8.5
- Pytorch 1.4.0
- transformers 3.5.1
- tqdm, sklearn, numpy, pandas

Detailed env is included in ```./package-list.txt```.

### Code Structure
```
|_src/
      |_read_data.py --> Codes for reading and processing datasets
      |_preprocess.py --> Preprocess datasets
      |_model.py --> Codes for baseline and IDBR model
      |_finetune.py --> Codes for finetune Baseline
      |_naivereplay.py --> Codes for naivereplay Baseline
      |_multitasklearning.py --> Codes for multitasklearning Baseline
      |_train.py --> Codes for IDBR
|_data/
      |_ag/
      |_amazon/
      |_dbpedia/
      |_yelp/
      |_yahoo/
      |_data/
```
All folders under ```./data``` will be generated automatically in the "Downloading and Pre-processing the data" step.  

### Downloading and Pre-processing the data

We used the data provided by LAMOL. You can find the data from [link to data](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view). Please download it and put it into the data folder. Then uncompress and pre-process the data:

```
cd ./data
tar -xvzf LAMOL.tar.gz
cd ../src
python preprocess.py
```
### Training models

#### Finetune 

Please run ```./src/finetune.py``` to train the Finetune Baseline model:

```
#Example for length-3 task sequence
python finetune.py --tasks ag yelp yahoo --epochs 4 3 2   

#Example for length-5 task sequence
python finetune.py --tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1   
```

#### Naive Replay 

Please run ```./src/naivereplay.py``` to train the Naive Replay Baseline model:

```
#Example for length-3 task sequence
python naivereplay.py --tasks ag yelp yahoo --epochs 4 3 2   

#Example for length-5 task sequence
python naivereplay.py --tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 
```

#### Information-Disentanglement-Based-Regularization  

Please run ```./src/train.py``` to train the IDBR model: 

```
#IDBR + kmeans for sample selection
python train.py --tasks ag yelp yahoo --epochs 4 3 2 --disen True --reg 1 --reggen 0.5 --regspe 0.5 --kmeans True

#Reg-only + kmeans for sample selection
python train.py --tasks ag yelp yahoo --epochs 4 3 2 --reg 1 --reggen 0.5 --regspe 0.5 --kmeans True --tskcoe 0.0

#IDBR + random sampling for sample selection
python train.py --tasks ag yelp yahoo --epochs 4 3 2 --disen True --reg 1 --reggen 0.5 --regspe 0.5 

#IDBR + store ratio=1 + kmeans for sample selection
python train.py --tasks ag yelp yahoo --epochs 1 1 1 --disen True --reg 1 --reggen 0.5 --regspe 0.5 --kmeans True --store_ratio 1
```

#### Multitask Learning 

Please run ```./src/multitasklearning.py``` to train the multitask-learning model:

```
#Multitask Learning w. Information Disentanglement
python multitasklearning.py --tasks ag yelp yahoo --disen True

#Multitask Learning w/o Information Disentanglement
python multitasklearning.py --tasks ag yelp yahoo
```
