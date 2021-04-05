# Information-Disentanglement-Based-Regularization(IDBR)

This repo contains codes for the following paper:

If you would like to refer to it, please cite the paper mentioned above.

## Getting Started

These instructions will get you running the codes of IDBR.

### Requirements

### Code Structure

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

Please run ```./src/multitasklearning.py" to train the multitask-learning model:

```

```
