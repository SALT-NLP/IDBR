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

#### Training Finetune Baseline Model

Please run ```./src/finetune.py``` to train the Finetune Baseline model:

```
python finetune.py --tasks ag yelp yahoo --epochs 4 3 2   #Example for length-3 task sequence
python finetune.py --tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1   # Example for length-5 task sequence
```

Please run ```./src/naivereplay.py``` to train the Naive Replay Baseline model:

```
python naivereplay.py --tasks ag yelp yahoo --epochs 4 3 2   #Example for length-3 task sequence
python naivereplay.py --tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 # Example for length-5 task sequence
```


