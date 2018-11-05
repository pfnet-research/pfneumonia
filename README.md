# PFNeumonia

This repo contains the 6th place solution for RSNA Pneumonia Detection Challenge.

All trainings and predictions were done with eight Tesla P100s.

# Preparation
- Locate Kaggle RSNA Pneumonia datasets on `./RSNA`.
The names of subdirectories must be `stage_2_train_images` and `stage_2_test_images`.
Labels for training dataset must be written in `stage_2_train_labels.csv`.
- Download ResNet pretrained models from `https://github.com/KaimingHe/deep-residual-networks` and locate them on `~/.chainer/dataset/pfnet/chainer/models`.
- Install requirements by executing `pip install -r requirements.txt`

# How to train
First, shuffle patient IDs by executing the command below. This process is required if you are going to train with a new dataset other than the official Stage2 train or test images.

```bash
PYTHONPATH=. python scripts/shuffle_patients.py
```

Our final submission is an ensemble of 10 models which derive from a 10-fold cross-validation.
To run a 10-fold CV, run the command:

```bash
./run_cross_validation.sh
```

# How to predict
Predict twice with even (02468) and odd (13579) five models.

```bash
./pred02468.sh even.csv
./pred13579.sh odd.csv
```

Merge these predictions to get the merged CSV.

```bash
PYTHONPATH=. python examples/segmentation/ensemble_submission.py even.csv odd.csv -t0.5 -o merged.csv
```

Finally, adjust threshold of confidence to get the final submission file.

```bash
PYTHONPATH=. python scripts/increase_threshold.py -i merged.csv -o final.csv -t 0.3
```

# Links

- Official: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
- Official Leaderboard: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/leaderboard
