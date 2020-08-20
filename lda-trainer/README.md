# lda-trainer
Scripts to train scikit-learn Latent Dirichlet Allocation models given a set of hyperparameters and a preprocessed dataset. The dataset must be preprocessed using the ```text-preprocessor``` tools of this project.

This script trains LDA models and outputs the training results to a CSV file containing details about each model trained, like coherence scores
or hyperparameters used. Also, the models are saved as [joblib](https://joblib.readthedocs.io/en/latest/) objects, where the most important components of each one are persisted.

### Setup
Activate/create the Pipenv environment with: ```pipenv shell```

Then run: ```pipenv install```

### Running
Execute the ```python3 src/train_lda.py``` script. These are the arguments that must be passed over to the script:

* ```--dataset``` - path of the training dataset
* ```--field``` - dataset field containg the text data to fit
* ```--topics``` - list of **K** values to use as number of topics
* ```--alphas``` - list of *alpha* values to be used with LDA for training
* ```--betas``` - list of *beta* values to be used with LDA for training

Below, a command example:

```python3 train_lda.py --dataset datasets/reddit_pt_2005_2020_desabafos_brasil[preprocessed].json --field body --topics 5 10 15 20 25 30 --alphas 0.1 0.2 0.3 0.4 0.6 0.8 1.0 --betas 0.1 0.2 0.3 0.4 0.6 0.8 1.0```
