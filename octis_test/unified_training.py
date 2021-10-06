from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Integer
import argparse
import os

parser = argparse.ArgumentParser(description='Preprocessor destined for OCTIS dataset creation')
parser.add_argument('--dataset_path', type=str, help='base path for the dataset files', required=True)
parser.add_argument('--embeddings_path', type=str, default='empty', help='path for the Word2Vec embeddings files', required=False)
parser.add_argument('--models', type=str, nargs='+', default=[], required=False)
parser.add_argument('--all-models', dest='all_models', action='store_true')
parser.set_defaults(all_models=False)
args = parser.parse_args()

models_to_train = ["lda", "ctm", "etm"] if args.all_models else args.models

if args.all_models or "etm" in args.models:
    assert args.embeddings_path != 'empty', "embeddings_path must be provided to train ETM models!"

dataset = Dataset()
dataset.load_custom_dataset_from_folder(args.dataset_path)

language = args.dataset_path.split(os.path.sep)[-1]

models = []
for model_name in models_to_train:
    if model_name == "lda":
        models.append(LDA())
    elif model_name == "ctm":
        models.append(
            CTM(bert_model="distiluse-base-multilingual-cased-v1" if language == "pt" else "bert-base-nli-mean-tokens"),
        )
    elif model_name == "etm":
        models.append(
            ETM(train_embeddings=False, embeddings_path=args.embeddings_path),
        )

coherence_metric = Coherence(topk=10, measure="c_npmi", texts=dataset.get_corpus())

# Define the search space. To see which hyperparameters to optimize, see the topic model's initialization signature
search_space = {"num_topics": Integer(low=3, high=30)}

for i in range(len(models)):
    # Initialize an optimizer object and start the optimization.
    optimizer=Optimizer()
    result = optimizer.optimize(models[i], dataset, coherence_metric, search_space, save_path="../optm_results", # path to store the results
                                number_of_call=30, # number of optimization iterations
                                model_runs=5) # number of runs of the topic model
    #save the results of the optimization in file
    result.save(f'{models_to_train[i]}.json')
    