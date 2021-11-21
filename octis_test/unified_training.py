import json
from utils.utils import get_best_hyperparameters
from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
from octis.models.CTM import CTM
from octis.models.ETM import ETM
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Categorical, Integer
import argparse
import os

parser = argparse.ArgumentParser(description='Preprocessor destined for OCTIS dataset creation')
parser.add_argument('--dataset_path', type=str, help='base path for the dataset files', required=True)
parser.add_argument('--embeddings_path', type=str, default='empty', help='path for the Word2Vec embeddings files', required=False)
parser.add_argument('--models', type=str, nargs='+', default=[], required=False)
parser.add_argument('--all-models', dest='all_models', action='store_true')
parser.set_defaults(all_models=False)
parser.add_argument('--min_k', type=int, default=3, help='Minimum no. of topics (k) in the search space', required=False)
parser.add_argument('--max_k', type=int, default=30, help='Maximum no. of topics (k) in the search space', required=False)
args = parser.parse_args()

models_to_train = ["lda", "ctm", "etm"] if args.all_models else args.models

if args.all_models or "etm" in args.models:
    assert args.embeddings_path != 'empty', "embeddings_path must be provided to train ETM models!"

dataset = Dataset()
dataset.load_custom_dataset_from_folder(args.dataset_path)

language = args.dataset_path.split(os.path.sep)[-1]

topics_dimension = Integer(low=int(args.min_k), high=int(args.max_k))

models = []
for model_name in models_to_train:
    if model_name == "lda":
        models.append({
            "model": LDA(),
            "search_space": {
                "num_topics": topics_dimension,
                "alpha": Categorical(["asymmetric", "auto"]),
                "eta": Categorical([None, "auto"])
            }
        })
    elif model_name == "ctm":
        bert_model = "distiluse-base-multilingual-cased-v1" \
            if language == "pt" else "bert-base-nli-mean-tokens"
        models.append({
            "model": CTM(inference_type="combined", 
                bert_model=bert_model),
            "search_space": {
                "num_topics": topics_dimension,
            }
        })
    elif model_name == "etm":
        models.append({
            "model": ETM(train_embeddings=False, embeddings_type='keyedvectors', 
                embeddings_path=args.embeddings_path),
            "search_space": {
                "num_topics": topics_dimension,
            }
        })


coherence_metric = Coherence(topk=10, measure="c_npmi", texts=dataset.get_corpus())

for i in range(len(models)):
    optimizer=Optimizer()
    result = optimizer.optimize(models["model"][i], dataset, coherence_metric, models["search_space"][i], 
                                save_path="../optm_results", # path to store the results
                                number_of_call=30, # number of optimization iterations
                                model_runs=5, plot_best_seen=True) # number of runs of the topic model
    #save the results of the optimization in file
    results_file = f'{models_to_train[i]}.json'
    result.save(results_file)
    results_with_best_hyperparams = get_best_hyperparameters(results_file)
    json.dump(results_with_best_hyperparams, open(results_file, "w"), indent=4)
    print(f'Optimization finished. Results can be found at: {results_file}')
    