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
parser.add_argument('--embeddings_path', type=str, help='path for the Word2Vec embeddings files', required=True)
args = parser.parse_args()

dataset = Dataset()
dataset.load_custom_dataset_from_folder(args.dataset_path)

language = args.dataset_path.split(os.path.sep)[-1]

lda_model = LDA()
ctm_model = CTM(bert_model="distilusebase-multilingual-cased" if language == "pt" else "bert-base-nli-mean-tokens")
etm_model = ETM(train_embeddings=False, embeddings_path=args.embeddings_path)

coherence_metric = Coherence(topk=10, measure="c_npmi", texts=dataset.get_corpus())

# Define the search space. To see which hyperparameters to optimize, see the topic model's initialization signature
search_space = {"num_topics": Integer(low=3, high=30)}

for model in [lda_model, ctm_model, etm_model]:
    # Initialize an optimizer object and start the optimization.
    optimizer=Optimizer()
    result = optimizer.optimize(model, dataset, coherence_metric, search_space, save_path="../optm_results", # path to store the results
                                number_of_call=30, # number of optimization iterations
                                model_runs=5) # number of runs of the topic model
    #save the results of the optimization in file
    result.save(f'{model.info()["name"]}.json')
    