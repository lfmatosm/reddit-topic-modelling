from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset

handler = TextHandler("documents.txt")
handler.prepare() # create vocabulary and training data

# generate BERT data
training_bert = bert_embeddings_from_file("documents.txt", "distiluse-base-multilingual-cased")

training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="combined", n_components=50)

ctm.fit(training_dataset) # run the model
