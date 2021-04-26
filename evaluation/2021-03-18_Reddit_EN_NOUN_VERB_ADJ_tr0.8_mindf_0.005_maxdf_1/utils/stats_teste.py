from stats import get_word_probabilities, merge_word_probabilities
import numpy as np
import torch
import joblib

etm = joblib.load("/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/models_evaluation/Reddit pt - 12-11-2020 - Dataset tokenizado/utils/test/etm_k5")
etm_word_probs = get_word_probabilities(etm["idx_to_word"], etm["topic_word_dist"].cpu().numpy(), etm["topic_word_matrix"])

print(f'ETM word probabilities: {list(etm_word_probs.items())[:5]}')

lda = joblib.load("/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/models_evaluation/Reddit pt - 12-11-2020 - Dataset tokenizado/utils/test/lda_k5")

softmax = torch.nn.Softmax(dim=1)
print(np.sum(lda["topic_word_dist"][0]))
topic_word_mtx = softmax(torch.from_numpy(lda["topic_word_dist"])).cpu().numpy()
print(np.sum(topic_word_mtx[0]))

lda_word_probs = get_word_probabilities(lda["idx_to_word"], topic_word_mtx, lda["topic_word_matrix"])

print(f'LDA word probabilities: {list(lda_word_probs.items())[:5]}')
