from plots import plot_tsne_graph_for_model, plot_lexical_categories_histogram
from lexical_categories_analysis import get_empath_categories_for_topics, get_liwc_categories_for_topics
import joblib

# model = joblib.load("etm_k10")

# # plot_tsne_graph_for_model(model, "sakura_haruno")
# counts = get_empath_categories_for_topics(model, normalize=True)
# print(counts)
# plot_lexical_categories_histogram(counts)

model2 = joblib.load("lda_k5")
counts = get_liwc_categories_for_topics(model2, "LIWC2007_Portugues_win.dic", normalize=True)
print(counts)
plot_lexical_categories_histogram(counts)
