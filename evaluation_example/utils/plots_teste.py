from plots import plot_tsne_graph_for_model
import joblib

model = joblib.load("etm_k10")

plot_tsne_graph_for_model(model, "sakura_haruno")
