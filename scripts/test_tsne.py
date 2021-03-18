import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgba
import joblib
from sklearn.manifold import TSNE

DEFAULT_SIZE = 2000

path = "/home/luizmatos/Projetos/UFF/Python/reddit-topic-modelling/evaluation/2021-02-26_Reddit_Nouns_Only_Pt_min_df_0.01_max_df_0.8/models/lda/lda_k5"
model = joblib.load(path)
print(model["topics_vectors"]["word_weights"][0])

labels = [f'Tópico {idx}: {", ".join(topic[:3])}' for idx, topic in enumerate(model["topics"])]
print(f'labels: {labels}')

n_colors = len(labels)
colors_numbers = np.linspace(0, 1, n_colors)

# X = model["topics_vectors"]["word_vectors"][0]
# # print(f'len X = {len(X)} X: {X}')

# Y = TSNE().fit_transform(X)
# # print(f'len Y = {len(Y)} Y: {Y}')

# Z = Y.T
# print(f'len Z = {len(Z)} Z: {Z}')

X_topic_avgs = model["averaged_topics_vectors"]
# print(f'len X_topic_avgs = {len(X_topic_avgs)} X_topic_avgs: {X_topic_avgs}')

Y_topic_avgs = TSNE().fit_transform(X_topic_avgs)
# print(f'len Y_topic_avgs = {len(Y_topic_avgs)} Y_topic_avgs: {Y_topic_avgs}')

Z_topic_avgs = Y_topic_avgs.T
print(f'len Z_topic_avgs = {len(Z_topic_avgs)} Z_topic_avgs: {Z_topic_avgs}') 

color_seq = ['red', 'blue', 'green', 'purple', 'orange']
color_map = cm.get_cmap("gist_rainbow")

fig, ax = plt.subplots(1)
# ax.scatter(Z_topic_avgs[0], Z_topic_avgs[1], c=color_seq, marker="*")
ax.scatter(
    Z_topic_avgs[0], 
    Z_topic_avgs[1], 
    s=[100] * n_colors,
    c=colors_numbers, 
    cmap=color_map, 
    marker="*"
)
for idx, avg in enumerate(Y_topic_avgs):
    ax.annotate(labels[idx], tuple(avg))

for idx, X in enumerate(model["topics_vectors"]["word_vectors"]):
    Z = TSNE().fit_transform(X).T
    # ax.scatter(Z[0], Z[1], c=color_seq[idx])
    color = [list(color_map(colors_numbers[idx]))]
    # colors_temp = [colors_numbers[idx]] * len(Z[0])
    print(color)
    # weights = model["topics_vectors"]["word_weights"][idx] * DEFAULT_SIZE
    # ax.scatter(Z[0], Z[1], s=weights, c=color, cmap=color_map)
    ax.scatter(Z[0], Z[1], c=color, cmap=color_map)

plt.axis('off')
plt.show()


# x = [0.15, 0.3, 0.45, 0.6, 0.75]
# x2 = [0.95, 0.3, 0.6, 0.8, 0.75]
# y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]

# color_seq = ['red', 'blue', 'green', 'purple', 'orange']

# fig, ax = plt.subplots(1)
# ax.scatter(x, y, c=color_seq)
# ax.scatter(x2, y, c=color_seq)

# for i, txt in enumerate(labels):
#     ax.annotate(txt, (x[i], y[i]))
# plt.axis('off')
# plt.show()