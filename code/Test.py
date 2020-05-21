import pickle
import numpy as np
import pandas as pd
from datum import Datum
import networkx as nx
import graph
from gensim.models import Word2Vec

data_dir = '../data/'
output_name = 'graph'

data = Datum()

data.data_prepare()
weight_matrix = np.copy(data.weight_matrix)
weight_matrix_total = np.sum(weight_matrix, axis=0)


arr_tmp = []
for stock_index in range(len(data.list_stocks)):
    a_edge_index = np.where(weight_matrix_total[stock_index] != 0)[0]
    for edge_index in a_edge_index:
        arr_tmp.append([stock_index, len(data.list_stocks) + edge_index, weight_matrix_total[stock_index, edge_index]])
arr_tmp = np.array(arr_tmp)
pd_tmp = pd.DataFrame(arr_tmp)
pd_tmp[0] = pd_tmp[0].astype(int)
pd_tmp[1] = pd_tmp[1].astype(int)
path = data_dir + 'graph/{}.csv'.format(output_name)
pd_tmp.to_csv(path, index=False, sep=' ')


nx_G = nx.read_edgelist(path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
nx_G = nx_G.to_undirected()
G = graph.Graph(nx_G, False, 1, 1)
G.preprocess_transition_probs()

walks = G.simulate_walks(200, 200)
walks = [list(map(str, walk)) for walk in walks]

from gensim.models import Word2Vec

model = Word2Vec(walks, size=32, window=6, min_count=0, sg=1, workers=2, iter=30)
model.wv.save_word2vec_format(data_dir + 'embedding/{}.emb'.format(output_name))

