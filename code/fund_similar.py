import pickle
import numpy as np
import pandas as pd
from datum import Datum
import networkx as nx
import graph
from gensim.models import Word2Vec

import pickle


def load_obj(name): 
    if '.pkl' not in name:
        name+='.pkl'
    with open( name , 'rb') as f:
        return pickle.load(f)

f = '../data/fund_data.pkl'
d = load_obj(f)


import numpy as np

data_dir = '../data/'
fundhold=d[((d['rpt_date']=='2018-06-30') | (d['rpt_date']=='2018-12-31'))]
fund=fundhold[['fund_code','rpt_date','stock_code','proportiontototalstockinvestments']]

fund_values = fund.values

list_funds = pd.Series(fund['fund_code'].unique())
list_dates = pd.Series(fund['rpt_date'].unique())
list_stocks = pd.Series(fund['stock_code'].unique())

weight_matrix = np.zeros((len(list_dates), len(list_funds), len(list_stocks)))  
for ind in range(len(fund_values)):
    fund_index = list_funds.index[list_funds==fund_values[ind,0]][0]
    time_index = list_dates.index[list_dates==fund_values[ind,1]][0]
    stock_index = list_stocks.index[list_stocks==fund_values[ind,2]][0]
    weight_matrix[time_index, fund_index, stock_index] = fund_values[ind,3]
select_date=list_dates.values

weight_matrix = np.copy(weight_matrix)
weight_matrix_total = np.sum(weight_matrix, axis=0)

arr_tmp = []
for fund_index in range(len(list_funds)):
    a_edge_index = np.where(weight_matrix_total[fund_index] != 0)[0]
    for edge_index in a_edge_index:
        arr_tmp.append([fund_index, len(list_funds) + edge_index, weight_matrix_total[fund_index, edge_index]])
arr_tmp = np.array(arr_tmp)
pd_tmp = pd.DataFrame(arr_tmp)
pd_tmp[0] = pd_tmp[0].astype(int)
pd_tmp[1] = pd_tmp[1].astype(int)
output_name='fund'
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

total_embedding = np.array(pd.read_csv(data_dir+'embedding/'+output_name+'.emb', header=None, sep=' ', skiprows=1))
embedding = np.zeros((len(list_funds), total_embedding.shape[1]-1))
use_index = []
for emb in total_embedding:
    if int(emb[0]) < len(list_funds):
        embedding[int(emb[0])] = emb[1:]
        use_index.append(int(emb[0]))
use_index = np.array(use_index)

result=[]
tmp=embedding
for i in range(embedding.shape[0]):
    result.append(np.corrcoef(tmp[0],tmp[i])[0][1])

result_df=pd.DataFrame({'fund_code':list_funds,'corrcoef':result})
result_df.merge(fundhold[['fund_code','sec_name']].drop_duplicates(),on='fund_code',how='left').sort_values('corrcoef',ascending=False).to_csv('fund_corrcoef.csv',index=False,encoding="utf_8_sig")