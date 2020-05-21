from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import graph
import networkx as nx

data_dir = '../data/'

def get_data(start_date, end_date, stock_code):
    # get the data, the stock_data directory is extracted from /data/AMC/stock_feature and /data/AMC/stock_price
    price_csv = pd.read_csv(data_dir + 'stock_data/'+str(int(stock_code))+'_price.csv', encoding="ISO-8859-1")
    dates = np.array(price_csv['TradingDay'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_price = np.array(price_csv['ClosePrice'][select_index])
    
    feature_csv = pd.read_csv(data_dir + 'stock_data/'+str(int(stock_code))+'_feature.csv', encoding="ISO-8859-1")
    dates = np.array(feature_csv['trading_day'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_feature = np.array(feature_csv)[select_index, 1:]  
    return return_price, return_feature

class Datum:
    def __init__(self, param=None):
        self.list_funds = []
        self.list_stocks = []
        self.embedding = np.array([0])
        self.dict_code2name = {}
        self.price_data = np.array([0])
        self.feature_data = np.array([0])
        self.code_tag = []
        self.dimension = 32
        
        self.select_date = []
        for year in range(2008 * 10000, 2017 * 10000, 10000):
            for month_day in [0, 400, 700, 1000]:
                self.select_date.append(year + month_day)
        self.select_date.append(2017 * 10000) 
            
    def data_prepare(self):
        head = '2018-06-30'
        tail = '2018-12-31'
        fundhold = pd.read_csv(data_dir + 'mutualfundholding.csv')
        #index = np.where((date >= '2018-06-30') & (date <= '2018-12-31'))[0]
        fund = fundhold[(fundhold['rpt_date']>=head) & (fundhold['rpt_date']<=tail)]
        fund_values = fund.values

        self.list_stocks_name = pd.Series(fund['stock_code'].unique())
        self.list_funds = pd.Series(fund['fund_code'].unique())
        self.list_dates = pd.Series(fund['rpt_date'].unique())
        self.list_stocks = pd.Series(fund['stock_code'].unique())

        self.weight_matrix = np.zeros((len(self.list_dates), len(self.list_stocks), len(self.list_funds)))  
        for ind in range(len(fund)):
            fund_index = self.list_funds.index[self.list_funds==fund_values[ind,0]][0]
            time_index = self.list_dates.index[self.list_dates==fund_values[ind,1]][0]
            stock_index = self.list_stocks.index[self.list_stocks==fund_values[ind,2]][0]
            self.weight_matrix[time_index, stock_index, fund_index] = fund_values[ind,3]
        self.select_date=self.list_dates.values
            
#         # holding data to matrix
#         fundhold = pd.read_csv(data_dir + 'mutualfundholding.csv')

#         fund = np.array(fundhold)[:, 0]
#         date = np.array(fundhold)[:, 1]
#         stock = np.array(fundhold)[:, 3]
#         value = np.array(fundhold)[:, 4]

#         index = np.where((date >= 20080000) & (date <= 20170000))[0]

#         raw_funds = fund[index]
#         raw_dates = date[index]
#         raw_stocks = stock[index]
#         raw_values = value[index]

#         list_funds = []
#         list_dates = []
#         list_stocks = []
#         for fund in raw_funds:
#             if fund not in list_funds:
#                 list_funds.append(fund)
#         for date in raw_dates:
#             if date not in list_dates:
#                 list_dates.append(date)

#         self.list_funds = list_funds
#         self.list_stocks = np.load(data_dir+'stock_list.npy').tolist()

#         select_date = np.array(self.select_date)[:-1]
      
#         self.weight_matrix = np.zeros((len(select_date), len(self.list_stocks), len(self.list_funds)))

#         for ind in range(len(raw_funds)):
#             try:
#                 stock_index = self.list_stocks.index(raw_stocks[ind].split('.')[0])
#             except:
#                 continue
#             fund_index = self.list_funds.index(raw_funds[ind])
#             time_index = np.where(select_date < raw_dates[ind])[0][-1]
#             self.weight_matrix[time_index, stock_index, fund_index] = raw_values[ind]                
            
#         # stock code to Chinese
#         industry = pd.read_csv(data_dir + 'industry.csv')
#         for ele in np.array(industry):
#             ele[0] = str(ele[0])
#             for _ in range(6-len(ele[0])):
#                 ele[0] = '0' + ele[0]        
#             if not ele[0] in self.dict_code2name:
#                 self.dict_code2name[ele[0]] = ele[4]+'-'+ele[2]+';'
#             else:
#                 name = ele[4] + '-' + ele[2]
#                 if name not in self.dict_code2name[ele[0]].split(';'):
#                     self.dict_code2name[ele[0]] += name+';'
            
    def get_embedding(self, file_name):
        total_embedding = np.array(pd.read_csv(data_dir+'embedding/'+file_name+'.emb', header=None, sep=' ', skiprows=1))
        print(total_embedding.shape)
        # use_index = np.load('/data/zhige_data/embedding_rotation/embedding/stable_index_'+self.param+'.npy')
        # self.list_stocks = [self.list_stocks[i] for i in use_index]
        self.embedding = np.zeros((len(self.list_stocks), total_embedding.shape[1]-1))
        self.use_index = []
        for emb in total_embedding:
            if int(emb[0]) < len(self.list_stocks):
                self.embedding[int(emb[0])] = emb[1:]
                self.use_index.append(int(emb[0]))
        self.use_index = np.array(self.use_index)
            
    def supervised_data_prepare(self, start_date, end_date):
#         self.price_data = []
#         self.feature_data = []
#         for count, code in enumerate(self.list_stocks):
#             a_p, a_f = get_data(start_date, end_date, int(code))
#             self.price_data.append(a_p)
#             self.feature_data.append(a_f)
#         self.price_data = np.array(self.price_data)
#         self.feature_data = np.array(self.feature_data)
        self.feature_data=np.load(data_dir+'X.npy')
        self.price_data=np.load(data_dir+'y_10.npy')
        
    def ic_prepare(self, start_date, end_date):
        self.data.ar_ic=self.price_data
#         day_need = get_data(start_date, end_date, 1)[0].shape[0]
#         self.ar_ic = np.zeros((len(self.list_stocks), day_need, 4))
#         # mkt1, 3, 5, 10
#        for count, code in enumerate(self.list_stocks):
#             a_p, a_f = get_data(start_date, end_date+30, code)
#             a_p = a_p[:day_need+11]
#             for day in range(day_need):
#                 for look_count, look_after in enumerate([1, 3, 5, 10]):
#                     self.ar_ic[count, day, look_count] = (a_p[day+1+look_after] - a_p[day+1]) / a_p[day+1]

                
        
if __name__ == "__main__":
    data = Datum()
