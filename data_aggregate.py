import numpy as np
import pandas as pd


import os

news = [x for x in os.listdir('./') if 'NEWS' in x]
stocks = [x for x in os.listdir('./') if 'STOCK' in x]

def aggregate(source_files, fields, mode, target_path):
	res = []
	for f in source_files:
		data = pd.read_csv(f).groupby(fields).aggregate(mode)
		res.append(data)
	final = pd.concat(res)
	final.to_csv(target_path)
	print('Done!')


aggregate(news, ['date'], 'sum', 'final_news.csv')
aggregate(stocks, ['trade_date'], 'mean', 'final_stock.csv')

