

import pandas as pd
#from openpyxl import load_workbook
import time
import copy
#import random
import numpy as np
#from functools import reduce

#%% 
filename = "G:/"
filename_2 = "1/"

user_num_extract = 10000
#date_extract_start = '2021-02-01 00:00:00'
#date_extract_end = '2021-03-15 00:00:00'

#%% Data reading
start_read = time.perf_counter()
# user data
user_data = pd.read_csv(filename + "user.csv", delimiter = '\t', header = None)
user_data.columns = ['user_id', 'sex', 'age', 'province', 'city', 'country']
user_data = user_data.sort_values(by = 'user_id').reset_index(drop=True)
# product data
product_data = pd.read_csv(filename + "product.csv", delimiter = '\t', header = None)
product_data.columns = ['sku_id', 'sku_name', 'first_cate_id',	'second_cate_id', 'third_cate_id', 'first_cate_name', 'second_cate_name', 'third_cate_name']
product_data = product_data.sort_values(by = 'sku_id').reset_index(drop=True)

# action sequence data
user_action_sequence_data = pd.read_csv(filename + filename_2 + "user_action_sequence_oneweek.csv")
elapsed_read = (time.perf_counter() - start_read) 
print("The time for reading data： ",elapsed_read)   
print('********************Data reading done!*****************')

#%%

# Only first_cate
product_data_new = copy.copy(product_data.iloc[:,[0,1,2,5]])
# 将类别为废弃商品的改为食品饮料1320
product_data_new.loc[product_data_new.loc[(product_data_new['first_cate_name']=='废弃商品')].index.tolist(),['first_cate_id','first_cate_name']]=[1320,'食品饮料']
product_data_new.loc[product_data_new.loc[(product_data_new['first_cate_name']=='(京喜供应链中心)方便速食')].index.tolist(),['first_cate_id','first_cate_name']]=[1320,'食品饮料']
product_data_new.loc[product_data_new.loc[(product_data_new['first_cate_name']=='生鲜')].index.tolist(),['first_cate_id','first_cate_name']]=[1320,'食品饮料']


#%%
# user id
list_user = pd.DataFrame(user_action_sequence_data.user_id.unique())  #10000
list_user.columns=['user_id']
# sku id
list_product = pd.DataFrame(user_action_sequence_data.sku_id.unique()) #56065
list_product.columns=['sku_id']
# sku_name and first cate
user_action_sequence_product = pd.merge(user_action_sequence_data, product_data_new, on='sku_id', how='left')
product_noinfo = user_action_sequence_product[user_action_sequence_product.isnull().T.any()]  

# fill nan
user_action_sequence_product = user_action_sequence_product.fillna({'first_cate_id':0, 'sku_name':'unknown', 'first_cate_name':'unknown'})
# delete no details
# user_action_sequence_product = pd.merge(user_action_sequence_data, product_data_new, on='sku_id', how='inner') 
# extact user and product data
user_dataframe = pd.DataFrame(copy.copy(user_action_sequence_product['user_id'].unique()))
user_dataframe.columns=['user_id']
product_dataframe = pd.DataFrame(copy.copy(user_action_sequence_product['sku_id'].unique()))
product_dataframe.columns=['sku_id']
user_product_dataframe = pd.DataFrame(copy.copy(user_action_sequence_product.iloc[:,0:2]))
# the number of orders from '2021-02-01 00:00:00' to '2021-03-15 00:00:00'
list_action_sequence = list(user_action_sequence_product.action_sequence)
order_num_total = ''.join(list_action_sequence).count('o') #72728
# delete actions after 'o' in the sequence
for i in range(0,user_action_sequence_product.shape[0]):
    #i = 8
    s = copy.copy(user_action_sequence_product['action_sequence'][i])
    user_action_sequence_product['action_sequence'][i]=s.partition('o')[0]+s.partition('o')[1]+'s'
user_action_sequence_product.to_csv(filename + filename_2 + "user_action_sequence_product.csv", index=False, sep=',', header=True)                
user_action_sequence_product = pd.read_csv(filename + filename_2 + "user_action_sequence_product.csv")


#%%
user_action_sequence_product_num = copy.copy(user_action_sequence_product)
for i in range(0,user_action_sequence_product.shape[0]):
    print('i=',i)
    s = copy.copy(user_action_sequence_product['action_sequence'][i])
    user_action_sequence_product_num.loc[i,('action_pv_num')] = s.count('p')
    user_action_sequence_product_num.loc[i,('action_fav_num')] = s.count('f')
    user_action_sequence_product_num.loc[i,('action_cart_num')] = s.count('c')
    user_action_sequence_product_num.loc[i,('action_order_num')] = s.count('o')
    user_action_sequence_product_num.loc[i,('action_interupt_num')] = s.count('-')
    user_action_sequence_product_num.loc[i,('action_total_num')] = len(s)
    user_action_sequence_product_num.loc[i,('action_want_num')] = s.count('f')+s.count('c')+s.count('o')
user_action_sequence_product_num.to_csv(filename + filename_2 + "user_action_sequence_product_num.csv", index=False, sep=',', header=True)                
user_action_sequence_product_num = pd.read_csv(filename + filename_2 + "user_action_sequence_product_num.csv")

#%%               
def min_max_normalize(df,name):
	# uniformization
	max_number = df[name].max()
	min_number = df[name].min()
	df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
	return df

#%% extract user feature

def user_action_rate(action_name):
    #action_name = 'fav'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    # num of each user and each action
    name = 'action_'+action_name+'_num'
    normalized_frame = min_max_normalize(grouped[name].sum(),name)[name]
    # rate of actions of one user and one action to all actions
    ratio_frame = grouped[name].sum()[name]/grouped['action_total_num'].sum()['action_total_num']    
    user_action_rate = pd.concat([user_dataframe['user_id'],normalized_frame,ratio_frame],axis=1)
    user_action_rate.columns = ['user_id', 'user_action_'+action_name+'_rate_1', 'user_action_'+action_name+'_rate_2']     
    return user_action_rate

user_action_pv_rate = user_action_rate('pv')
user_action_fav_rate = user_action_rate('fav')
user_action_cart_rate = user_action_rate('cart')
user_action_order_rate = user_action_rate('order')
user_action_interupt_rate = user_action_rate('interupt')


# user action
def user_action_total_rates():
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    name = 'action_total_num'
    normalized_frame = min_max_normalize(grouped[name].sum(),name)[name]
    user_action_total_rates = pd.concat([user_dataframe,normalized_frame],axis=1)
    user_action_total_rates.columns = ['user_id', 'user_action_total_rates']     
    return user_action_total_rates

user_action_total_rates = user_action_total_rates()

# weight on each action
action_weight=[1,2,4,8,0]

user_action_sequence_product_num_weight = copy.copy(user_action_sequence_product_num)
user_action_sequence_product_num_weight['action_total_weight'] = user_action_sequence_product_num_weight.apply(lambda x: np.dot(x[6:11],action_weight),axis=1)
user_action_sequence_product_num_weight.to_csv(filename + filename_2 + "user_action_sequence_product_num_weight.csv", index=False, sep=',', header=True)                
user_action_sequence_product_num_weight = pd.read_csv(filename + filename_2 + "user_action_sequence_product_num_weight.csv")

def user_weight_action_rate():
    grouped = user_action_sequence_product_num_weight.groupby('user_id',as_index=False)
    name = 'action_total_weight'
    normalized_frame = min_max_normalize(grouped[name].sum(),name)[name]
    user_weight_action_rate = pd.concat([user_dataframe,normalized_frame],axis=1)
    user_weight_action_rate.columns = ['user_id', 'user_weight_action_rate']     
    return user_weight_action_rate

user_weight_action_rate = user_weight_action_rate()
user_action_rates=[user_action_pv_rate,user_action_fav_rate,user_action_cart_rate,user_action_order_rate,user_action_interupt_rate,user_action_total_rates,user_weight_action_rate] 

def user_consume_action_rate():
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    # ratio of buying actions to all actions, if not buying, is -1
    ratio_frame = grouped['action_order_num'].sum()['action_order_num']/grouped['action_total_num'].sum()['action_total_num']    
    ratio_frame = ratio_frame.replace(0, -1)
    user_consume_action_rate = pd.concat([user_dataframe['user_id'],ratio_frame],axis=1)
    user_consume_action_rate.columns = ['user_id', 'user_consume_action_rate']     
    return user_consume_action_rate

user_consume_action_rate = user_consume_action_rate()

def user_consume_want_rate():
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    # ratio of buying actions to all actions, if not buying, is -1
    grouped_want = grouped['action_want_num'].sum()['action_want_num']+1
    ratio_frame = grouped['action_order_num'].sum()['action_order_num']/grouped_want
    ratio_frame = ratio_frame.replace(0, -1)
    user_consume_want_rate = pd.concat([user_dataframe['user_id'],ratio_frame],axis=1)
    user_consume_want_rate.columns = ['user_id', 'user_consume_want_rate']     
    return user_consume_want_rate

user_consume_want_rate = user_consume_want_rate()
user_consume_rates = [user_consume_action_rate,user_consume_want_rate]

def user_action_category_counts():
    name = 'user_action_category_counts'
    grouped = user_action_sequence_product_num[['user_id', 'first_cate_id']].groupby('user_id')['first_cate_id'].nunique().reset_index()
    grouped.columns = ['user_id', name]
    user_action_category_counts = min_max_normalize(grouped, name)
    return user_action_category_counts

user_action_category_counts = user_action_category_counts()

def user_want_category_counts():
    name = 'user_want_category_counts'
    mask = pd.Series(list(map(lambda x: True if x>0 else False, user_action_sequence_product_num['action_want_num'])))
    grouped = user_action_sequence_product_num[mask][['user_id', 'first_cate_id']].groupby('user_id')['first_cate_id'].nunique().reset_index()
    grouped.columns = ['user_id', name]
    user_want_category_counts = min_max_normalize(grouped, name)
    return user_want_category_counts

user_want_category_counts = user_want_category_counts()

def user_consume_category_counts():
    name = 'user_consume_category_counts'
    mask = pd.Series(list(map(lambda x: True if x>0 else False, user_action_sequence_product_num['action_order_num'])))
    grouped = user_action_sequence_product_num[mask][['user_id', 'first_cate_id']].groupby('user_id')['first_cate_id'].nunique().reset_index()
    grouped.columns = ['user_id', name]
    user_consume_category_counts = min_max_normalize(grouped, name)
    return user_consume_category_counts

user_consume_category_counts = user_consume_category_counts()
user_category_counts = [user_action_category_counts, user_want_category_counts, user_consume_category_counts]

def user_category_total_duplication():
    name = 'user_category_total_duplication'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    grouped_category = user_action_sequence_product_num.groupby(['user_id','first_cate_id'],as_index=False)
    user_action_total_num = grouped['action_total_num'].sum()
    user_category_total_num = grouped_category['action_total_num'].sum()
    unique_user_category = user_action_sequence_product_num[['user_id', 'first_cate_id']].drop_duplicates(['user_id', 'first_cate_id']).reset_index(drop=True)
    frame = pd.merge(user_category_total_num, user_action_total_num, on ='user_id', how='left')
    temp = pd.Series(list(map(lambda x, y: x*x/((y+1)*(y+1)), frame['action_total_num_x'], frame['action_total_num_y'])))
    user_category_total_duplication = pd.concat([unique_user_category, temp], axis=1)
    user_category_total_duplication.columns=('user_id','first_cate_id',name)
    user_category_total_duplication = pd.DataFrame(user_category_total_duplication.groupby('user_id',as_index=False)[name].sum())
    return user_category_total_duplication

user_category_total_duplication = user_category_total_duplication()

def user_category_want_duplication():
    name = 'user_category_want_duplication'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    grouped_category = user_action_sequence_product_num.groupby(['user_id','first_cate_id'],as_index=False)
    user_action_want_num = grouped['action_want_num'].sum()
    user_category_want_num = grouped_category['action_want_num'].sum()
    unique_user_category = user_action_sequence_product_num[['user_id', 'first_cate_id']].drop_duplicates(['user_id', 'first_cate_id']).reset_index(drop=True)
    frame = pd.merge(user_category_want_num, user_action_want_num, on ='user_id', how='left')
    temp = pd.Series(list(map(lambda x, y: x*x/((y+1)*(y+1)), frame['action_want_num_x'], frame['action_want_num_y'])))
    user_category_want_duplication = pd.concat([unique_user_category, temp], axis=1)
    user_category_want_duplication.columns=('user_id','first_cate_id',name)
    user_category_want_duplication = pd.DataFrame(user_category_want_duplication.groupby('user_id',as_index=False)[name].sum())
    return user_category_want_duplication

user_category_want_duplication = user_category_want_duplication()

def user_category_consume_duplication():
    name = 'user_category_consume_duplication'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    grouped_category = user_action_sequence_product_num.groupby(['user_id','first_cate_id'],as_index=False)
    user_action_consume_num = grouped['action_order_num'].sum()
    user_category_consume_num = grouped_category['action_order_num'].sum()
    unique_user_category = user_action_sequence_product_num[['user_id', 'first_cate_id']].drop_duplicates(['user_id', 'first_cate_id']).reset_index(drop=True)
    frame = pd.merge(user_category_consume_num, user_action_consume_num, on ='user_id', how='left')
    temp = pd.Series(list(map(lambda x, y: x*x/((y+1)*(y+1)), frame['action_order_num_x'], frame['action_order_num_y'])))
    user_category_consume_duplication = pd.concat([unique_user_category, temp], axis=1)
    user_category_consume_duplication.columns=('user_id','first_cate_id',name)
    user_category_consume_duplication = pd.DataFrame(user_category_consume_duplication.groupby('user_id',as_index=False)[name].sum())
    return user_category_consume_duplication

user_category_consume_duplication = user_category_consume_duplication()
    
user_category_duplication = [user_category_total_duplication, user_category_want_duplication, user_category_consume_duplication]

def user_item_total_duplication():
    name = 'user_item_total_duplication'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    grouped_item = user_action_sequence_product_num.groupby(['user_id','sku_id'],as_index=False)
    user_action_total_num = grouped['action_total_num'].sum()
    user_item_total_num = grouped_item['action_total_num'].sum()
    unique_user_item = user_action_sequence_product_num[['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id']).reset_index(drop=True)
    frame = pd.merge(user_item_total_num, user_action_total_num, on ='user_id', how='left')
    temp = pd.Series(list(map(lambda x, y: x*x/((y+1)*(y+1)), frame['action_total_num_x'], frame['action_total_num_y'])))
    user_item_total_duplication = pd.concat([unique_user_item, temp], axis=1)
    user_item_total_duplication.columns=('user_id','sku_id',name)
    user_item_total_duplication = pd.DataFrame(user_item_total_duplication.groupby('user_id',as_index=False)[name].sum())
    return user_item_total_duplication

user_item_total_duplication = user_item_total_duplication()

def user_item_want_duplication():
    name = 'user_item_want_duplication'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    grouped_item = user_action_sequence_product_num.groupby(['user_id','sku_id'],as_index=False)
    user_action_want_num = grouped['action_want_num'].sum()
    user_item_want_num = grouped_item['action_want_num'].sum()
    unique_user_item = user_action_sequence_product_num[['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id']).reset_index(drop=True)
    frame = pd.merge(user_item_want_num, user_action_want_num, on ='user_id', how='left')
    temp = pd.Series(list(map(lambda x, y: x*x/((y+1)*(y+1)), frame['action_want_num_x'], frame['action_want_num_y'])))
    user_item_want_duplication = pd.concat([unique_user_item, temp], axis=1)
    user_item_want_duplication.columns=('user_id','sku_id',name)
    user_item_want_duplication = pd.DataFrame(user_item_want_duplication.groupby('user_id',as_index=False)[name].sum())
    return user_item_want_duplication

user_item_want_duplication = user_item_want_duplication()

def user_item_consume_duplication():
    name = 'user_item_consume_duplication'
    grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    grouped_item = user_action_sequence_product_num.groupby(['user_id','sku_id'],as_index=False)
    user_action_consume_num = grouped['action_order_num'].sum()
    user_item_consume_num = grouped_item['action_order_num'].sum()
    unique_user_item = user_action_sequence_product_num[['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id']).reset_index(drop=True)
    frame = pd.merge(user_item_consume_num, user_action_consume_num, on ='user_id', how='left')
    temp = pd.Series(list(map(lambda x, y: x*x/((y+1)*(y+1)), frame['action_order_num_x'], frame['action_order_num_y'])))
    user_item_consume_duplication = pd.concat([unique_user_item, temp], axis=1)
    user_item_consume_duplication.columns=('user_id','sku_id',name)
    user_item_consume_duplication = pd.DataFrame(user_item_consume_duplication.groupby('user_id',as_index=False)[name].sum())
    return user_item_consume_duplication

user_item_consume_duplication = user_item_consume_duplication()

user_item_duplication = [user_item_total_duplication, user_item_want_duplication, user_item_consume_duplication]

def add_user_features():
	user_features = []
	user_features.extend(user_action_rates)
	user_features.extend(user_consume_rates)
	user_features.extend(user_category_counts)
	user_features.extend(user_category_duplication)
	user_features.extend(user_item_duplication)
    user_feature_data = copy.copy(user_dataframe)
	for f in user_features:
        user_feature_data = pd.merge(user_feature_data, f, on ='user_id', how='left')
	user_feature_data.fillna(-1, inplace=True)#默认填充-1，就地修改则为inplace=True
    return user_feature_data

user_feature_data = add_user_features()
user_feature_data.to_csv(filename + filename_2 + 'user_feature_data.csv', index=False, sep=',', header=True)
user_feature_data=pd.read_csv(filename + filename_2 + 'user_feature_data.csv')
#user_feature_data.rename(columns={'user_action_rates': 'user_action_total_rates'}, inplace=True)
#%% extract product feature
def item_action_rate(action_name):
    #action_name = 'pv'
    grouped = user_action_sequence_product_num.groupby('sku_id',as_index=False)
    # num of each product and each action
    name = 'action_'+action_name+'_num'
    normalized_frame = min_max_normalize(grouped[name].sum(),name)[name]
    # rate of actions of one user and one action to all actions
    ratio_frame = grouped[name].sum()[name]/grouped['action_total_num'].sum()['action_total_num']    
    item_action_rate = pd.concat([product_dataframe,normalized_frame,ratio_frame],axis=1)
    item_action_rate.columns = ['sku_id', 'item_action_'+action_name+'_rate_1', 'item_action_'+action_name+'_rate_2']     
    return item_action_rate

item_action_pv_rate = item_action_rate('pv')
item_action_fav_rate = item_action_rate('fav')
item_action_cart_rate = item_action_rate('cart')
item_action_order_rate = item_action_rate('order')
item_action_interupt_rate = item_action_rate('interupt')

# item action
def item_action_total_rates():
    grouped = user_action_sequence_product_num.groupby('sku_id',as_index=False)
    name = 'action_total_num'
    normalized_frame = min_max_normalize(grouped[name].sum(),name)[name]
    item_action_total_rates = pd.concat([product_dataframe,normalized_frame],axis=1)
    item_action_total_rates.columns = ['sku_id', 'item_action_total_rates']     
    return item_action_total_rates

item_action_total_rates = item_action_total_rates()

def item_weight_action_rate():
    grouped = user_action_sequence_product_num_weight.groupby('sku_id',as_index=False)
    name = 'action_total_weight'
    normalized_frame = min_max_normalize(grouped[name].sum(),name)[name]
    item_weight_action_rate = pd.concat([product_dataframe,normalized_frame],axis=1)
    item_weight_action_rate.columns = ['sku_id', 'item_weight_action_rate']     
    return item_weight_action_rate

item_weight_action_rate = item_weight_action_rate()

item_action_rates=[item_action_pv_rate,item_action_fav_rate,item_action_cart_rate,item_action_order_rate,item_action_interupt_rate,item_action_total_rates,item_weight_action_rate] 

def item_action_total_user_counts():
    name = 'item_action_total_user_counts'
    grouped = user_action_sequence_product_num[['sku_id', 'user_id']].groupby('sku_id')['user_id'].nunique().reset_index()
    grouped.columns = ['sku_id', name]
    item_action_total_user_counts = min_max_normalize(grouped, name)    
    return item_action_total_user_counts

item_action_total_user_counts = item_action_total_user_counts()

def item_want_user_counts():
    name = 'item_want_user_counts'
    mask = pd.Series(list(map(lambda x: True if x>0 else False, user_action_sequence_product_num['action_want_num'])))
    grouped = user_action_sequence_product_num[['sku_id', 'user_id']].groupby('sku_id')['user_id'].nunique().reset_index()
    grouped.columns = ['sku_id', name]
    item_want_user_counts = min_max_normalize(grouped, name)    
    return item_want_user_counts

item_want_user_counts = item_want_user_counts()

def item_consume_user_counts():
    name = 'item_consume_user_counts'
    mask = pd.Series(list(map(lambda x: True if x>0 else False, user_action_sequence_product_num['action_order_num'])))
    grouped = user_action_sequence_product_num[['sku_id', 'user_id']].groupby('sku_id')['user_id'].nunique().reset_index()
    grouped.columns = ['sku_id', name]
    item_consume_user_counts = min_max_normalize(grouped, name)    
    return item_consume_user_counts

item_consume_user_counts = item_consume_user_counts()

item_user_counts = [item_action_total_user_counts, item_want_user_counts, item_consume_user_counts]

def item_consume_action_rate():
    name = 'item_consume_action_rate'
    grouped = user_action_sequence_product_num.groupby('sku_id',as_index=False)
    # ratio of buying actions to all actions, if not buying, is -1
    ratio_frame = grouped['action_order_num'].sum()['action_order_num']/grouped['action_total_num'].sum()['action_total_num']    
    ratio_frame = ratio_frame.replace(0, -1)
    item_consume_action_rate = pd.concat([product_dataframe['sku_id'],ratio_frame],axis=1)
    item_consume_action_rate.columns = ['sku_id', name] 
    return item_consume_action_rate

item_consume_action_rate = item_consume_action_rate()

def item_consume_want_rate():
    name = 'item_consume_want_rate'
    grouped = user_action_sequence_product_num.groupby('sku_id',as_index=False)
    # ratio of buying actions to all actions, if not buying, is -1
    ratio_frame = grouped['action_order_num'].sum()['action_order_num']/grouped['action_total_num'].sum()['action_total_num']    
    ratio_frame = ratio_frame.replace(0, -1)
    item_consume_want_rate = pd.concat([product_dataframe['sku_id'],ratio_frame],axis=1)
    item_consume_want_rate.columns = ['user_id', name] 
    return item_consume_want_rate

item_consume_want_rate = item_consume_want_rate()

item_consume_rates = [item_consume_action_rate, item_consume_want_rate]

def add_item_features():
	item_features = []
	item_features.extend(item_action_rates)
	user_features.extend(item_user_counts)
	user_features.extend(item_consume_rates)
    item_feature_data = copy.copy(product_dataframe)
	for f in item_features:
        item_feature_data = pd.merge(item_feature_data, f, on ='sku_id', how='left')
	item_feature_data.fillna(-1, inplace=True)
	return item_feature_data

item_feature_data=add_item_features()
item_feature_data.to_csv(filename + filename_2 + 'item_feature_data.csv', index=False, sep=',', header=True)
item_feature_data=pd.read_csv(filename + filename_2 + 'item_feature_data.csv')


#%% user_item_features

def user_action_item_rate():
    user_grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    a = user_grouped['action_total_num'].sum()['action_total_num']
    a = pd.concat([user_dataframe,a],axis=1)
    a.rename(columns={'action_total_num':'user_action_total_num'},inplace=True)
    user_item_grouped = user_action_sequence_product_num.groupby(['user_id','sku_id'],as_index=False)
    b = user_item_grouped['action_total_num'].sum()['action_total_num']
    b = pd.concat([user_product_dataframe,b],axis=1)
    b.rename(columns={'action_total_num':'user_item_action_total_num'},inplace=True)
    user_action_item=pd.merge(b,a,on='user_id',how='left')
    user_action_item_rate=copy.copy(user_product_dataframe)
    user_action_item_rate['user_action_item_rate']=user_action_item['user_item_action_total_num']/user_action_item['user_action_total_num']
    return user_action_item_rate

user_action_item_rate = user_action_item_rate()

def user_want_item_rate():
    user_grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    a = user_grouped['action_want_num'].sum()['action_want_num']
    a = pd.concat([user_dataframe,a],axis=1)
    a.rename(columns={'action_want_num':'user_action_want_num'},inplace=True)
    user_item_grouped = user_action_sequence_product_num.groupby(['user_id','sku_id'],as_index=False)
    b = user_item_grouped['action_want_num'].sum()['action_want_num']
    b = pd.concat([user_product_dataframe,b],axis=1)
    b.rename(columns={'action_want_num':'user_item_action_want_num'},inplace=True)
    user_want_item=pd.merge(b,a,on='user_id',how='left')
    user_want_item_rate=copy.copy(user_product_dataframe)
    user_want_item_rate['user_want_item_rate']=user_want_item['user_item_action_want_num']/user_want_item['user_action_want_num']
    return user_want_item_rate

user_want_item_rate = user_want_item_rate()

def user_consume_item_rate():
    user_grouped = user_action_sequence_product_num.groupby('user_id',as_index=False)
    a = user_grouped['action_order_num'].sum()['action_order_num']
    a = pd.concat([user_dataframe,a],axis=1)
    a.rename(columns={'action_order_num':'user_action_order_num'},inplace=True)
    user_item_grouped = user_action_sequence_product_num.groupby(['user_id','sku_id'],as_index=False)
    b = user_item_grouped['action_order_num'].sum()['action_order_num']
    b = pd.concat([user_product_dataframe,b],axis=1)
    b.rename(columns={'action_order_num':'user_item_action_order_num'},inplace=True)
    user_consume_item=pd.merge(b,a,on='user_id',how='left')
    user_consume_item_rate=copy.copy(user_product_dataframe)
    user_consume_item_rate['user_consume_item_rate']=user_consume_item['user_item_action_order_num']/user_consume_item['user_action_order_num']    
    return user_consume_item_rate

user_consume_item_rate = user_consume_item_rate()

user_item_rates=[user_action_item_rate,user_want_item_rate,user_consume_item_rate]

def add_user_item_features():
	user_item_features=[]
	user_item_features.extend(user_item_rates)
	user_item_feature_data=copy.copy(user_product_dataframe)
	for f in user_item_features:
		user_item_feature_data=pd.merge(user_item_feature_data,f,on=['user_id','sku_id'],how='left')
	return user_item_feature_data

user_item_feature_data=add_user_item_features()
user_item_feature_data.to_csv(filename + filename_2 + 'user_item_feature_data.csv', index=False, sep=',', header=True)
user_item_feature_data=pd.read_csv(filename + filename_2 + 'user_item_feature_data.csv')
