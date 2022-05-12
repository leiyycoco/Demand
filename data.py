

import pandas as pd
#from openpyxl import load_workbook
import time
import copy
import random

#%% 
filename = "G:/"
filename_2 = "1/"

user_num_extract = 10000
date_extract_start = '2021-02-01 00:00:00'
date_extract_end = '2021-03-15 00:00:00'

#%% read data
start_read = time.perf_counter()
# user
user_data = pd.read_csv(filename + "user.csv", delimiter = '\t', header = None)
user_data.columns = ['user_id', 'sex', 'age', 'province', 'city', 'country']
user_data = user_data.sort_values(by = 'user_id').reset_index(drop=True)
# product
product_data = pd.read_csv(filename + "product.csv", delimiter = '\t', header = None)
product_data.columns = ['sku_id', 'sku_name', 'first_cate_id',	'second_cate_id', 'third_cate_id', 'first_cate_name', 'second_cate_name', 'third_cate_name']
product_data = product_data.sort_values(by = 'sku_id').reset_index(drop=True)
# user preview
action_pv_data = pd.read_csv(filename + "action_pv.csv", delimiter = '\t', header = None)
action_pv_data.columns = ['user_id', 'sku_id', 'action_time']
# action_pv_NaN = action_pv_data[action_pv_data.isnull().T.any()]
action_pv_data = action_pv_data[action_pv_data['sku_id'].notna()]
action_pv_data['sku_id'] = action_pv_data['sku_id'].astype('int64')
action_pv_data['action_time'] = pd.to_datetime(action_pv_data['action_time'])
action_pv_data = action_pv_data.sort_values(by = ['action_time','user_id','sku_id']).reset_index(drop=True)
# user favor
action_fav_data = pd.read_csv(filename + "action_fav.csv", delimiter = '\t', header = None)
action_fav_data.columns = ['user_id', 'sku_id', 'action_time']
action_fav_data['action_time'] = pd.to_datetime(action_fav_data['action_time'])
action_fav_data = action_fav_data.sort_values(by = ['action_time','user_id', 'sku_id']).reset_index(drop=True)
# user cart 
action_cart_data = pd.read_csv(filename + "action_cart.csv", delimiter = '\t', header = None)
action_cart_data.columns = ['user_id', 'sku_id', 'action_time']
action_cart_data['action_time'] = pd.to_datetime(action_cart_data['action_time'])
action_cart_data = action_cart_data.sort_values(by = ['action_time','user_id', 'sku_id']).reset_index(drop=True)
# user order
action_order_data = pd.read_csv(filename + "action_order.csv", delimiter = '\t', header = None)
action_order_data = action_order_data.drop([0], axis = 1)
action_order_data.columns = ['user_id', 'sku_id', 'sale_qtty', 'after_prefr_amount', 'action_time']
action_order_data['action_time'] = pd.to_datetime(action_order_data['action_time'])
action_order_data = action_order_data.sort_values(by = ['action_time','user_id', 'sku_id']).reset_index(drop=True)
elapsed_read = (time.perf_counter() - start_read)
print("The time for reading data： ",elapsed_read)   
print('********************Data reading done!*****************')

#%% Extract partial data from action sequence
start_extract = time.perf_counter()
action_pv_data_oneweek = action_pv_data.loc[(action_pv_data['action_time'] >= date_extract_start) & (action_pv_data['action_time'] <= date_extract_end)]
#user_pv_oneweek = action_pv_data_oneweek['user_id'].unique().shape[0]  
#sku_pv_oneweek = action_pv_data_oneweek['sku_id'].unique().shape[0]  
action_fav_data_oneweek = action_fav_data.loc[(action_fav_data['action_time'] >= date_extract_start) & (action_fav_data['action_time'] <= date_extract_end)]
action_cart_data_oneweek = action_cart_data.loc[(action_cart_data['action_time'] >= date_extract_start) & (action_cart_data['action_time'] <= date_extract_end)]
action_order_data_oneweek = action_order_data.loc[(action_order_data['action_time'] >= date_extract_start) & (action_order_data['action_time'] <= date_extract_end)]

action_pv_data_oneweek.to_csv(filename + filename_2 + "action_pv_data_oneweek.csv", index=False, sep=',', header=True)                
action_fav_data_oneweek.to_csv(filename + filename_2 + "action_fav_data_oneweek.csv", index=False, sep=',', header=True)                
action_cart_data_oneweek.to_csv(filename + filename_2 + "action_cart_data_oneweek.csv", index=False, sep=',', header=True)                
action_order_data_oneweek.to_csv(filename + filename_2 + "action_order_data_oneweek.csv", index=False, sep=',', header=True)                
elapsed_extract = (time.perf_counter() - start_extract)
print("The time for extracting data： ",elapsed_extract)   
print('********************Data extracting done*****************')

#%% extract part user randomly
start_randomuser = time.perf_counter()
list_user_pv = list(action_pv_data_oneweek['user_id'].unique())
list_user_pv_random = random.sample(list_user_pv, user_num_extract)
list_user_pv_random = pd.DataFrame(list_user_pv_random)
list_user_pv_random.to_csv(filename + filename_2 + "list_user_pv_random.csv", index=False, sep=',', header=True) 
#list_user_pv_random = pd.read_csv(filename + filename_2 + "list_user_pv_random.csv")   
list_user_pv_random = list(list_user_pv_random.iloc[:,0])             
action_pv_data_oneweek = action_pv_data_oneweek[action_pv_data_oneweek["user_id"].isin(list_user_pv_random)]
action_fav_data_oneweek = action_fav_data_oneweek[action_fav_data_oneweek["user_id"].isin(list_user_pv_random)]
action_cart_data_oneweek = action_cart_data_oneweek[action_cart_data_oneweek["user_id"].isin(list_user_pv_random)]
action_order_data_oneweek = action_order_data_oneweek[action_order_data_oneweek["user_id"].isin(list_user_pv_random)]
elapsed_randomuser = (time.perf_counter() - start_randomuser)
print("The time for extracting partial user： ",elapsed_randomuser) 

#%% delete duplication
print('+++++++Data to heavy++++++++++++')
#action_pv_data_oneweek = pd.read_csv(filename + filename_2 + "action_pv_data_oneweek.csv")
#action_fav_data_oneweek = pd.read_csv(filename + filename_2 + "action_fav_data_oneweek.csv")
#action_cart_data_oneweek = pd.read_csv(filename + filename_2 + "action_cart_data_oneweek.csv")
#action_order_data_oneweek = pd.read_csv(filename + filename_2 + "action_order_data_oneweek.csv")

action_pv_data_oneweek = action_pv_data_oneweek.sort_values(by = ['user_id','sku_id','action_time']).reset_index(drop=True)
action_fav_data_oneweek = action_fav_data_oneweek.sort_values(by = ['user_id','sku_id','action_time']).reset_index(drop=True)
action_cart_data_oneweek = action_cart_data_oneweek.sort_values(by = ['user_id','sku_id','action_time']).reset_index(drop=True)
action_order_data_oneweek = action_order_data_oneweek.sort_values(by = ['user_id','sku_id','action_time']).reset_index(drop=True)

def delte_repeatdata(user_action):
    #user_action = copy.copy(action_pv_data_oneweek)
    user_action_0 = copy.copy(user_action)
    if_drop = 0
    j = 0
    i = 0
    while i < user_action_0.shape[0]:
    #while i < 1000:    
        print('i= ', i)
        print('j= ', j)
        if if_drop == 1:
            i = copy.copy(j)
            #print('---i= ',i)
        if i == 0:
            #print('000000000i= ',i)
            pass
        else:
            if_drop = 0
            if (user_action_0['user_id'][i] == user_action_0['user_id'][i-1]) & (user_action_0['sku_id'][i] == user_action_0['sku_id'][i-1]):
                #print('%%%%%% i = ', i)
                #print (user_action_0['user_id'][i])
                #print (user_action_0['user_id'][i-1])
                #print(user_action_0['sku_id'][i])
                #print(user_action_0['sku_id'][i-1])
                #print ('&&&&&&&&')
                if abs((user_action_0['action_time'][i] - user_action_0['action_time'][i-1]).total_seconds()) < 3600:
                    j = copy.copy(i)
                    #print('++++++++++i= ',i)
                    #print('++++++++++j= ',j)
                    user_action_0.drop([i], axis = 0, inplace=True)
                    user_action_0 = user_action_0.reset_index(drop=True)
                    if_drop = 1   
                    #print('++++++++++if_drop= ',if_drop)
                    #A = user_action_0.head(50)
            #print('**********if_drop= ',if_drop)
        i = i+1
    return user_action_0

start_1 = time.perf_counter()
action_pv_data_oneweek_delterepeat = delte_repeatdata(action_pv_data_oneweek)   
elapsed_1 = (time.perf_counter() - start_1)
print("The time for data to heavy of pv",elapsed_1)   
action_pv_data_oneweek_delterepeat.to_csv(filename + filename_2 + "action_pv_data_oneweek_delterepeat.csv", index=False, sep=',', header=True)                

   
start_2 = time.perf_counter()
action_fav_data_oneweek_delterepeat = delte_repeatdata(action_fav_data_oneweek)
elapsed_2 = (time.perf_counter() - start_2)
print("The time for data to heavy of fav",elapsed_2)    
action_fav_data_oneweek_delterepeat.to_csv(filename + filename_2 + "action_fav_data_oneweek_delterepeat.csv", index=False, sep=',', header=True)                


start_3 = time.perf_counter() 
action_cart_data_oneweek_delterepeat = delte_repeatdata(action_cart_data_oneweek)
elapsed_3 = (time.perf_counter() - start_3)
print("The time for data to heavy of cart",elapsed_3)  
action_cart_data_oneweek_delterepeat.to_csv(filename + filename_2 + "action_cart_data_oneweek_delterepeat.csv", index=False, sep=',', header=True)                


start_4 = time.perf_counter()   
action_order_data_oneweek_delterepeat = delte_repeatdata(action_order_data_oneweek)       
elapsed_4 = (time.perf_counter() - start_4)
print("The time for data to heavy of order",elapsed_4)       
action_order_data_oneweek_delterepeat.to_csv(filename + filename_2 + "action_order_data_oneweek_delterepeat.csv", index=False, sep=',', header=True)                


#%% Date merge
start_5 = time.perf_counter()
action_pv_data_oneweek_delterepeat.insert(2, 'action', "pv")
action_fav_data_oneweek_delterepeat.insert(2, 'action', "fav")
action_cart_data_oneweek_delterepeat.insert(2, 'action', "cart")
action_order_data_oneweek_delterepeat.insert(2, 'action', "order")
action_order_data_oneweek_delterepeat_2 = action_order_data_oneweek_delterepeat.drop('sale_qtty', axis=1, inplace=False)
action_order_data_oneweek_delterepeat_2.drop('after_prefr_amount', axis=1, inplace=True) 
action_data_oneweek = pd.concat([action_pv_data_oneweek_delterepeat, action_fav_data_oneweek_delterepeat, action_cart_data_oneweek_delterepeat, action_order_data_oneweek_delterepeat_2])

action_data_oneweek = action_data_oneweek.sort_values(by = ['user_id','sku_id','action_time']).reset_index(drop=True)

action_data_oneweek.to_csv(filename + filename_2 + "action_data_oneweek.csv", index=False, sep=',', header=True)                

elapsed_5 = (time.perf_counter() - start_5)
print("The time for data merge",elapsed_5)  

#%% extract action sequence
#action_data_oneweek = pd.read_csv(filename + filename_2 + "action_data_oneweek.csv")
action_data_oneweek['action_time'] = pd.to_datetime(action_data_oneweek['action_time'])
count_sku = action_data_oneweek.groupby(['user_id']).count().iloc[:,0]
count_action = action_data_oneweek.groupby(['user_id','sku_id']).count().iloc[:,0]
user_action_sequence_oneweek = pd.DataFrame(columns = ['user_id', 'sku_id', 'action_sequence'])
count_num = -1
action_uni_user_oneweek = action_data_oneweek['user_id'].unique()
#action_uni_sku_oneweek = action_data_oneweek['sku_id'].unique()
for user_index in range(0, action_uni_user_oneweek.shape[0]):
    #user_index = 1
    start_extract = time.perf_counter()
    user_id_seq_oneweek = copy.copy(action_uni_user_oneweek[user_index])
    if user_index == 0:
        sku_uni_start = 0
        sku_uni_end = count_sku[user_index]-1
    else:
        sku_uni_start = sku_uni_end + 1
        sku_uni_end = sku_uni_end + count_sku[user_index]
    sku_uni_seq_oneweek = action_data_oneweek.iloc[sku_uni_start:sku_uni_end+1,1].unique()
    for sku_index in range(0, sku_uni_seq_oneweek.shape[0]):
        count_num = count_num +1
        #sku_index = 5
        sku_id_seq_oneweek = copy.copy(sku_uni_seq_oneweek[sku_index])
        user_action_sequence_oneweek.loc[count_num,'user_id'] = copy.copy(user_id_seq_oneweek)
        user_action_sequence_oneweek.loc[count_num,'sku_id'] = copy.copy(sku_id_seq_oneweek)
        if count_num == 0:
            line_start = 0
            line_end = count_action[count_num] - 1
        else:
            line_start = line_end + 1 
            line_end = line_end + count_action[count_num]
        data_temp = action_data_oneweek.iloc[line_start:line_end+1,2:4].reset_index(drop=True)
        if count_action[count_num] == 1:
            user_action_sequence_oneweek.loc[count_num,'action_sequence'] = copy.copy(data_temp.iloc[0,0][0])
        else:
            seq_str = copy.copy(data_temp.iloc[0,0][0])
            for i in range(1,count_action[count_num]):
                if (data_temp.iloc[i,1]-data_temp.iloc[i-1,1]).total_seconds() > 86400: #一天的时间
                    seq_str = seq_str + '-' + data_temp.iloc[i,0][0]
                else:
                    seq_str = seq_str + data_temp.iloc[i,0][0]
            user_action_sequence_oneweek.loc[count_num,'action_sequence'] = copy.copy(seq_str)
        print('user_index: ', user_index)
        print('sku_index: ', sku_index)
    elapsed_extract = (time.perf_counter() - start_extract)
    print("The time for extract a user action sequence",elapsed_extract)

user_action_sequence_oneweek.to_csv(filename + filename_2 + "user_action_sequence_oneweek.csv", index=False, sep=',', header=True)                
action_sequence_total = user_action_sequence_oneweek['action_sequence'].unique()

#%%
        