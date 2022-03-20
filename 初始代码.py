# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:52:32 2022

@author: leiyy
"""

import pandas as pd
#from openpyxl import load_workbook
import time
import copy

# 用户数据
user_data = pd.read_csv("E:/代码/user.csv", delimiter = '\t', header = None)
user_data.columns = ['user_id', 'sex', 'age', 'province', 'city', 'country']
user_data = user_data.sort_values(by = 'user_id').reset_index(drop=True)
# 商品数据
product_data = pd.read_csv("E:/代码/product.csv", delimiter = '\t', header = None)
product_data.columns = ['sku_id', 'sku_name', 'first_cate_id',	'second_cate_id', 'third_cate_id', 'first_cate_name', 'second_cate_name', 'third_cate_name']
product_data = product_data.sort_values(by = 'sku_id').reset_index(drop=True)
# 用户浏览数据
action_pv_data = pd.read_csv("E:/代码/action_pv.csv", delimiter = '\t', header = None)
action_pv_data.columns = ['user_id', 'sku_id', 'action_time']
# action_pv_NaN = action_pv_data[action_pv_data.isnull().T.any()]
action_pv_data = action_pv_data[action_pv_data['sku_id'].notna()]
action_pv_data['sku_id'] = action_pv_data['sku_id'].astype('int64')
action_pv_data = action_pv_data.sort_values(by = ['user_id', 'sku_id']).reset_index(drop=True)
# 用户收藏数据
action_fav_data = pd.read_csv("E:/代码/action_fav.csv", delimiter = '\t', header = None)
action_fav_data.columns = ['user_id', 'sku_id', 'action_time']
action_fav_data = action_fav_data.sort_values(by = ['user_id', 'sku_id']).reset_index(drop=True)
# 用户加购物车数据
action_cart_data = pd.read_csv("E:/代码/action_cart.csv", delimiter = '\t', header = None)
action_cart_data.columns = ['user_id', 'sku_id', 'action_time']
action_cart_data = action_cart_data.sort_values(by = ['user_id', 'sku_id']).reset_index(drop=True)
# 用户下单数据
action_order_data = pd.read_csv("E:/代码/action_order.csv", delimiter = '\t', header = None)
action_order_data = action_order_data.drop([0], axis = 1)
action_order_data.columns = ['user_id', 'sku_id', 'sale_qtty', 'after_prefr_amount', 'action_time']
action_order_data = action_order_data.sort_values(by = ['user_id', 'sku_id']).reset_index(drop=True)



# action_pv_data.head(5)
# action_pv_data.loc[(action_pv_data['user_id'] == '-古堡男爵-')]

def append_action_line(var_name, sku_name, user_sku_selected_action):
    append_time = pd.DataFrame(columns = ['action', 'time', 'sale_qtty', 'after_prefr_amount'])
    varname = 'user_selected_' + var_name
    name = eval(varname)
    append_time['time'] = name.loc[name.sku_id == sku_name].action_time
    append_time['action'] = var_name
    if var_name == 'order':
        append_time['sale_qtty'] = name.loc[name.sku_id == sku_name].sale_qtty
        append_time['after_prefr_amount'] = name.loc[name.sku_id == sku_name].after_prefr_amount
    append_time = append_time.reset_index(drop=True)
    user_sku_selected_action = pd.concat([user_sku_selected_action, append_time])
    return user_sku_selected_action


#%%
for user_index in range(0, user_data.shape[0]):
    # 以用户id作为表示，看看一个人会有那些行为
    #user_index = 0
    user_selected = user_data.loc[user_index,['user_id']][0]
    user_selected_pv = action_pv_data.loc[(action_pv_data['user_id'] == user_selected)]
    user_selected_pv = user_selected_pv.reset_index(drop=True)
    user_selected_fav = action_fav_data.loc[(action_fav_data['user_id'] == user_selected)]
    user_selected_fav = user_selected_fav.reset_index(drop=True)
    user_selected_cart = action_cart_data.loc[(action_cart_data['user_id'] == user_selected)]
    user_selected_cart = user_selected_cart.reset_index(drop=True)
    user_selected_order = action_order_data.loc[(action_order_data['user_id'] == user_selected)]
    user_selected_order = user_selected_order.reset_index(drop=True)
    # 查看某一用户针对某一sku的所有行为
    for sku_index in range (0, user_selected_pv['sku_id'].unique().shape[0]):
        #sku_index = 1
        sku_selected = user_selected_pv['sku_id'].unique()[sku_index]
        user_sku_selected_action = pd.DataFrame(columns = ['user_id', 'sku_id', 'action', 'time', 'sale_qtty', 'after_prefr_amount'])
        user_sku_selected_action = append_action_line('pv', sku_selected, user_sku_selected_action)
        user_sku_selected_action = append_action_line('fav', sku_selected, user_sku_selected_action)
        user_sku_selected_action = append_action_line('cart', sku_selected, user_sku_selected_action)
        user_sku_selected_action = append_action_line('order', sku_selected, user_sku_selected_action)
        user_sku_selected_action = user_sku_selected_action.reset_index(drop=True)
        user_sku_selected_action['user_id'] = user_selected
        user_sku_selected_action['sku_id'] = sku_selected
        user_sku_selected_action = user_sku_selected_action.sort_values(by = 'time')
        #outputpath="G:/lyy代码/user_sku_action/action_set"
        if user_index == 0 and sku_index==0:
            user_sku_selected_action.to_csv("E:/代码/user_sku_action/user_action.csv", mode = 'w', index=False, sep=',', header=True)
        else:
            user_sku_selected_action.to_csv("E:/代码/user_action.csv", mode = 'a', index=False, sep=',', header=False)
     
        
        #user_sku_selected_action.to_excel((outputpath + str(user_index) + ".xlsx"), sheet_name = str(sku_index), index=None, header=True)
        #pd.ExcelWriter((outputpath + str(user_index) + ".xlsx"), engine='openpyxl')
        #user_sku_selected_action.to_excel((outputpath + str(user_index) + ".xlsx"), sheet_name = str(sku_index), index=False, header=True)
        #writer = pd.ExcelWriter(str(outputpath + str(user_index) + ".xlsx"), engine='openpyxl')
        #writer = pd.ExcelWriter(path = 'G:/lyy代码/user_sku_action/2.xlsx', engine='openpyxl')
        #book = load_workbook(writer.path)
        #writer.book = book
        #user_sku_selected_action.to_excel(excel_writer=writer,sheet_name=str(sku_index), index=None, header=True)
        #writer.save()
        #writer.close()
        print('user_index = ', user_index)
        print('sku_index = ', sku_index)
            
#%% 
user_action = pd.read_csv("E:/代码/user_sku_action/user_action.csv")
user_action['time'] = pd.to_datetime(user_action['time'])
user_action_0 = user_action
if_drop = 0
for i in range(0, user_action_0.shape[0]):
    #i = 3 
    print(i)
    if if_drop == 1:
        i = i-1
    if i == 0:
        pass
    else:
        if user_action_0['user_id'][i] == user_action_0['user_id'][i-1] and user_action_0['sku_id'][i] == user_action_0['sku_id'][i-1] and user_action_0['action'][i] == user_action_0['action'][i-1]:
            if (user_action_0['time'][i] - user_action_0['time'][i-1]).total_seconds() < 3600:
                user_action_0 = user_action_0.drop([i], axis = 0, inplace=False)
                user_action_0 = user_action_0.reset_index(drop=True)
                if_drop = 1
user_action_0.to_csv("E:/代码/user_sku_action/user_action_0.csv", index=False, sep=',', header=True)                
user_action_1 = user_action_0.head(1178280)
user_action_1.to_csv("E:/代码/user_sku_action/user_action_1.csv", index=False, sep=',', header=True)                

user_num_1 = user_action_1['user_id'].unique().shape[0]  #5955
sku_num_1 = user_action_1['sku_id'].unique().shape[0]  #66021
#%%提取行为序列
user_action_1 = pd.read_csv("E:/代码/user_sku_action/user_action_1.csv")
user_action_1['time'] = pd.to_datetime(user_action_1['time'])
D = user_action_1.groupby(['user_id','sku_id']).count().iloc[:,0]
user_action_sequence = pd.DataFrame(columns = ['user_id', 'sku_id', 'action_sequence'])
count_num = -1
line_end = 0
action_uni_user = user_action_1['user_id'].unique()
action_uni_sku = user_action_1['sku_id'].unique()
for user_index in range(0, action_uni_user.shape[0]):
    #user_index = 0
    start = time.perf_counter()
    user_id_seq = action_uni_user[user_index]
    for sku_index in range(0, action_uni_sku.shape[0]):
        count_num = count_num +1
        #sku_index = 5
        sku_id_seq = copy.copy(action_uni_sku[sku_index])
        user_action_sequence.loc[count_num,'user_id'] = copy.copy(user_id_seq)
        user_action_sequence.loc[count_num,'sku_id'] = copy.copy(sku_id_seq)
        if count_num == 0:
            line_start = 0
        else:
            line_start = line_start + D[count_num-1] 
        line_end = line_end + D[count_num]
        if D[count_num] == 1:
            data_temp = user_action_1.iloc[line_start:line_start+1,2:4].reset_index(drop=True)
            user_action_sequence.loc[count_num,'action_sequence'] = copy.copy(data_temp.iloc[0,0][0])
        else:
            data_temp = user_action_1.iloc[line_start:line_end,2:4].reset_index(drop=True)
            seq_str = data_temp.iloc[0,0][0]
            for i in range(1,D[count_num]):
                if (data_temp.iloc[i,1]-data_temp.iloc[i-1,1]).total_seconds() > 86400: #一天的时间
                    seq_str = seq_str + '-' + data_temp.iloc[i,0][0]
                else:
                    seq_str = seq_str + data_temp.iloc[i,0][0]
            user_action_sequence.loc[count_num,'action_sequence'] = copy.copy(seq_str)
        print('user_index: ', user_index)
        print('sku_index: ', sku_index)
    elapsed = (time.perf_counter() - start)
    print("一个user的时间: ",elapsed)

user_action_sequence.to_csv("E:/代码/user_sku_action/user_action_sequence.csv", index=False, sep=',', header=True)                
action_sequence_total = user_action_sequence['action_sequence'].unique()
#%%
def func(_str):
     _list = list(_str)
     n = len(_list)
     if n <= 1:
        print(_str)
        return
     list1 = []
     for i in range(n-1):
         if _list[i] != _list[i+1]:
             list1.append(_list[i])
     list1.append(_list[-1])
     str1 = ''.join(list1)
     return str1

#print(func('pp'))
user_action_sequence_0 = pd.DataFrame(columns = ['user_id', 'sku_id', 'action_sequence'])
user_action_sequence_0 = copy.copy(user_action_sequence)
for i in range(0, user_action_sequence.shape[0]):
    print(i)
    if len(user_action_sequence.loc[i,'action_sequence'])==1:
        user_action_sequence_0.loc[i,'action_sequence'] = user_action_sequence.loc[i,'action_sequence']
    else:
        user_action_sequence_0.loc[i,'action_sequence'] = func(user_action_sequence.loc[i,'action_sequence'])

action_sequence_0_total = user_action_sequence_0['action_sequence'].unique()        
        
#%%
        