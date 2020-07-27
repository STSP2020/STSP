import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
import progressbar
import itertools
import math
from math import sin, cos, sqrt, atan2, radians
import collections
from six.moves import cPickle as pickle 
import random
from pandas import Series, DataFrame
import csv
import pickle


def Category_Encoder(args):
    print('Start category data preprocess')
    augment_sample = args.augment_sample
    min_seq_len = args.min_seq_len
    min_seq_num = args.min_seq_num
    category_neg_sample_num = args.category_neg_sample_num
    data=pd.read_csv('data/'+ args.dataset +'_checkin_reindexed.csv')
    # form visit sequences 
    visit_sequences, max_seq_len, valid_visits, user_reIndex_mapping = generate_sequence(data, min_seq_len, min_seq_num)
    if augment_sample:
        visit_sequences, ground_truth_dict = aug_sequence(visit_sequences, min_len=3)
    # generate new id  sequence
    POI_sequences, POI_reIndex_mapping = generate_new_sequences(data, visit_sequences)
    # generate location id sequence
    Location_sequences, Location_reIndex_mapping = generate_location_sequences(data, visit_sequences)
    # generate time (hour) sequence
    time_sequences = generate_time_sequences(data, visit_sequences)
    # generate time (weekday or weekend) sequence
    weekend_sequences = generate_weekend_sequences(data, visit_sequences)
    # generate new id type sequence
    type_sequences = generate_type_sequence(data, visit_sequences)
    # generate category id sequence
    cat_sequences, cat_reIndex_mapping = generate_cat_sequences(data, visit_sequences)
    # generate new id ground truth for each sequence
    ground_truth_sequences = generate_new_ground_truth_sequences(data, ground_truth_dict, POI_reIndex_mapping)
    # generate location id ground truth for each sequence
    location_ground_truth_sequences = generate_location_ground_truth_sequences(data, ground_truth_dict, Location_reIndex_mapping)
    # generate category sequence
    specific_cate_sequences = generate_specific_cate_sequences(data, ground_truth_dict)
    # generate location sequence
    specific_location_sequences = generate_specific_location_sequences(data, ground_truth_dict)
    # generate collective POI's category distribution
    poi_cat_distrib = generate_cat_distrib(data, valid_visits, POI_reIndex_mapping, cat_reIndex_mapping)
    # generate category transition data
    count_dict= matrix_preparation(POI_sequences)    
    category_transition= Category_Transition(cat_reIndex_mapping,count_dict,poi_cat_distrib,144)#144 is the first collective POI index
    # generate negative sample
    neg_sequences = generate_neg_sequences(POI_sequences,category_neg_sample_num, data, POI_reIndex_mapping,cat_reIndex_mapping)
    # form sample sets
    sample_sets = form_sample_sets(POI_sequences, time_sequences,weekend_sequences, type_sequences, cat_sequences, ground_truth_sequences,specific_cate_sequences,Location_sequences, location_ground_truth_sequences, neg_sequences)

    #save data
    dir = './'+args.dataset+'/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_dict(sample_sets, dir + 'sample_sets.pkl')
    np.save(dir + 'POI_reIndex_mapping.npy', POI_reIndex_mapping)
    np.save(dir + 'Location_reIndex_mapping.npy', Location_reIndex_mapping)
    np.save(dir + 'user_reIndex_mapping.npy', user_reIndex_mapping)
    np.save(dir + 'cat_reIndex_mapping.npy', cat_reIndex_mapping)
    save_dict(poi_cat_distrib, dir + 'poi_cat_distrib.pkl')
    np.save(dir + 'max_seq_len.npy', max_seq_len)
    np.save(dir + 'category_neg_sample_num.npy', category_neg_sample_num) 
    np.save(dir + 'category_transition_marix.npy',category_transition)
    
    
    list_L2_id=[]
    for i in range(len(cat_reIndex_mapping)):
        print('[',i,',',cat_reIndex_mapping[i],']')
        list_L2_id.append(cat_reIndex_mapping[i])    
    L2_id=pd.DataFrame(list_L2_id)
    L2_id.to_csv('category result/'+args.dataset+'L2_id_mapping_'+args.dataset+'.csv')
    
    
# documents needed from category encoder
#'category result/CAL/reindex_data_CAL.csv'         %%%%%  all reindexed checkin data
#'category result/CAL/L2_id_mapping_CAL.csv'        %%%%%  mapping index——>true_L2_id
#'category result/CAL/train_CAL.txt'                %%%%%  train data used in category encoder
#'category result/CAL/result_CAL.txt'               %%%%%  category list and POI Type recommanded in category encoder; the POI ground truth

def POI_Encoder(args):
    print('Start POI data preprocess')
    City = args.dataset
    Category = args.category_colname
    df_mall = pd.read_csv('data/mall_Info_' + str(City) + '.csv')
    df_check = pd.read_csv('data/' + str(City) + '_checkin.csv')
    df_check_filtered = pd.read_csv('category result/' + str(City) + '/reindex_data_' + str(City) + '.csv')#
    df_POI = DataFrame(df_check,columns = ['POI_id','Location_id',Category,'POI_id_Longitude','POI_id_Latitude','POI_Type','POI_id_Popular','category_Pnum','stars'])
    df_POI = df_POI.drop_duplicates() 
    df_POI.reset_index(drop = True,inplace = True)
    
    location_old2new = DataFrame(df_check_filtered,columns = ['Location_id','location_id_reindex'])
    location_old2new = location_old2new.drop_duplicates()
    location_old2new = location_old2new.sort_values(by = 'location_id_reindex')
    location_old2new.reset_index(drop = True,inplace = True)
    
    df_POI_filtered = pd.merge(df_POI,location_old2new)
    df_POI_filtered = df_POI_filtered.drop_duplicates() 
    df_POI_filtered.reset_index(drop = True,inplace = True)
    df_POI_filtered.to_csv('data/' + str(City) + '/1_lstm_POI.csv')
    
    locationID_mapping = dict() #locationID_mapping[location_id_reindex]:
                                #(POI_id,Location_id,Category_2,POI_id_Longitude,POI_id_Latitude,POI_Type,POI_id_Popular,category_Pnum,stars)
    for i in range(len(df_POI_filtered.axes[0])):
        location_id_reindex = df_POI_filtered['location_id_reindex'][i]
        POI_id = df_POI_filtered['POI_id'][i]
        Location_id = df_POI_filtered['Location_id'][i]
        Category_2 = df_POI_filtered[Category][i]
        POI_id_Longitude = df_POI_filtered['POI_id_Longitude'][i]
        POI_id_Latitude = df_POI_filtered['POI_id_Latitude'][i]
        POI_Type = df_POI_filtered['POI_Type'][i]
        POI_id_Popular = df_POI_filtered['POI_id_Popular'][i]
        category_Pnum = df_POI_filtered['category_Pnum'][i]
        stars = df_POI_filtered['stars'][i]
        locationID_mapping[location_id_reindex] = [POI_id,Location_id,Category_2,POI_id_Longitude,POI_id_Latitude,POI_Type,POI_id_Popular,category_Pnum,stars]
    
    df_userID_mapping = DataFrame(df_check_filtered,columns = ['User_id','user_id_reindex'])
    df_userID_mapping = df_userID_mapping.drop_duplicates()
    df_userID_mapping.reset_index(drop = True,inplace = True)
    userID_mapping = dict()# user_id_reindex->User_id
    userID_old2new = dict()# User_id->user_id_reindex
    for i in range(len(df_userID_mapping.axes[0])):
        user_id_reindex = df_userID_mapping['user_id_reindex'][i]
        User_id = df_userID_mapping['User_id'][i]
        userID_mapping[user_id_reindex] = User_id
    for i in range(len(df_userID_mapping.axes[0])):
        user_id_reindex = df_userID_mapping['user_id_reindex'][i]
        User_id = df_userID_mapping['User_id'][i]
        userID_old2new[User_id] = user_id_reindex
    with open('data/' + str(City) + '/1_lstm_userID_old2new.pickle', "wb") as fp:   #存储
        pickle.dump(userID_old2new, fp, protocol = pickle.HIGHEST_PROTOCOL)
        
    df_categoryID_mapping = pd.read_csv('category result/' + str(City) + '/L2_id_mapping_' + str(City) + '.csv')
    categoryResult_mapping = dict()#result->Category
    for i in range(len(df_categoryID_mapping)):
        index = df_categoryID_mapping['index'][i]
        L2_id = df_categoryID_mapping['true_L2_id'][i]
        df_category = df_check_filtered[df_check_filtered['L2_id'] == L2_id]
        df_category.reset_index(drop = True,inplace = True)
        category = df_category[Category][0]
        categoryResult_mapping[index] = category
    
#Train---------------------------------------------------------------------------------------------------------------------
    df_train_org = DataFrame(columns = ['User_id','Location_id_seq','Location_id1','Location_id2'])
    with open('category result/' + str(City) + '/train_' + str(City) + '.txt', 'rb') as filein:
        for line in filein:
            #print(line)
            #print( type(line.decode()))
            line_list = line.decode().strip('()').split(', [')
            User_id = line_list[0]
            Location_id_seq = line_list[1][:-4].split(',')
            if Location_id_seq == 2:
                Location_id1 = Location_id_seq[0]
                Location_id2 = Location_id_seq[1]
                df_train_org = df_train_org.append(DataFrame({'User_id':[User_id],'Location_id_seq':[Location_id_seq],'Location_id1':[Location_id1],'Location_id2':[Location_id2]}))
            else:
                p = 0
                while p<(len(Location_id_seq)-1):
                    Location_id1 = Location_id_seq[p]
                    Location_id2 = Location_id_seq[p+1]
                    df_train_org = df_train_org.append(DataFrame({'User_id':[User_id],'Location_id_seq':[Location_id_seq],'Location_id1':[Location_id1],'Location_id2':[Location_id2]}))
                    p+=1
    df_train_org = df_train_org.drop(columns = 'Location_id_seq',axis = 1)
    df_train_org = df_train_org.drop_duplicates()
    df_train_org.reset_index(drop = True,inplace = True)
    #Positive sample of training--------------------------------------
    df_Positive_seq = trans_Train_data(df_train_org,locationID_mapping,userID_mapping,df_check)
    #Negitive sample of training--------------------------------------
    df_Negitive_seq1 = prepare_Neg_Seq(df_Positive_seq,df_check,df_POI_filtered,Category)
    df_Positive_seq.to_csv('data/' + str(City) + '/1_lstm_positive_data.csv')
    df_Negitive_seq1.to_csv('data/' + str(City) + '/1_lstm_negetive_data.csv')
    
    df_Train = df_Positive_seq.append(df_Negitive_seq1)
    df_Train.reset_index(drop = True,inplace = True)
    for i in range(len(df_Train.axes[0])):
        df_Train['user_id_reindex'][i] = userID_old2new[df_Train['User_id'][i]]
    df_Train.to_csv('data/' + str(City) + '/1_lstm_Train.csv')
    
#Test--------------------------------------------------------------------------------------------------------------------
    Type_mapping = dict()
    Type_mapping[1] = 'Combined'
    Type_mapping[0] = 'Independent'
    df_result = DataFrame(columns = ['User_id','POI_id1','POI_id2','Location_id1','Location_id2','Pred_Type','Category1','Category5','Category10'])
    with open('category result/' + str(City) + '/result_' + str(City) + '.txt', 'rb') as filein:
        for line in filein:
            #print( type(line.decode()))
            line_list = line.decode().strip('[').split('], [')
            #print(line_list[0].split(', ['))
            user_id_reindex = int(line_list[0].split(', [')[0])
            User_id = userID_mapping[user_id_reindex]
            location1_id_reindex = int(line_list[3].split(',')[-2])
            POI_id1 = locationID_mapping[location1_id_reindex][0]
            Location_id1 = locationID_mapping[location1_id_reindex][1]
            location2_id_reindex = int(line_list[3].split(',')[-1])
            POI_id2 = locationID_mapping[location2_id_reindex][0]
            Location_id2 = locationID_mapping[location2_id_reindex][1]
            Type_ID = int(line_list[7])
            Pred_Type = Type_mapping[Type_ID]
            Category1_id = int(line_list[8])
            Category1 = categoryResult_mapping[Category1_id]

            Category5_list = line_list[9].split(',')
            Category5 = []
            for i in range(5):
                Category5_ID = int(Category5_list[i])
                Category = categoryResult_mapping[Category5_ID]
                Category5.append(Category)

            Category10_list = line_list[10][:-3].split(',')
            Category10 = []
            for j in range(10):
                Category10_ID = int(Category10_list[j])
                Cate = categoryResult_mapping[Category10_ID]
                
                Category10.append(Cate)
            df_result = df_result.append(DataFrame({'User_id':[User_id],'POI_id1':[POI_id1],'POI_id2':[POI_id2],'Location_id1':[Location_id1],'Location_id2':[Location_id2],'Pred_Type':[Pred_Type],'Category1':[Category1],'Category5':[Category5],'Category10':[Category10]}))
    df_result.reset_index(drop = True,inplace =True)
    df_result
    df_result.to_csv('data/' + str(City) + '/1_lstm_Test.csv')


#-----------------------------------------------function about category encoder date procesess--------------------------------------------

#data preparation for category transition
def matrix_preparation(POI_sequences):
    all_matrix_data=[]
    split_seq=[]
    for i in range(len(POI_sequences)):
        for j in range(len(POI_sequences[i])):
            all_matrix_data.append(POI_sequences[i][j])
            
    for i in range(len(all_matrix_data)):
        list_=[]
        if len(all_matrix_data[i])<3:
            list_.append(all_matrix_data[i])
            split_seq.append(list_)
        else:
            list11=[]
            for j in range(len(all_matrix_data[i])-1):
                list1=[]
                list1.append(all_matrix_data[i][j])
                list1.append(all_matrix_data[i][j+1])
                list11.append(list1)
            split_seq.append(list11)
    
    all_split=[]
    for i in range(len(split_seq)):
        for j in range(len(split_seq[i])):
            all_split.append(split_seq[i][j])
    list_tuple=[]
    for i in range(len(all_split)):
        s=tuple(all_split[i])
        list_tuple.append(s)
    
    count_dict={}
    for i in list_tuple:
        count_dict[i]=count_dict.get(i,0)+1 
        
    return count_dict

#category transition  matrix
def Category_Transition(cat_reIndex_mapping,count_dict,poi_cat_distrib,first_mall_index):    
    category_transition= np.zeros([len(cat_reIndex_mapping), len(cat_reIndex_mapping)])
    sum_dict=sum(count_dict.values())
    for (m,n) in count_dict.keys():
        if (m >(first_mall_index-1)) & (n<first_mall_index):
            for cate_id in poi_cat_distrib[m].keys():
                total_num=sum(poi_cat_distrib[m].values())
                ratio=poi_cat_distrib[m][cate_id] / total_num 
                category_transition[cate_id][n]+=ratio*count_dict[m,n]
        else:
            if (m <first_mall_index)& (n>(first_mall_index-1)):
                for cate_id in poi_cat_distrib[n].keys():
                    total_num=sum(poi_cat_distrib[n].values())
                    ratio=poi_cat_distrib[n][cate_id] / total_num 
                    category_transition[m][cate_id]+=ratio*count_dict[m,n]
            else:
                if (m>(first_mall_index-1)) & (n>(first_mall_index-1)):
                    total_num1=sum(poi_cat_distrib[m].values())
                    total_num2=sum(poi_cat_distrib[n].values())
                    for cate_id1 in poi_cat_distrib[m].keys():
                        ratio1=poi_cat_distrib[m][cate_id1] / total_num1
                        for cate_id2 in poi_cat_distrib[n].keys():
                            ratio2=poi_cat_distrib[n][cate_id2]/ total_num2 
                            category_transition[cate_id1][cate_id2]+=ratio1*ratio2*count_dict[m,n]
                else:
                     category_transition[m][n]=count_dict[m,n]
    category_transition=category_transition/sum_dict
    return category_transition

#generate visit sequences for each user
def generate_sequence(input_data, min_seq_len, min_seq_num):  
            
    bar = progressbar.ProgressBar(maxval=input_data.index[-1], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_sg_time'])
    
    total_sequences_dict = {} 
    
    max_seq_len = 0 

    valid_visits = []
    
    bar.start()
    for user in input_data['User_id'].unique():
        user_visits = input_data[input_data['User_id'] == user]
        user_sequences = [] 

        unique_date_group = user_visits.groupby([user_visits['Local_sg_time'].dt.date]) 
        for date in unique_date_group.groups:
            single_date_visit = unique_date_group.get_group(date)
            single_sequence = _remove_consecutive_visit(single_date_visit, bar) 
            
            if len(single_sequence) >= min_seq_len: 
                user_sequences.append(single_sequence)
                if len(single_sequence) > max_seq_len: 
                    max_seq_len = len(single_sequence) 
            
        if len(user_sequences) >= min_seq_num:
            total_sequences_dict[user]=np.array(user_sequences)
            valid_visits = valid_visits + list(itertools.chain.from_iterable(user_sequences))

    bar.finish()

    user_reIndex_mapping = np.array(list(total_sequences_dict.keys()))
    
    return total_sequences_dict, max_seq_len, valid_visits, user_reIndex_mapping


#augment each sequence to increase sample size
def aug_sequence(input_sequence_dict, min_len):

    augmented_sequence_dict, ground_truth_dict = {}, {}
    
    for user in input_sequence_dict.keys():
        
        user_sequences, ground_truth_sequence = [], []
        
        for seq in input_sequence_dict[user]:
            if len(seq)>min_len:
                for i in range(len(seq)-min_len+1): 
                    user_sequences.append(seq[0:i+min_len])
                    ground_truth_sequence.append(seq[i+min_len-1:])
            else: 
                user_sequences.append(seq)
                ground_truth_sequence.append([seq[-1]])
        
        augmented_sequence_dict[user] = np.array(user_sequences)
        ground_truth_dict[user] = np.array(ground_truth_sequence)
        
    return augmented_sequence_dict, ground_truth_dict



# flatten a 3d list into 1d list    
def _flatten_3d_list(input_list):
    
    twoD_lists = input_list.flatten()
    
    return np.hstack([np.hstack(twoD_list) for twoD_list in twoD_lists])


# given an old id and a mapping, return the new id
def _old_id_to_new(mapping, old_id):
  
    return np.where(mapping == old_id)[0].flat[0]



# given an old id and a mapping, return the new id
def _new_id_to_old(mapping, new_id):
  
    return mapping[new_id]



# reIndex an id-list to form consecutive id started from 0
def _reIndex_3d_list(input_list):
    
    flat_list = _flatten_3d_list(input_list)

    index_map = np.unique(flat_list)

    if index_map[0] == -1:
        index_map = np.delete(index_map, 0)

    reIndexed_list = [] 
    
    for user in input_list:
        
        reIndexed_user_list = [] 
        
        for seq in user:
            reIndexed_user_list.append([_old_id_to_new(index_map, poi) if poi != -1 else -1 for poi in seq])
            
        reIndexed_list.append(reIndexed_user_list)
        
    reIndexed_list = np.array(reIndexed_list)

    check_list = _flatten_3d_list(reIndexed_list)

    if -1 in check_list:
        check_list = check_list[ check_list >= 0 ]
    
    check_is_consecutive(check_list, 0) 
    

    return reIndexed_list, index_map


# generate POI id (with new id) sequences for each valid user   
def generate_new_sequences(input_data, visit_sequence_dict):
    
    POI_sequences = []
    
    for user in visit_sequence_dict:
        
        user_POI_sequences = []
        
        for seq in visit_sequence_dict[user]:
            
            POI_sequence = []
            
            for visit in seq:
                if visit != -1:
                    POI_sequence.append(input_data['new_id'][visit])
                else: 
                    POI_sequence.append(-1)
            
            user_POI_sequences.append(POI_sequence)
        
        POI_sequences.append(user_POI_sequences)

    reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences))
    
    return reIndexed_POI_sequences, POI_reIndex_mapping
  

# generate location id sequences for each valid user 
def generate_location_sequences(input_data, visit_sequence_dict):
    POI_sequences = []
    for user in visit_sequence_dict:
        user_POI_sequences = []
        for seq in visit_sequence_dict[user]:
            POI_sequence = []
            for visit in seq:
                if visit != -1:
                    POI_sequence.append(input_data['Location_id'][visit])
                else: 
                    POI_sequence.append(-1)
            user_POI_sequences.append(POI_sequence)
        POI_sequences.append(user_POI_sequences)
    reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences))
    return reIndexed_POI_sequences, POI_reIndex_mapping


# generate POI type sequences for each valid user 
def generate_type_sequence(input_data, visit_sequence_dict):
    type_sequences = []
    
    for user in visit_sequence_dict:
        
        user_type_sequences = []
        
        for seq in visit_sequence_dict[user]:
            
            type_sequence = []
            
            for visit in seq:
                if visit != -1:
                    type_sequence.append(int(input_data['POI_Type'][visit]=='Combined'))
                else: 
                    type_sequence.append(-1)
            
            user_type_sequences.append(type_sequence)
        
        type_sequences.append(user_type_sequences)
    
    return np.array(type_sequences)


# generate POI time sequences for each valid user 
def generate_time_sequences(input_data, visit_sequence_dict):
    input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_sg_time'])
    
    time_sequences = []
    
    for user in visit_sequence_dict:
        
        user_time_sequences = []
        
        for seq in visit_sequence_dict[user]:
            
            time_sequence = []
            
            for visit in seq:
                if visit != -1:
                    time_sequence.append(input_data['Local_sg_time'][visit].hour)
                else: 
                    time_sequence.append(-1)
            
            user_time_sequences.append(time_sequence)
        
        time_sequences.append(user_time_sequences)
    
    return np.array(time_sequences)


# generate POI weekday or weekend sequences for each valid user 
def generate_weekend_sequences(input_data, visit_sequence_dict):
    input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_sg_time'])
    weekend_sequences = []
    for user in visit_sequence_dict:
        user_weekend_sequences = []
        for seq in visit_sequence_dict[user]:
            weekend_sequence = []
            for visit in seq:
                if visit != -1:
                    w=input_data['Local_sg_time'][visit].dayofweek
                    if w>4:
                        weekend_sequence.append(1)
                    else:
                        weekend_sequence.append(0)
                else: 
                    weekend_sequence.append(-1)
            user_weekend_sequences.append(weekend_sequence)
        weekend_sequences.append(user_weekend_sequences)
    return np.array(weekend_sequences)
  

# generate category sequences for each valid user 
def generate_cat_sequences(input_data, visit_sequence_dict):
    cat_sequences = []
    for user in visit_sequence_dict:
        user_cat_sequences = []
        for seq in visit_sequence_dict[user]:
            cat_sequence = []
            for visit in seq:
                if visit != -1:
                    cat_sequence.append(input_data['L2_id'][visit])
                else:
                    cat_sequence.append(-1)
            user_cat_sequences.append(cat_sequence)
        cat_sequences.append(user_cat_sequences)
    reIndexed_cat_sequences, cat_reIndex_mapping = _reIndex_3d_list(np.array(cat_sequences))
    return reIndexed_cat_sequences, cat_reIndex_mapping


# generate POI (new POI) ground truth sequences for each valid user 
def generate_new_ground_truth_sequences(input_data, ground_truth_dict, POI_reindex_mapping):

    ground_truth_sequences = []
    for user in ground_truth_dict:
        
        user_ground_truth_sequence = []
        
        for seq in ground_truth_dict[user]:
            
            ground_truth_sequence = []
            
            for visit in seq:
                if visit != -1:
                    ground_truth_sequence.append(_old_id_to_new(POI_reindex_mapping, input_data['new_id'][visit]))
                else: 
                    ground_truth_sequence.append(-1)
            
            user_ground_truth_sequence.append(ground_truth_sequence)
        
        ground_truth_sequences.append(user_ground_truth_sequence)
        
    return ground_truth_sequences



# generate location ground truth sequences for each valid user 
def generate_location_ground_truth_sequences(input_data, ground_truth_dict, POI_reindex_mapping):
    ground_truth_sequences = []
    for user in ground_truth_dict:
        
        user_ground_truth_sequence = []
        
        for seq in ground_truth_dict[user]:
            
            ground_truth_sequence = []
            
            for visit in seq:
                if visit != -1:
                    ground_truth_sequence.append(_old_id_to_new(POI_reindex_mapping, input_data['Location_id'][visit]))          
                else: 
                    ground_truth_sequence.append(-1)
            user_ground_truth_sequence.append(ground_truth_sequence)
        ground_truth_sequences.append(user_ground_truth_sequence)
    return ground_truth_sequences


def generate_specific_cate_sequences(input_data, ground_truth_dict):

    specific_cate_sequences = []
    
    for user in ground_truth_dict:
        
        user_ground_truth_sequence = []
        
        for seq in ground_truth_dict[user]:
            
            ground_truth_sequence = []
            
            for visit in seq:
                if visit != -1:
                    ground_truth_sequence.append(input_data['L2_id'][visit])
                else: 
                    ground_truth_sequence.append(-1)
            
            user_ground_truth_sequence.append(ground_truth_sequence)
        
        specific_cate_sequences.append(user_ground_truth_sequence)
        
    return specific_cate_sequences


# generate specific location sequences for the second step (POI prediction)
def generate_specific_location_sequences(input_data, ground_truth_dict):

    specific_poi_sequences = []
    
    for user in ground_truth_dict:
        
        user_ground_truth_sequence = []
        
        for seq in ground_truth_dict[user]:
            
            ground_truth_sequence = []
            
            for visit in seq:
                if visit != -1:
                    ground_truth_sequence.append(input_data['Location_id'][visit])
                else: 
                    ground_truth_sequence.append(-1)
            
            user_ground_truth_sequence.append(ground_truth_sequence)
        
        specific_poi_sequences.append(user_ground_truth_sequence)
        
    return specific_poi_sequences

# generate category (L2) distribution for each collective POI
def generate_cat_distrib(input_data, valid_visits, POI_reIndex_mapping, cat_reIndex_mapping):
    all_poi_cat_distrib = {}

    valid_data = input_data[input_data.index.isin(valid_visits)]

    collective_POI_visits = valid_data[valid_data['POI_Type'] == 'Combined']

    for collective_POI in collective_POI_visits['POI_id'].unique():

        collective_POI_visit = collective_POI_visits[collective_POI_visits['POI_id'] == collective_POI]

        collective_POI_visit['L2_id'] = collective_POI_visit['L2_id'].apply(lambda x: _old_id_to_new(cat_reIndex_mapping, x))

        poi_cat_distrib = collections.Counter(collective_POI_visit['L2_id'])

        all_poi_cat_distrib[_old_id_to_new(POI_reIndex_mapping, collective_POI)] = poi_cat_distrib

    return all_poi_cat_distrib


# form negative samples for each visit sequence
def generate_neg_sequences(POI_sequences, category_neg_sample_num, input_data, POI_reIndex_mapping,cat_reIndex_mapping):
    total_neg_sequences = []
    POI_num =POI_reIndex_mapping.shape[0] 
    for user in POI_sequences:
        user_neg_sequences = []
        for seq in user:
            no_pad_seq = [x for x in seq if x != -1] 
            neg_cand = list(set(np.arange(POI_num)) - set(seq)) # 
            neg_poi_sequence=random.sample(neg_cand, category_neg_sample_num)     
            neg_sequence = [] 
            for poi in neg_poi_sequence:        
                poi_entry = input_data[input_data['new_id'] == _new_id_to_old(POI_reIndex_mapping, poi)].iloc[0]
                poi_type = int(poi_entry['POI_Type'] == 'Combined')
                if poi_type: 
                    poi_cat = -1
                else:
                    poi_cat = _old_id_to_new(cat_reIndex_mapping, poi_entry['L2_id'])                  
                    poi_time =  poi_entry['hour']
                    poi_weekend =poi_entry['weekend']  
                neg_sequence.append([poi, poi_cat,poi_type])
            user_neg_sequences.append(neg_sequence)
        total_neg_sequences.append(user_neg_sequences)
    return total_neg_sequences

#form sample set for each valid user
def form_sample_sets(POI_sequences,time_sequences,weekend_sequences, type_sequences, cat_sequences, ground_truth_sequences, specific_poi_sequences, location_sequences,location_ground_truth_sequences,neg_sequences):
    sample_set = {} 
    user_count = 0
    sample_count = 0
    for user_pos, user in enumerate(POI_sequences):
        user_sample = []
        for seq_pos, seq in enumerate(user):
            user_sample.append((POI_sequences[user_pos][seq_pos], 
                                time_sequences[user_pos][seq_pos],
                                weekend_sequences[user_pos][seq_pos],
                                type_sequences[user_pos][seq_pos],
                                cat_sequences[user_pos][seq_pos],
                                ground_truth_sequences[user_pos][seq_pos],
                                specific_poi_sequences[user_pos][seq_pos],
                                location_sequences[user_pos][seq_pos],
                                location_ground_truth_sequences[user_pos][seq_pos],
                                neg_sequences[user_pos][seq_pos]
                            ))
            sample_count += 1
        sample_set[user_count] = user_sample
        user_count += 1

    print('Total user: %d -- Total sample: %d' %(user_count, sample_count))

    return sample_set


# check if an integer(ID) list is consecutive
def check_is_consecutive(check_list, start_index):

    
    assert check_list.max() == len(np.unique(check_list)) + start_index - 1, 'ID is not consecutive'



# remove consecutive visits (to the same POI) in a visit sequence
def _remove_consecutive_visit(visit_record, bar):
    
    clean_sequence = []
    
    for index,visit in visit_record.iterrows():
        bar.update(index)
        clean_sequence.append(index)
    return clean_sequence



# save a dictionary to a static file
def save_dict(dic, path):

    with open(path, 'wb') as f:
        pickle.dump(dic, f)



# load a dictionary from a static file
def load_dict(path):

    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic



# shuffle an input array
def shuffle(input):

    random.seed(2019)
    random.shuffle(input)
    return input
    
#-----------------------------------------------function about POI encoder date procesess--------------------------------------------

#calculate the distance between two POIs
def haversine(lng1, lat1, lng2, lat2): 

    R = 6373.0

    r_lat1 = radians(lat1)
    r_lng1 = radians(lng1)
    r_lat2 = radians(lat2)
    r_lng2 = radians(lng2)

    dlon = r_lng2 - r_lng1
    dlat = r_lat2 - r_lat1

    a = sin(dlat / 2)**2 + cos(r_lat1) * cos(r_lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c 
    return distance

#inset mall location to the orginal checkin dataset file
def inset_mall_location(df_data,df_mall):
    df_data.insert(0, 'POI_id_Latitude', value=None)
    df_data.insert(0, 'POI_id_Longitude', value=None)
    for i in range(len(df_data.axes[0])):
        POI_id = df_data['POI_id'][i]
        Type = df_data['POI_Type'][i]
        if Type == 'Combined':
            df_mall1 = df_mall[df_mall['POI_id'] == POI_id]
            df_mall1.reset_index(drop = True,inplace = True)
            df_data['POI_id_Longitude'][i] = df_mall1['Longitude'][0]
            df_data['POI_id_Latitude'][i] = df_mall1['Latitude'][0]
        elif Type == 'Independent':
            df_data['POI_id_Longitude'][i] = df_data['Longitude'][i]
            df_data['POI_id_Latitude'][i] = df_data['Latitude'][i]
    df_data = df_data.drop(['Unnamed: 0'],axis = 1)
    return df_data

#transform the dataset file 'category result/CAL/train_CAL.txt'  to a CSV file, and the columns have ['user_id_reindex','User_id','POI_id1','POI_id2','Location_id1','Location_id2','Type1','Type2','POI1_category','POI2_category','Distance','POI_Popular1','POI_Popular2','category_Pnum1','category_Pnum2','user_preference','ground_True']
# these sample is the positive sample of poi_Encoder training task
def trans_Train_data(df_train_org,locationID_mapping,userID_mapping,df_check_org):
    df_Positive_seq = DataFrame(columns = ['user_id_reindex','User_id','POI_id1','POI_id2','Location_id1','Location_id2','Type1','Type2','POI1_category','POI2_category','Distance','POI_Popular1','POI_Popular2','category_Pnum1','category_Pnum2','user_preference','ground_True'])
    for i in range(len(df_train_org.axes[0])):
        location1_id_reindex = int(df_train_org['Location_id1'][i])
        location2_id_reindex = int(df_train_org['Location_id2'][i])
        user_id_reindex = int(df_train_org['User_id'][i])
        User_id = userID_mapping[user_id_reindex]
        POI_id1 = locationID_mapping[location1_id_reindex][0]
        POI_id2 = locationID_mapping[location2_id_reindex][0]
        Location_id1 = locationID_mapping[location1_id_reindex][1]
        Location_id2 = locationID_mapping[location2_id_reindex][1]
        Type1 = locationID_mapping[location1_id_reindex][5]
        Type2 = locationID_mapping[location2_id_reindex][5]
        POI1_category = locationID_mapping[location1_id_reindex][2]
        POI2_category = locationID_mapping[location2_id_reindex][2]
        lng1 = locationID_mapping[location1_id_reindex][3]
        lng2 = locationID_mapping[location2_id_reindex][3]
        lat1 = locationID_mapping[location1_id_reindex][4]
        lat2 = locationID_mapping[location2_id_reindex][4]
        Distance = int(haversine(lng1, lat1, lng2, lat2))
        POI_Popular1 = locationID_mapping[location1_id_reindex][6]
        POI_Popular2 = locationID_mapping[location2_id_reindex][6]
        category_Pnum1 = locationID_mapping[location1_id_reindex][7]
        category_Pnum2 = locationID_mapping[location2_id_reindex][7]
        
        df_user_sequence = df_check_org[df_check_org['User_id'] == User_id]
        df_user_sequence.reset_index(drop = True,inplace = True)
        k = len(df_user_sequence.axes[0])
        p_num = len(df_user_sequence[df_user_sequence['POI_id'] == POI_id2])
        user_preference = int((p_num/k)*100)
        ground_True = 1
        df_Positive_seq = df_Positive_seq.append(DataFrame({'user_id_reindex':[user_id_reindex],'User_id':[User_id],
                                                            'POI_id1':[POI_id1],'POI_id2':[POI_id2],'Location_id1':[Location_id1],
                                                            'Location_id2':[Location_id2],'Type1':[Type1],'Type2':[Type2],
                                                            'POI1_category':[POI1_category],'POI2_category':[POI2_category],
                                                            'Distance':[Distance],'POI_Popular1':[POI_Popular1],
                                                            'POI_Popular2':[POI_Popular2],'category_Pnum1':[category_Pnum1],
                                                            'category_Pnum2':[category_Pnum2],'user_preference':[user_preference],
                                                            'ground_True':[ground_True]}))
    df_Positive_seq.reset_index(drop = True,inplace = True)
    return df_Positive_seq


# neative sampling for poi_Encoder training task
# one positive sample matches one negative sample
def prepare_Neg_Seq(df_Positive_seq,df_sorted,df_POI,Category):
    
    df_Negitive_seq = DataFrame(columns = ['User_id','POI_id1','POI_id2','Type1','Type2',
                                'POI1_category','POI2_category','Distance','POI_Popular1','POI_Popular2',
                                'category_Pnum1','category_Pnum2','user_preference','ground_True'])
    User_id_list = []
    finish = 0
    user_num = len(df_Positive_seq.axes[0])
    bar = progressbar.ProgressBar(maxval=len(df_Positive_seq.axes[0]), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for index, row in df_Positive_seq.iterrows():
        bar.update(index)
        User_id = row['User_id']
        if User_id not in User_id_list:
            User_id_list.append(User_id)
            user_pos_ratings = dict()
            user_neg_ratings = dict()
            POI_id1_list = []
            df_user_sequence = df_sorted[df_sorted['User_id'] == User_id]
            df_user_sequence.reset_index(drop = True,inplace = True)
            df_user_list = df_Positive_seq[df_Positive_seq['User_id'] == User_id]
            df_user_list = df_user_list.drop_duplicates()
            df_user_list.reset_index(drop = True, inplace = True)
            for index1, row1 in df_user_list.iterrows():
                POI_id1 = row1['POI_id1']
                if POI_id1 not in POI_id1_list:
                    user_pos_ratings[POI_id1] = []
                    user_neg_ratings[POI_id1] = []
                    POI_id1_list.append(POI_id1) 
                    for index2,row2 in df_POI.iterrows():
                        POI_id = row2['POI_id']
                        num = len(df_user_list[(df_user_list['POI_id1'] == POI_id1)&(df_user_list['POI_id2'] == POI_id)])
                        if num>=1:
                            user_pos_ratings[POI_id1].append(POI_id)
                        else:
                            user_neg_ratings[POI_id1].append(POI_id)
                    num1 = len(df_Positive_seq[(df_Positive_seq['User_id'] == User_id)&(df_Positive_seq['POI_id1'] == POI_id1)])
                    neg =  np.random.choice(user_neg_ratings[POI_id1],num1,replace = False)
                    for Location_neg_id in neg:
                        df_Location_id1 = df_POI[df_POI['POI_id'] == POI_id1]
                        df_Location_neg_id = df_POI[df_POI['POI_id'] == Location_neg_id]
                        df_Location_id1.reset_index(drop = True, inplace = True)
                        df_Location_neg_id.reset_index(drop = True, inplace = True)
                        Category1 = df_Location_id1[Category][0]
                        Category2 = df_Location_neg_id[Category][0]
                        POI_id2 = Location_neg_id
                        Type1 = df_Location_id1['POI_Type'][0]
                        Type2 = df_Location_neg_id['POI_Type'][0]
                        lng1 = df_Location_id1['POI_id_Longitude'][0]
                        lat1 = df_Location_id1['POI_id_Latitude'][0]
                        lng2 = df_Location_neg_id['POI_id_Longitude'][0]
                        lat2 = df_Location_neg_id['POI_id_Latitude'][0]
                        Distance = int(haversine(lng1, lat1, lng2, lat2))
                        POI_Popular1 = df_Location_id1['POI_id_Popular'][0]
                        POI_Popular2 = int(df_Location_neg_id['POI_id_Popular'][0])
                        category_Pnum1 = df_Location_id1['category_Pnum'][0]
                        category_Pnum2 = df_Location_neg_id['category_Pnum'][0]
                        p_num = len(df_user_sequence[df_user_sequence['POI_id'] == POI_id2])
                        user_preference = int((p_num/len(df_user_sequence.axes[0]))*100) #percentage
                        ground_True = 0
                        df_Negitive_seq = df_Negitive_seq.append(DataFrame({
                                'User_id':[User_id],'POI_id1':[POI_id1],
                                'POI_id2':[POI_id2],'Type1':[Type1],'Type2':[Type2],'POI1_category':[Category1],
                                'POI2_category':[Category2],'Distance':[Distance],'POI_Popular1':[POI_Popular1],
                                'POI_Popular2':[POI_Popular2],'category_Pnum1':[category_Pnum1],'category_Pnum2':[category_Pnum2],
                                'user_preference':[user_preference],'ground_True':[ground_True]}))
        #finish+=1
    bar.finish()
        
    df_Negitive_seq.reset_index(drop = True,inplace = True)
    return df_Negitive_seq     

def value2index(df_all_seq,column):
    value_list = list(df_all_seq[column].unique())
    value_list.sort()
    Value2Index = dict()
    index = 0
    for i in value_list:
        Value2Index[i] = index
        index+=1
    return Value2Index 


