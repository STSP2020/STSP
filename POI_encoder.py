import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Util
import pickle
import matplotlib.pyplot as plt
from numpy import argmax #one-hot encoder  
from keras.utils import to_categorical #one-hot decoder  


def POI_encoder(args):
#---------------------------------load file--------------------------------   
    df_POI = pd.read_csv('data/' + args.dataset + '/1_lstm_POI.csv')
    df_checkin = pd.read_csv('data/' + args.dataset + '_checkin.csv')
    test_set = pd.read_csv('data/' + args.dataset + '/1_lstm_Test.csv')
    train_set = pd.read_csv('data/' + args.dataset + '/1_lstm_Train.csv')
    
    #prepare for one-hot, reindex

    #POI_pop
    POI_pop2Index = Util.value2index(df_POI,'POI_id_Popular')

    #category_num
    Category_num2Index = Util.value2index(df_POI,'category_Pnum')

    #userId——>embedding
    dict_file = 'data/'+ args.dataset + '/1_lstm_userID_old2new.pickle'
    with open(dict_file, "rb") as fp:   #read dictionary
        userID_old2new = pickle.load(fp)


    user_matrix = np.load('category result/'+ args.dataset + '/user_rep_'+ args.dataset + '/' + str(0) + '_full.npy', allow_pickle=True) 
    m = (user_matrix).shape[2] #embeding size
    user_matrix = (user_matrix).reshape(1,m)
    for i in range(1,len(userID_old2new.keys())):
        user_rep_vec1 = np.load('category result/'+ args.dataset + '/user_rep_'+ args.dataset + '/' + str(i) + '_full.npy', allow_pickle=True) 
        user_rep_vec1 = (user_rep_vec1).reshape(1,m)
        user_matrix = np.concatenate((user_matrix,user_rep_vec1),axis=0)

    
#-----------------------------------parameters----------------------------

    listmax = list(range(0,150+1))
    listprefer = list(range(0,100+1))

    #one-hot
    POI_popularity_matrix = to_categorical(list(POI_pop2Index.values()))
    POI_number_matrix = to_categorical(list(Category_num2Index.values()))
    user_Prefer_matrix = to_categorical(listprefer) # the max percentage is 100%
    distance_matrix = to_categorical(listmax) #the max distance is smaller than 150km


    # set parameters
    lr = args.POI_lr #learning rate
    iter_num = args.POI_iter_num #no. of iteration
    reg_beta = args.POI_reg_beta # overfit control
    break_threshold = args.POI_break_threshold# iteration control
    seed = args.POI_seed
    candidate_num = args.POI_candidate_num # Top-K
    if candidate_num == 1:
        Category_name = 'Category1'
    elif candidate_num == 5:
        Category_name = 'Category5'
    elif candidate_num == 10:
        Category_name = 'Category10'

    distance_size = distance_matrix.shape[1]
    pop_size = POI_popularity_matrix.shape[1]
    number_size = POI_number_matrix.shape[1]
    prefer_size = user_Prefer_matrix.shape[1]



    # define placeholders (inputs)
    alpha = tf.placeholder(tf.float32,shape=None)
    truth = tf.placeholder(tf.float32,shape=None)
    user_embedding = tf.placeholder(tf.float32, shape=[1, m])
    distance = tf.placeholder(tf.float32, shape=[1, distance_size])
    POI_popularity = tf.placeholder(tf.float32, shape=[1, pop_size])
    POI_number = tf.placeholder(tf.float32, shape=[1, number_size])
    user_Prefer = tf.placeholder(tf.float32, shape=[1, prefer_size])


    # define variables (weights)
    #init_weight = tf.truncated_normal([n , m], stddev = 1.0/np.sqrt(n))
    d_weight = tf.Variable(tf.truncated_normal([distance_size , m], stddev = 1.0/np.sqrt(distance_size),seed = seed))
    IP_weight = tf.Variable(tf.truncated_normal([pop_size , m], stddev = 1.0/np.sqrt(pop_size),seed = seed))
    MP_weight = tf.Variable(tf.truncated_normal([pop_size , m], stddev = 1.0/np.sqrt(pop_size),seed = seed))
    num_weight = tf.Variable(tf.truncated_normal([number_size , m], stddev = 1.0/np.sqrt(number_size),seed = seed))
    perfer_weight = tf.Variable(tf.truncated_normal([prefer_size , m], stddev = 1.0/np.sqrt(prefer_size),seed = seed))



#-------------------------------------------------construct model-------------------------------------------------------------
    r = ((1 - alpha) * (tf.matmul(distance, d_weight) + tf.matmul(POI_popularity, IP_weight))
         + alpha * (tf.matmul(distance, d_weight) + tf.matmul(POI_popularity, MP_weight) + tf.matmul(POI_number, num_weight))) + (tf.matmul(user_Prefer, perfer_weight))

    r = tf.transpose(r)
    score = tf.matmul(user_embedding , r)# (1,1)

    regularization1 = tf.nn.l2_loss(d_weight) + tf.nn.l2_loss(IP_weight) + tf.nn.l2_loss(perfer_weight)
    regularization2 = tf.nn.l2_loss(d_weight) + tf.nn.l2_loss(MP_weight) + tf.nn.l2_loss(num_weight) + tf.nn.l2_loss(perfer_weight)

    #loss = tf.math.square(truth - score) #  non regularization
    loss = (1 - alpha) * (tf.math.square(truth - score) + reg_beta * regularization1) + alpha *(tf.math.square(truth - score) + reg_beta * regularization2)


    # define training algorithm
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

#--------------------------------------------------------train----------------------------------------------------------------------
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        entropy_loss = []
        iter_result = 1
        i = 0
        #'Distance', 'Location_id1', 'Location_id2',
        #'POI1_category', 'POI2_category', 'POI_Popular1', 'POI_Popular2',
        #'POI_id1', 'POI_id2', 'Type1', 'Type2', 'User_id', 'category_Pnum1',
        #'category_Pnum2', 'ground_True', 'user_id_reindex', 'user_preference'
        while iter_result > break_threshold:

            if i < iter_num:
                sample_counter = 0
                iter_total_loss = 0.0
                for index,row in train_set.iterrows():
                    if (row['Type2'] == 'Combined'):
                         _, _loss = sess.run([train,loss], {alpha: 1,
                                         truth: row['ground_True'],
                                         distance: (distance_matrix[row['Distance']]).reshape(1,distance_size),
                                         POI_popularity: (POI_popularity_matrix[POI_pop2Index[row['POI_Popular2']]]).reshape(1,pop_size),
                                         POI_number: (POI_number_matrix[Category_num2Index[row['category_Pnum2']]]).reshape(1,number_size),
                                         user_Prefer: (user_Prefer_matrix[row['user_preference']]).reshape(1,prefer_size),
                                         user_embedding: (user_matrix[userID_old2new[row['User_id']]]).reshape(1,m)                                                   
                                        })
                    elif (row['Type2'] == 'Independent'):
                        _, _loss = sess.run([train,loss], {alpha: 0,
                                         truth: row['ground_True'],
                                         distance: (distance_matrix[row['Distance']]).reshape(1,distance_size),
                                         POI_popularity: (POI_popularity_matrix[POI_pop2Index[row['POI_Popular2']]]).reshape(1,pop_size),
                                         POI_number: (POI_number_matrix[Category_num2Index[row['category_Pnum2']]]).reshape(1,number_size),
                                         user_Prefer: (user_Prefer_matrix[row['user_preference']]).reshape(1,prefer_size),
                                         user_embedding: (user_matrix[userID_old2new[row['User_id']]]).reshape(1,m) 
                                        })
                    iter_total_loss += _loss
                    sample_counter += 1

                avg_loss = iter_total_loss / sample_counter
                entropy_loss.append(avg_loss)
                print('iteration: %d, entropy loss: %f' %(i, avg_loss))

                if i >= 1:
                    iter_result = entropy_loss[-2] - entropy_loss[-1]     
                    iter_result = abs(iter_result)
                    
            print(iter_result)
            i+=1

        saver.save(sess, 'model/'+ args.dataset + '/path1.ckpt')

#-------------------------------------------test-----------------------------------------------------------

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, 'model/'+ args.dataset + '/path1.ckpt')

        count10 = 0
        sample = 0
        map10_list = []

        #['User_id', 'POI_id1', 'POI_id2', 'Location_id1','Location_id2', 'Pred_Type', 'Category1', 'Category5', 'Category10']
        for index,row in test_set.iterrows():
            User_id = row['User_id']
            POI_id1 = row['POI_id1']
            POI_id2 = row['POI_id2']
            Location_id2 = row['Location_id2']
            Type = row['Pred_Type']
            Category = row[Category_name]
            df_POI_id1 = df_POI[df_POI['POI_id'] == POI_id1]
            df_POI_id1.reset_index(drop = True,inplace = True)
            lng1 = df_POI_id1['POI_id_Longitude'][0]
            lat1 = df_POI_id1['POI_id_Latitude'][0]
            Type_require = df_POI['POI_Type'].map(lambda T : T == Type)
            Catgory_require = df_POI['Category'].map(lambda c2 : c2 in Category)
            df_candidate = df_POI[Type_require & Catgory_require] #10个candidate category的情况
            df_candidate = df_candidate.drop_duplicates(subset='POI_id', keep="first")
            df_result = DataFrame(columns = ['candidate_id','score'])
            for index1,row1 in df_candidate.iterrows():
                candidate_id = row1['POI_id']
                lat2 = row1['POI_id_Latitude']
                lng2 = row1['POI_id_Longitude']
                Dis = Util.haversine(lng1, lat1, lng2, lat2)
                Distance_candidate = int(Dis)
                num = len(df_checkin[(df_checkin['User_id'] == User_id)&(df_checkin['POI_id'] == candidate_id)])
                total = len(df_checkin[df_checkin['User_id'] == User_id])
                preference_candidate = int((num/total)*100)          
                if (Type == 'Combined'):
                    pred = sess.run(score, {alpha: 1,
                                         distance: (distance_matrix[Distance_candidate]).reshape(1,distance_size),
                                         POI_popularity: (POI_popularity_matrix[POI_pop2Index[int(row1['POI_id_Popular'])]]).reshape(1,pop_size),
                                         POI_number: (POI_number_matrix[Category_num2Index[row1['category_Pnum']]]).reshape(1,number_size),
                                         user_Prefer: (user_Prefer_matrix[preference_candidate]).reshape(1,prefer_size),
                                         user_embedding: (user_matrix[userID_old2new[User_id]]).reshape(1,m) 
                                        })
                elif (Type == 'Independent'):
                    pred = sess.run(score, {alpha: 0,
                                         distance: (distance_matrix[Distance_candidate]).reshape(1,distance_size),
                                         POI_popularity: (POI_popularity_matrix[POI_pop2Index[int(row1['POI_id_Popular'])]]).reshape(1,pop_size),
                                         POI_number: (POI_number_matrix[Category_num2Index[row1['category_Pnum']]]).reshape(1,number_size),
                                         user_Prefer: (user_Prefer_matrix[preference_candidate]).reshape(1,prefer_size),
                                         user_embedding: (user_matrix[userID_old2new[User_id]]).reshape(1,m)
                                        })

                df_result = df_result.append(DataFrame({'candidate_id':[candidate_id],'score':[pred]}))
            df_result = df_result.sort_values(by = 'score',ascending = False)
#--------------------------------------------------prediction---------------------------------------------------------------------
            if (Type == 'Combined'):
                Top10 = list(df_result['candidate_id'][:10])
                POI_id_require = df_POI['POI_id'].map(lambda x : x in Top10)
                Catgory_require = df_POI['Category'].map(lambda c1 : c1 in Category)
                df_candidate_location = df_POI[POI_id_require & Catgory_require]
                df_candidate_location = df_candidate_location.sort_values(by = 'stars',ascending = False)

                location_Top10 = list(df_candidate_location['Location_id'][:10])
                if Location_id2 in location_Top10:
                    map10 =  (1/(location_Top10.index(Location_id2)+1))
                else:
                    map10 = 0

                map10_list.append(map10)

                if Location_id2 in location_Top10:
                    count10 += 1

                sample += 1
               

            elif (Type == 'Independent'):

                Top10 = list(df_result['candidate_id'][:10])

                if POI_id2 in Top10:
                    map10 =  (1/(Top10.index(POI_id2)+1))
                else:
                    map10 = 0

                map10_list.append(map10)


                if POI_id2 in Top10:
                    count10 += 1

                sample += 1
                

        MAP10 = np.mean(map10_list)
        recall10 = count10/sample

        print('lr: '+str(lr)+', reg_bete: '+str(reg_beta)+', break_threshold: '+str(break_threshold)+', '+str(candidate_num)+' category, ten POI_id result:')
        print('MAP@10',MAP10)
        print('recall@10',recall10)