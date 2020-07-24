import tensorflow as tf
import numpy as np
import random
from data_preprocess import load_dict, generate_cat_vec_seq


def category_encoder(args):
    city = args.dataset
#---------------------------------load file--------------------------------
    dir = './'+ args.dataset + '/'
    samples = load_dict(dir + 'sample_sets.pkl') 
    POI_cat_distrib = load_dict(dir + 'poi_cat_distrib.pkl')
    POI_reIndex_mapping = np.load(dir + 'POI_reIndex_mapping.npy', allow_pickle=True)
    cat_reIndex_mapping = np.load(dir + 'cat_reIndex_mapping.npy', allow_pickle=True)
    cat_transition=np.load(dir+'category_transition_marix.npy')
    user_reIndex_mapping = np.load(dir + 'user_reIndex_mapping.npy', allow_pickle=True)
    max_seq_len = np.load(dir + 'max_seq_len.npy', allow_pickle=True)
    neg_num = np.load(dir + 'neg_sample_num.npy', allow_pickle=True)
#-----------------------------------parameters----------------------------
    
    train_portion = args.train_portion
    np_rand_seed = args.category_seed
    np.random.seed(np_rand_seed)
    random.seed(np_rand_seed)
    RNN_stack_layers = args.RNN_stack_layers
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    lr =args.category_lr
    iter_num = args.category_iter_num
    break_threshold = args.category_break_threshold
    w_c = args.category_w_c
    w_t= args.category_w_t
    keep_prob = args.category_keep_prob
    reg_beta = args.category_reg_beta
    poi_size = len(POI_reIndex_mapping)
    time_size = args.category_time_size
    weekend_size= args.category_weekend_size
    user_size = len(user_reIndex_mapping)
    cat_size = len(cat_reIndex_mapping)
    
    # # Prepare Train Test Set
    # train test split after shuffling
    all_samples = []
    for key in samples.keys():
        user_samples = samples[key]
        random.shuffle(user_samples)   
        all_samples.append(user_samples)
    # split train test samples
    all_training_samples, all_testing_samples = [], []
    for user_samples in all_samples:
        N = len(user_samples)
        train_test_boundary = int(train_portion*N)
        all_training_samples.append(user_samples[:train_test_boundary])
        all_testing_samples.append(user_samples[train_test_boundary:])
        
        
    # Construct Model
    x_poi = tf.placeholder(tf.int32, shape = [None, None]) 
    x_time = tf.placeholder(tf.int32, shape = [None, None]) 
    x_weekend = tf.placeholder(tf.int32, shape = [None, None]) 
    x_type = tf.placeholder(tf.int32, shape = [None, None]) 
    x_cat = tf.placeholder(tf.float32, shape = [None, None, cat_size]) 
    y_poi = tf.placeholder(tf.int32, shape = [None, 1]) 
    y_time = tf.placeholder(tf.int32, shape = [None, 1]) 
    y_weekend = tf.placeholder(tf.int32, shape = [None, 1]) 
    y_type = tf.placeholder(tf.int32, shape = [None, 1]) 
    y_cat = tf.placeholder(tf.float32, shape = [None, 1, cat_size]) 
    true_y_cat = tf.placeholder(tf.int32, shape = [None, 1]) 
    saved_user_rep = tf.placeholder(tf.float32, shape = [None, 1, hidden_size]) 
    ground_truth_set = tf.placeholder(tf.int32, shape = [None, None]) 
    prediction_set = tf.placeholder(tf.int32, shape = [None, None]) 


    neg_poi = tf.placeholder(tf.int32, [None, neg_num])
    neg_time = tf.placeholder(tf.int32, [None, neg_num]) 
    neg_weekend = tf.placeholder(tf.int32, [None, neg_num]) 
    neg_type = tf.placeholder(tf.int32, [None, neg_num])
    neg_cat = tf.placeholder(tf.float32, [None, neg_num, cat_size]) 


    # embeddings
    user_emb = tf.Variable(tf.random_uniform([user_size, hidden_size], -1.0, 1.0))
    cat_emb1 = tf.Variable(tf.random_uniform([cat_size, hidden_size], -1.0, 1.0))
    cat_emb=tf.matmul(tf.cast(cat_transition,tf.float32),cat_emb1)+cat_emb1
    type_emb = tf.Variable(tf.random_uniform([2, hidden_size], -1.0, 1.0))
    time_emb = tf.Variable(tf.random_uniform([time_size, hidden_size], -1.0, 1.0))
    weekend_emb = tf.Variable(tf.random_uniform([weekend_size, hidden_size], -1.0, 1.0))


    # weights
    init_weight = tf.truncated_normal([hidden_size, hidden_size], stddev = 1.0/np.sqrt(hidden_size))
    W_time = tf.Variable(init_weight)
    W_weekend = tf.Variable(init_weight)
    W_cat = tf.Variable(init_weight)
    W_type = tf.Variable(init_weight)
    W_h_c = tf.Variable(init_weight)
    W_h_t= tf.Variable(tf.truncated_normal([1, hidden_size], stddev = 1.0/np.sqrt(hidden_size)))

    # RNN Module
    #  x inputs: category, time,weekend, type
    input_x_cat = tf.matmul(x_cat, tf.expand_dims(cat_emb, 0)) 
    input_x_time = tf.nn.embedding_lookup(time_emb, x_time) 
    input_x_weekend = tf.nn.embedding_lookup(weekend_emb, x_weekend)
    input_x_type = tf.nn.embedding_lookup(type_emb, x_type) 
    inputs_x_l=tf.matmul(input_x_time,tf.expand_dims(W_time,0)) +tf.matmul(input_x_weekend, tf.expand_dims(W_weekend,0))+ tf.matmul(input_x_cat, tf.expand_dims(W_cat,0)) + tf.matmul(input_x_type, tf.expand_dims(W_type,0)) # [batch, seq_len, dim]

    # y inputs: category, time, type
    input_y_cat = tf.matmul(y_cat, tf.expand_dims(cat_emb, 0)) 
    input_y_time = tf.nn.embedding_lookup(time_emb, y_time)
    input_y_weekend = tf.nn.embedding_lookup(weekend_emb, y_weekend)
    input_y_type = tf.nn.embedding_lookup(type_emb, y_type) 
    inputs_y_l=tf.matmul(input_y_time, tf.expand_dims(W_time,0)) + tf.matmul(input_y_weekend, tf.expand_dims(W_weekend,0)) +tf.matmul(input_y_cat, tf.expand_dims(W_cat,0)) + tf.matmul(input_y_type, tf.expand_dims(W_type,0)) # [batch, seq_len(1), dim]

    #---------------------------------------------
    # negative inputs 
    input_neg_cat = tf.matmul(neg_cat, tf.expand_dims(cat_emb, 0)) 
    input_neg_time = tf.nn.embedding_lookup(time_emb, y_time) 
    input_neg_weekend = tf.nn.embedding_lookup(weekend_emb, y_weekend) 
    input_neg_type = tf.nn.embedding_lookup(type_emb, neg_type) 
    inputs_neg_l = tf.matmul(input_neg_time, tf.expand_dims(W_time,0)) + tf.matmul(input_neg_weekend, tf.expand_dims(W_weekend,0))+ tf.matmul(input_neg_cat, tf.expand_dims(W_cat,0)) + tf.matmul(input_neg_type, tf.expand_dims(W_type,0))  
    
    # RNN model
    with tf.variable_scope("rnn"):

        cell_l = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
        cell_l = tf.nn.rnn_cell.DropoutWrapper(cell_l, output_keep_prob=keep_prob)
        cell_l = tf.nn.rnn_cell.MultiRNNCell([cell_l] * RNN_stack_layers)
        initial_state_l = cell_l.zero_state(batch_size, tf.float32)
        outputs_l, states_l = tf.nn.dynamic_rnn(cell_l, inputs = inputs_x_l, initial_state = initial_state_l)

    regularization = tf.nn.l2_loss(W_time) + tf.nn.l2_loss(W_weekend) +tf.nn.l2_loss(W_cat) + tf.nn.l2_loss(cat_emb) + tf.nn.l2_loss(time_emb) +tf.nn.l2_loss(weekend_emb)+ tf.nn.l2_loss(W_h_c) + tf.nn.l2_loss(W_h_t) + tf.nn.l2_loss(W_type) + tf.nn.l2_loss(type_emb)   


    # loss 
    final_output_l = tf.expand_dims(tf.transpose(outputs_l, [1, 0, 2])[-1], 1)  

    # category loss
    output_h_c = tf.matmul(final_output_l, tf.expand_dims(W_h_c,0))

    r_cat = tf.matmul(output_h_c, tf.transpose(inputs_y_l, [0, 2, 1])) 
    r_cat_neg = tf.matmul(output_h_c, tf.transpose(inputs_neg_l, [0, 2, 1])) 

    loss_cat = tf.reduce_sum(1 + tf.log(tf.exp(-(tf.tile(r_cat, [0, 0, neg_num]) - r_cat_neg)))) 

    # type loss 
    output_h_t = tf.matmul(final_output_l, tf.expand_dims(tf.transpose(W_h_t),0)) 

    pred_t = tf.reduce_sum(tf.sigmoid(output_h_t)) 

    loss_type = - (tf.cast(tf.reduce_sum(y_type), tf.float32) * tf.log(pred_t) + tf.cast((1-tf.reduce_sum(y_type)), tf.float32) * tf.log(1 - pred_t))


    # final loss
    total_loss = w_c * loss_cat + w_t * loss_type + reg_beta * regularization

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    gradients, variables = zip(*optimizer.compute_gradients(total_loss))

    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

    train = optimizer.apply_gradients(zip(gradients, variables))



    #Testing 
    final_output_l = tf.expand_dims(tf.transpose(outputs_l, [1, 0, 2])[-1], 1) 

    final_rep_cat = tf.matmul(final_output_l + saved_user_rep, tf.expand_dims(W_h_c,0)) 

    all_cats = tf.matmul(tf.expand_dims(W_cat,0), tf.transpose(tf.expand_dims(cat_emb,0),[0,2,1])) 

    logits_cat = tf.matmul(final_rep_cat, all_cats) 

    final_rep_type=tf.matmul(final_output_l+saved_user_rep, tf.transpose(tf.expand_dims(W_h_t,0),[0,2,1])) 

    type_output = tf.sigmoid(final_rep_type)
    

    # evaluation : k = 1 
    prediction_1_cat = tf.nn.top_k(logits_cat,1)[1]
    expand_targets_1_cat = tf.tile(true_y_cat, [1, 1])
    isequal_1_cat = tf.equal(expand_targets_1_cat, prediction_1_cat)
    correct_prediction_1_cat = tf.reduce_sum(tf.cast(isequal_1_cat, tf.float32))
    precison_1_cat = correct_prediction_1_cat / tf.cast(batch_size*1,tf.float32)
    recall_1_cat = correct_prediction_1_cat / tf.cast(batch_size,tf.float32)
    f1_1_cat = 2 * precison_1_cat * recall_1_cat / (precison_1_cat + recall_1_cat + 1e-10)
        
    # evaluation: k = 5
    prediction_5_cat = tf.nn.top_k(logits_cat,5)[1]
    expand_targets_5_cat = tf.tile(true_y_cat, [1, 5])
    isequal_5_cat = tf.equal(expand_targets_5_cat, prediction_5_cat)
    correct_prediction_5_cat = tf.reduce_sum(tf.cast(isequal_5_cat, tf.float32))
    precison_5_cat = correct_prediction_5_cat / tf.cast(batch_size*5,tf.float32)
    recall_5_cat = correct_prediction_5_cat / tf.cast(batch_size,tf.float32)
    f1_5_cat = 2 * precison_5_cat * recall_5_cat / (precison_5_cat + recall_5_cat + 1e-10)
    
    # evaluation : k = 10
    prediction_10_cat = tf.nn.top_k(logits_cat,10)[1]
    expand_targets_10_cat = tf.tile(true_y_cat, [1, 10])
    isequal_10_cat = tf.equal(expand_targets_10_cat, prediction_10_cat)
    correct_prediction_10_cat = tf.reduce_sum(tf.cast(isequal_10_cat, tf.float32))
    precison_10_cat = correct_prediction_10_cat / tf.cast(batch_size*10,tf.float32)
    recall_10_cat = correct_prediction_10_cat / tf.cast(batch_size,tf.float32)
    f1_10_cat = 2 * precison_10_cat * recall_10_cat / (precison_10_cat + recall_10_cat + 1e-10)
    prediction_type = tf.to_int32(type_output > 0.5)


    isequal_type = tf.equal(tf.expand_dims(y_type,1), prediction_type) 
    accuracy_type = tf.reduce_sum(tf.cast(isequal_type, tf.float32)) 
    
    
#---------------------------------------------------training-----------------------------------------------------------------
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        entropy_loss = []
        prev_loss = 10000.0
        try:    
            for i in range(iter_num):

                user_counter = 0

                sample_counter = 0

                iter_total_loss = 0.0

                for user_training_samples in all_training_samples:

                    for sample in user_training_samples:
    #                   save training samples for the second step  (individual POI prediction)
                        training_data=str(user_counter,sample[7])
                        with open('train_CAL.txt','a') 
                            file_handle.write(training_data)    
                            file_handle.write('\n')        


                        feed_dict = data_feeder(sample)

                        _, _loss = sess.run([train, total_loss], 
                                            {x_time: feed_dict['x_time'], 
                                             x_weekend: feed_dict['x_weekend'],
                                             x_cat: feed_dict['x_cat'], 
                                             x_type: feed_dict['x_type'], 
                                             x_poi: feed_dict['x_poi'],                                       
                                             y_time: feed_dict['y_time'], 
                                             y_weekend: feed_dict['y_weekend'], 
                                             y_cat: feed_dict['y_cat'], 
                                             y_type: feed_dict['y_type'],
                                             y_poi: feed_dict['y_poi'],
                                             neg_cat: feed_dict['neg_cat'],
                                             neg_type: feed_dict['neg_type'] })
                        iter_total_loss += _loss

                        sample_counter += 1                  

                        # save user representation

                        user_rep = sess.run(final_output_l, {x_time: feed_dict['x_time'], 
                                                    x_weekend: feed_dict['x_weekend'], 
                                                    x_cat: feed_dict['x_cat'], 
                                                    x_type: feed_dict['x_type'],
                                                    y_time: feed_dict['y_time'], 
                                                    y_weekend: feed_dict['y_weekend'], 
                                                    y_cat: feed_dict['y_cat'], 
                                                    y_type: feed_dict['y_type'],
                                                    neg_cat: feed_dict['neg_cat'],
                                                    neg_type: feed_dict['neg_type']})
                        np.save('./user_rep_'+ args.dataset +'/' + str(user_counter) + '_full.npy', user_rep)

                    user_counter += 1

                avg_loss = iter_total_loss / sample_counter

                entropy_loss.append(avg_loss)

                if i % 1 == 0:
                    print('iteration: %d, entropy loss: %f' %(i, avg_loss))    
                if prev_loss - avg_loss > break_threshold: 
                    prev_loss = avg_loss  
                else: 
                    raise StopIteration
        except StopIteration: 
            print('End training at epoch: %d' %(i))
            saver.save(sess, './saved_model_'+ args.dataset +'/main_model_full.ckpt')
            pass

        saver.save(sess, './saved_model_'+ args.dataset +'/main_model_full.ckpt')
        
        saver = tf.train.Saver()

        
#-----------------------------------------------test-----------------------------------------------------------        
        
    with tf.Session() as sess:

        saver.restore(sess, './saved_model_'+ args.dataset +'/main_model_full.ckpt')


        total_precision_1_cat = 0

        total_map_1_cat = 0

        total_recall_1_cat = 0

        total_f1_1_cat = 0

        total_f1_1_cat_map = 0



        total_precision_5_cat = 0

        total_map_5_cat = 0

        total_recall_5_cat = 0

        total_f1_5_cat = 0

        total_f1_5_cat_map = 0



        total_precision_10_cat = 0

        total_map_10_cat = 0

        total_recall_10_cat = 0

        total_f1_10_cat = 0

        total_f1_10_cat_map = 0


        total_accuracy_type = 0


        sample_number = 0

        for user_counter in range(len(user_reIndex_mapping)):

            user_rep_vec = np.load('./user_rep_'+ args.dataset +'/' + str(user_counter) + '_full.npy', allow_pickle=True) 

            for sample in all_testing_samples[user_counter]:

                feed_dict = data_feeder(sample)

                c_precison_1_cat, c_recall_1_cat, c_f1_1_cat, c_precison_5_cat, c_recall_5_cat, c_f1_5_cat,c_precison_10_cat, c_recall_10_cat,c_f1_10_cat,c_accuracy_type, cat_true, cat_1_pred, cat_5_pred ,cat_10_pred,type_pred = sess.run([precison_1_cat, recall_1_cat, f1_1_cat, precison_5_cat, recall_5_cat, f1_5_cat ,precison_10_cat, recall_10_cat, f1_10_cat ,accuracy_type, true_y_cat, prediction_1_cat, prediction_5_cat,prediction_10_cat,prediction_type],

               {x_time: feed_dict['x_time'], 
                 x_weekend: feed_dict['x_weekend'],
                 x_cat: feed_dict['x_cat'],
                 x_type: feed_dict['x_type'],
                 true_y_cat: feed_dict['true_y_cat'],
                 y_type: feed_dict['y_type'],
                 saved_user_rep: user_rep_vec,
                 ground_truth_set: feed_dict['ground_truth_set']})

                total_precision_1_cat += c_precison_1_cat

                total_recall_1_cat += c_recall_1_cat

                total_f1_1_cat += c_f1_1_cat



                total_precision_5_cat += c_precison_5_cat

                total_recall_5_cat += c_recall_5_cat

                total_f1_5_cat += c_f1_5_cat



                total_precision_10_cat += c_precison_10_cat

                total_recall_10_cat += c_recall_10_cat

                total_f1_10_cat += c_f1_10_cat



                map_1_cat = MAP_score(cat_1_pred, cat_true)

                map_5_cat = MAP_score(cat_5_pred, cat_true)

                map_10_cat = MAP_score(cat_10_pred, cat_true)



                total_map_1_cat += map_1_cat

                total_map_5_cat += map_5_cat

                total_map_10_cat += map_10_cat



                c_f1_1_cat_map = 2 * map_1_cat * c_recall_1_cat / (map_1_cat + c_recall_1_cat + 1e-10)

                c_f1_5_cat_map = 2 * map_5_cat * c_recall_5_cat / (map_5_cat + c_recall_5_cat + 1e-10)

                c_f1_10_cat_map = 2 * map_10_cat * c_recall_10_cat / (map_10_cat + c_recall_10_cat + 1e-10)



                total_f1_1_cat_map += c_f1_1_cat_map

                total_f1_5_cat_map += c_f1_5_cat_map

                total_f1_10_cat_map += c_f1_10_cat_map


                total_accuracy_type += c_accuracy_type

               #save category and type prediction results for the second step (individual POI prediction)
                prediction_data=str(user_counter,sample[0],sample[5],sample[6],sample[7],sample[8],cat_true.flatten().flatten(),c_accuracy_type.flatten(),type_pred.flatten(),cat_1_pred.flatten(),cat_5_pred.flatten(),cat_10_pred.flatten())

                with open('result_CAL.txt','a') as file_handle:   
                    file_handle.write(prediction_data)    
                    file_handle.write('\n')         

                feed_dict = data_feeder(sample)

                sample_number += 1
        sample_number = len(user_reIndex_mapping)
        total_precision_1_cat /= sample_number

        total_recall_1_cat /= sample_number

        total_f1_1_cat /= sample_number

        total_f1_1_cat_map /= sample_number



        total_precision_5_cat /= sample_number

        total_recall_5_cat /= sample_number

        total_f1_5_cat /= sample_number

        total_f1_5_cat_map /= sample_number



        total_precision_10_cat /= sample_number

        total_recall_10_cat /= sample_number

        total_f1_10_cat /= sample_number

        total_f1_10_cat_map /= sample_number

        total_accuracy_type /= sample_number

        total_map_1_cat /= sample_number

        total_map_5_cat /= sample_number

        total_map_10_cat /= sample_number

    print('total_precision_1_cat', total_precision_1_cat)
    print('total_recall_1_cat',total_recall_1_cat)
    print('total_map_1_cat',total_map_1_cat)

    print('***************************************')
    print('total_precision_5_cat', total_precision_5_cat)
    print('total_recall_5_cat',total_recall_5_cat)
    print('total_map_5_cat',total_map_5_cat)


    print('***************************************')
    print('total_precision_10_cat', total_precision_10_cat)
    print('total_recall_10_cat',total_recall_10_cat)
    print('total_map_10_cat',total_map_10_cat)
        
#-------------------------------------------------------functions--------------------------------------------------------------
        
def data_feeder(sample):
    feed_dict = {}
    poi_x=sample[0][:-1]
    poi_y=sample[0][-1]
    time_x = sample[1][:-1]
    time_y = sample[1][-1]
    weekend_x = sample[2][:-1]
    weekend_y = sample[2][-1]
    type_x = sample[3][:-1]
    type_y = sample[3][-1]
    cat_x = generate_cat_vec_seq(poi_x, type_x, sample[4][:-1], POI_cat_distrib)
    cat_y = generate_cat_vec_seq([poi_y], [type_y], [sample[4][-1]], POI_cat_distrib)
    ground_truth = sample[5]
    location_x=sample[7][:-1]
    location_y=sample[7][-1]
    location_ground_truth = sample[8]
    type_neg = []
    cat_neg = []
    time_neg = []
    weekend_neg = []
    for neg_sample in sample[9]:
        type_neg.append(neg_sample[2])
        cat_neg.append(generate_cat_vec_seq([neg_sample[0]], [neg_sample[2]], [neg_sample[1]], POI_cat_distrib)[0])
        time_neg.append(time_y)
        weekend_neg.append(weekend_y)
    feed_dict['x_poi'] = [poi_x] 
    feed_dict['x_time'] = [time_x]
    feed_dict['x_weekend'] = [weekend_x]
    feed_dict['x_type'] = [type_x]
    feed_dict['x_cat'] = [cat_x]
    feed_dict['y_poi'] = [[poi_y]]
    feed_dict['y_time'] = [[time_y]]
    feed_dict['y_weekend'] = [[weekend_y]]
    feed_dict['y_type'] = [[type_y]]
    feed_dict['y_cat'] = [cat_y]
    feed_dict['true_y_cat'] = [[sample[6][-1]]] 
    feed_dict['ground_truth_set'] = [ground_truth]
    feed_dict['neg_time'] = [time_neg]
    feed_dict['neg_weekend'] = [weekend_neg]
    feed_dict['neg_type'] = [type_neg]
    feed_dict['neg_cat'] = [cat_neg]
    return feed_dict  

# evaluating function
def MAP_score(prediction, label):

    pred = prediction[0][0]

    true = label[0]

    visited_no = 0

    correct_no = 0

    total_sum = 0

    for guess in pred:

        visited_no += 1

        if guess in true:

            correct_no += 1

            total_sum += correct_no / visited_no
    return total_sum / len(label)