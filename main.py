import argparse
import numpy as np
from data_preprocess import Category_Encoder, POI_Encoder
from category_encoder import category_encoder
from POI_encoder import POI_encoder

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CAL', help='dataset')
parser.add_argument('--augment_sample', type=bool, default= True, help='augment each sequence to increase sample size')
parser.add_argument('--min_seq_len', type=int, default=2, help='min sequence length')
parser.add_argument('--min_seq_num', type=int, default=2, help='min number of sequence about the same user')
parser.add_argument('--category_neg_sample_num', type=int, default=5, help='negative sample number about category encoder data')
parser.add_argument('--category_colname', type=str, default='Category', help='the column name of category in data file about CAL')
parser.add_argument('--train_portion', type=float, default=0.9, help='the ratio of train data')
parser.add_argument('--category_seed', type=int, default=2019, help='category encoder model random seed')
parser.add_argument('--RNN_stack_layers', type=int, default=3, help='category encoder model RNN stack layers number')
parser.add_argument('--hidden_size', type=int, default=120, help='hidden size in RNN')
parser.add_argument('--batch_size', type=int, default=1, help='category encoder model batch size')
parser.add_argument('--category_lr', type=float, default=0.0001, help='category encoder model learning rate')
parser.add_argument('--category_iter_num', type=int, default=25, help='category encoder model iteration number')
parser.add_argument('--category_break_threshold', type=float, default=0.001, help='category encoder model break threshold number')
parser.add_argument('--category_w_c', type=float, default=0.5, help='category encoder model POI category weight in loss')
parser.add_argument('--category_w_t', type=float, default=0.5, help='category encoder model POI type weight in loss')
parser.add_argument('--category_keep_prob', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--category_reg_beta', type=float, default=0.0025, help='category encoder model regularization parameter')
parser.add_argument('--category_time_size', type=int, default=24, help='category encoder model time value size')
parser.add_argument('--category_weekend_size', type=int, default=2, help='category encoder model weekend value size')
parser.add_argument('--POI_lr', type=float, default=0.001, help='POI encoder model learning rate')
parser.add_argument('--POI_iter_num', type=int, default=1000, help='POI encoder model iteration number')
parser.add_argument('--POI_reg_beta', type=float, default=0.0025, help='POI encoder model regularization parameter')
parser.add_argument('--POI_break_threshold', type=float, default=0.001, help='POI encoder model break threshold number')
parser.add_argument('--POI_seed', type=int, default=6, help='POI encoder model random seed')
parser.add_argument('--POI_candidate_num', type=int, default=10, help='POI recommend number')

args = parser.parse_args()

#---------------------data preprocess--------------------
Category_Encoder(args)
POI_Encoder(args)
#-----------------model training and test----------------
category_encoder(args)
POI_encoder(args)
