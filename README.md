# STSP

This repository is the implementation of STSP 

STSP is a model proposed in 'Point-of-Interest Recommendation forUsers-Businesses with Uncertain Check-ins'. STSP is a novel framework, equipped with category- and location-aware encoders, which is designed to achieve next category and POI prediction with uncertain check-ins by fusing rich context features.


### Files in the folder

- `data/`
  - `CAL`,`CHA`,`PHO`. The three files are similar in structure, as explained here by CAL as an example.
    - `All_POI_CAL.csv`: all POIs of Calgary;
    - `All_collective_POI_info_CAL.csv`: raw collective POIs information of Calgary;
    - `Check-ins_CAL.csv`: reindexed check-ins information of Calgary;
- `category result/`
  - `CAL`
    - `user_rep_CAL`: user embedding folder of category encoder module, there are many .npy files;
    - `L2_id_mapping_CAL.csv`: category id mapping file;
    - `reindex_data_CAL.csv`: reindexed and filtered checkin file;
    - `result_CAL.txt`: the original category recommendation result of category encoder module;
    - `train_CAL.txt`: the original train data of category encoder;
- `main.py`: main file;
- `data_preprocess.py`: data preprocess file;
- `category_encoder.py`: category encoder module;
- `POI_encoder.py`: POI encoder module.


### More Experimental Settings
- Environment
  - Our proposed STSP and the deep learning based baseline, namely MCARNN are implemented using Tensorflow 2.0.0, with Python 3.6.9 from Anaconda 4.7.12. All the conventional baselines, including MostPop, CateMF, LBPR are implemented with Python 3.6.9. For HCT, we directly use the source code provided by the authors. All the experiments are carried out on a machine with Windows 10, Intel CORE i7-8565U CPU and 16G RAM. The following packages are needed (along with their dependencies):
    - tensorflow==2.0.0
    - numpy==1.17.3
    - pandas=0.25.3
    - keras==2.3.1
- Data Preprocessing
  - Following state-of-the-arts,  we filter out users and POIs with less than 10 check-in records. For each user, we split her check-in records into sequences by day, where the earlier 80\% of her sequences are used as training set; the latest 10\% of her sequences are test sets; and the rest 10\% in the middle is treated as validation set to help tune the hyper-parameters.
- Hyper-parameter Settings
  - Tables (1-5) summarize the optimal settings for the hyper-parameters of all the methods.
  
  Table1: Hyper-parameter settings for our STSP.
    |Hyper-paramters|CHA|PHO|CAL|
    |:---:|:---:|:---:|:---:|
    |learning rate for category prediction task η | 0.0001| 0.0001| 0.0001|
    learning rate for POI prediction task η | 0.0001| 0.0001| 0.001|
    regularization term λ|0.0025|0.0025|0.0025|
    number of recurrent layers| 3|3|3|
    embedding size D|120|100|120|
    category importance α|0.4|0.5|0.5|
    
  Table2: Hyper-parameter settings for CateMF.    
    |Hyper-paramters|CHA|PHO|CAL|
    |:---:|:---:|:---:|:---:|
    |learning rate η | 0.001| 0.001| 0.001|
    embedding size D|100|100|120|
  
  Table3: Hyper-parameter settings for LBPR.
    |Hyper-paramters|CHA|PHO|CAL|
    |:---:|:---:|:---:|:---:|
    |learning rate η | 0.001| 0.001| 0.001 |
    |embedding size D|100|100|120|
    |list size α|2|2|2|
 
  Table4: Hyper-parameter settings for MCARNN.
    |Hyper-paramters|CHA|PHO|CAL|
    |:---:|:---:|:---:|:---:|
    |learning rate η| 0.01| 0.01| 0.01 |
    |embedding size D | 200|200| 200 |
  
  Table5: Hyper-parameter settings for HCT.
    |Hyper-paramters|CHA|PHO|CAL|
    |:---:|:---:|:---:|:---:|
    |learning rate η | 0.001| 0.001| 0.001 |
    |embedding size  D | 200| 200| 200 |
    |window size ω| 2| 2 |2 |
    |weights of categories at layer 1 & 2 | 0.2&0.8| 0.2&0.8| 0.2&0.8|
    
  
- Training/Testing Time
  - The training and testing time of our STSP on the three real-world datasets are listed in Table 6.
  
  Table 6: Training and testing time (seconds) of STSP.
    |       | Training | Testing |
    |:-----:|:--------:|:-------:|
    |CHA    | 509.05   | 433.60  |
    |PHO    | 523.57   | 798.34  |
    |CAL    |111.61    | 79.10   |




### Running the code
```
$ python main.py (note: use -h to check optional arguments)
```
