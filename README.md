# STSP

This repository is the implementation of STSP 

STSP is a model proposed in 'A Win-Win Solution of Next POI Recommendation for Users-Businesses with Uncertain Check-ins'. STSP is a novel framework, equipped with category- and location-aware encoders, which is designed to achieve next category and POI prediction with uncertain check-ins by fusing rich context features.


### Files in the folder

- `data/`
  - `mall_Info_CAL.csv`: raw mall information of Calgary;
  - `CAL_checkin.csv`: raw checkin information of Calgary;
  - `CAL_checkin_reindexed.csv`: reindexed checkin information of Calgary;
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



### Required packages
The code has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
- tensorflow==2.0.0
- numpy==1.17.3
- pandas=0.25.3
- keras==2.3.1



### More Experimental Settings
- Environments
  -Our proposed STSP and the deep learning based baseline, namely MCARNN are implemented using Tensorflow 2.0.0, with Python 3.6.9 from Anaconda 4.7.12. All the conventional baselines, including MostPop, CateMF, LBPR are implemented with Python 3.6.9. For HCT, we directly use the source code provided by the authors.
All the experiments are carried out on a machine with Windows 10, Intel CORE i7-8565U CPU and 16G RAM.



### Running the code
```
$ python main.py (note: use -h to check optional arguments)
```
