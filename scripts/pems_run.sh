#!/bin/bash


python main.py --conf conf/PEMS/retrain_st_pems.json --gpuid 2 --seed 42

python main.py --conf conf/PEMS/retrain_st_pems.json --gpuid 2 --seed 43

python main.py --conf conf/PEMS/retrain_st_pems.json --gpuid 2 --seed 44

python main.py --conf conf/PEMS/retrain_st_pems.json --gpuid 2 --seed 45

python main.py --conf conf/PEMS/retrain_st_pems.json --gpuid 2 --seed 46



python main.py --conf conf/PEMS/pretrain_st_pems.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st_pems-42/2011/16.435.pkl" --gpuid 2 --seed 42

python main.py --conf conf/PEMS/pretrain_st_pems.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st_pems-43/2011/16.0545.pkl" --gpuid 2 --seed 43

python main.py --conf conf/PEMS/pretrain_st_pems.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st_pems-44/2011/16.2672.pkl" --gpuid 2 --seed 44

python main.py --conf conf/PEMS/pretrain_st_pems.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st_pems-45/2011/16.879.pkl" --gpuid 2 --seed 45

python main.py --conf conf/PEMS/pretrain_st_pems.json --load_first_year 1 --first_year_model_path "log/PEMS/retrain_st_pems-46/2011/16.367.pkl" --gpuid 2 --seed 46



python main.py --conf conf/PEMS/oneline_st_nn_pems.json --gpuid 2 --seed 42

python main.py --conf conf/PEMS/oneline_st_nn_pems.json --gpuid 2 --seed 43

python main.py --conf conf/PEMS/oneline_st_nn_pems.json --gpuid 2 --seed 44

python main.py --conf conf/PEMS/oneline_st_nn_pems.json --gpuid 2 --seed 45

python main.py --conf conf/PEMS/oneline_st_nn_pems.json --gpuid 2 --seed 46



python main.py --conf conf/PEMS/oneline_st_an_pems.json --gpuid 2 --seed 51

python main.py --conf conf/PEMS/oneline_st_an_pems.json --gpuid 2 --seed 52

python main.py --conf conf/PEMS/oneline_st_an_pems.json --gpuid 2 --seed 53

python main.py --conf conf/PEMS/oneline_st_an_pems.json --gpuid 2 --seed 54

python main.py --conf conf/PEMS/oneline_st_an_pems.json --gpuid 2 --seed 55


python main.py --conf conf/PEMS/trafficstream.json --gpuid 2 --seed 42

python main.py --conf conf/PEMS/trafficstream.json --gpuid 2 --seed 43

python main.py --conf conf/PEMS/trafficstream.json --gpuid 2 --seed 44

python main.py --conf conf/PEMS/trafficstream.json --gpuid 2 --seed 45

python main.py --conf conf/PEMS/trafficstream.json --gpuid 2 --seed 46


python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 47

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 48

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 49

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 50

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 51



python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 51

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 52

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 53

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 54

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 55
