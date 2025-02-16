#!/bin/bash


python main.py --conf conf/AIR/retrain_st_air.json --gpuid 3 --seed 42

python main.py --conf conf/AIR/retrain_st_air.json --gpuid 3 --seed 43

python main.py --conf conf/AIR/retrain_st_air.json --gpuid 3 --seed 44

python main.py --conf conf/AIR/retrain_st_air.json --gpuid 3 --seed 45

python main.py --conf conf/AIR/retrain_st_air.json --gpuid 3 --seed 46



python main.py --conf conf/AIR/pretrain_st_air.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st_air-42/2016/23.7181.pkl" --gpuid 0 --seed 42

python main.py --conf conf/AIR/pretrain_st_air.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st_air-43/2016/17.2116.pkl" --gpuid 0 --seed 43

python main.py --conf conf/AIR/pretrain_st_air.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st_air-44/2015/24.6758.pkl" --gpuid 0 --seed 44

python main.py --conf conf/AIR/pretrain_st_air.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st_air-45/2015/23.3046.pkl" --gpuid 0 --seed 45

python main.py --conf conf/AIR/pretrain_st_air.json --load_first_year 1 --first_year_model_path "log/AIR/retrain_st_air-46/2015/23.925.pkl" --gpuid 0 --seed 46



python main.py --conf conf/AIR/oneline_st_nn_air.json --gpuid 3 --seed 42

python main.py --conf conf/AIR/oneline_st_nn_air.json --gpuid 3 --seed 43

python main.py --conf conf/AIR/oneline_st_nn_air.json --gpuid 3 --seed 44

python main.py --conf conf/AIR/oneline_st_nn_air.json --gpuid 3 --seed 45

python main.py --conf conf/AIR/oneline_st_nn_air.json --gpuid 3 --seed 46



python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 42

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 51

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 52

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 53

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 54

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 55

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 56

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 57

python main.py --conf conf/AIR/oneline_st_an_air.json --gpuid 3 --seed 58



python main.py --conf conf/AIR/trafficstream.json --gpuid 1 --seed 42

python main.py --conf conf/AIR/trafficstream.json --gpuid 1 --seed 43

python main.py --conf conf/AIR/trafficstream.json --gpuid 1 --seed 44

python main.py --conf conf/AIR/trafficstream.json --gpuid 1 --seed 45

python main.py --conf conf/AIR/trafficstream.json --gpuid 1 --seed 46



python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 42

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 43

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 44

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 45

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 46


python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 42

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 43

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 44

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 45

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 46
