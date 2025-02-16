#!/bin/bash


python main.py --conf new_conf/PEMS-Few/retrain_st_pems-few.json --gpuid 2 --seed 42 &

python main.py --conf new_conf/PEMS-Few/retrain_st_pems-few.json --gpuid 2 --seed 43 &

python main.py --conf new_conf/PEMS-Few/retrain_st_pems-few.json --gpuid 2 --seed 44 &

python main.py --conf new_conf/PEMS-Few/retrain_st_pems-few.json --gpuid 2 --seed 45 &

python main.py --conf new_conf/PEMS-Few/retrain_st_pems-few.json --gpuid 2 --seed 46 &




python main.py --conf new_conf/PEMS-Few/pretrain_st_pems-few.json --load_first_year 1 --first_year_model_path "log/PEMS-Few/retrain_st_pems-few-42/2011/16.7513.pkl" --gpuid 2 --seed 42 &

python main.py --conf new_conf/PEMS-Few/pretrain_st_pems-few.json --load_first_year 1 --first_year_model_path "log/PEMS-Few/retrain_st_pems-few-43/2011/16.3668.pkl" --gpuid 2 --seed 43 &

python main.py --conf new_conf/PEMS-Few/pretrain_st_pems-few.json --load_first_year 1 --first_year_model_path "log/PEMS-Few/retrain_st_pems-few-44/2011/16.5548.pkl" --gpuid 2 --seed 44 &

python main.py --conf new_conf/PEMS-Few/pretrain_st_pems-few.json --load_first_year 1 --first_year_model_path "log/PEMS-Few/retrain_st_pems-few-45/2011/16.5332.pkl" --gpuid 2 --seed 45 &

python main.py --conf new_conf/PEMS-Few/pretrain_st_pems-few.json --load_first_year 1 --first_year_model_path "log/PEMS-Few/retrain_st_pems-few-46/2011/16.4992.pkl" --gpuid 2 --seed 46 &



python main.py --conf new_conf/PEMS-Few/oneline_st_nn_pems-few.json --gpuid 2 --seed 42 &

python main.py --conf new_conf/PEMS-Few/oneline_st_nn_pems-few.json --gpuid 2 --seed 43 &

python main.py --conf new_conf/PEMS-Few/oneline_st_nn_pems-few.json --gpuid 2 --seed 44 &

python main.py --conf new_conf/PEMS-Few/oneline_st_nn_pems-few.json --gpuid 2 --seed 45 &

python main.py --conf new_conf/PEMS-Few/oneline_st_nn_pems-few.json --gpuid 2 --seed 46 &



python main.py --conf new_conf/PEMS-Few/oneline_st_an_pems-few.json --gpuid 2 --seed 42 &

python main.py --conf new_conf/PEMS-Few/oneline_st_an_pems-few.json --gpuid 2 --seed 43 &

python main.py --conf new_conf/PEMS-Few/oneline_st_an_pems-few.json --gpuid 2 --seed 44 &

python main.py --conf new_conf/PEMS-Few/oneline_st_an_pems-few.json --gpuid 2 --seed 45 &

python main.py --conf new_conf/PEMS-Few/oneline_st_an_pems-few.json --gpuid 2 --seed 46 &



python main.py --conf new_conf/PEMS-Few/trafficstream.json --gpuid 1 --seed 42 &

python main.py --conf new_conf/PEMS-Few/trafficstream.json --gpuid 1 --seed 43 &

python main.py --conf new_conf/PEMS-Few/trafficstream.json --gpuid 1 --seed 44 &

python main.py --conf new_conf/PEMS-Few/trafficstream.json --gpuid 1 --seed 45 &

python main.py --conf new_conf/PEMS-Few/trafficstream.json --gpuid 1 --seed 46 &


python stkec_main.py --conf new_conf/PEMS-Few/stkec.json --gpuid 0 --seed 42 &

python stkec_main.py --conf new_conf/PEMS-Few/stkec.json --gpuid 0 --seed 43 &

python stkec_main.py --conf new_conf/PEMS-Few/stkec.json --gpuid 0 --seed 44 &

python stkec_main.py --conf new_conf/PEMS-Few/stkec.json --gpuid 0 --seed 45 &

python stkec_main.py --conf new_conf/PEMS-Few/stkec.json --gpuid 0 --seed 46 &



python main.py --conf new_conf/PEMS-Few/eac.json --gpuid 0 --seed 42 &

python main.py --conf new_conf/PEMS-Few/eac.json --gpuid 0 --seed 43 &

python main.py --conf new_conf/PEMS-Few/eac.json --gpuid 0 --seed 44 &

python main.py --conf new_conf/PEMS-Few/eac.json --gpuid 0 --seed 45 &

python main.py --conf new_conf/PEMS-Few/eac.json --gpuid 0 --seed 46 &
