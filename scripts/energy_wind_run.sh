#!/bin/bash



python main.py --conf conf/ENERGY-Wind/retrain_st_energy-wind.json --gpuid 2 --seed 42

python main.py --conf conf/ENERGY-Wind/retrain_st_energy-wind.json --gpuid 2 --seed 43

python main.py --conf conf/ENERGY-Wind/retrain_st_energy-wind.json --gpuid 2 --seed 44

python main.py --conf conf/ENERGY-Wind/retrain_st_energy-wind.json --gpuid 2 --seed 45

python main.py --conf conf/ENERGY-Wind/retrain_st_energy-wind.json --gpuid 2 --seed 46




python main.py --conf conf/ENERGY-Wind/pretrain_st_energy-wind.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st_energy-wind-42/0/2.9985.pkl" --gpuid 2 --seed 42

python main.py --conf conf/ENERGY-Wind/pretrain_st_energy-wind.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st_energy-wind-43/0/3.0333.pkl" --gpuid 2 --seed 43

python main.py --conf conf/ENERGY-Wind/pretrain_st_energy-wind.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st_energy-wind-44/0/2.8954.pkl" --gpuid 2 --seed 44

python main.py --conf conf/ENERGY-Wind/pretrain_st_energy-wind.json --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/retrain_st_energy-wind-45/0/3.1246.pkl" --gpuid 2 --seed 45





python main.py --conf conf/ENERGY-Wind/oneline_st_nn_energy-wind.json --gpuid 1 --seed 42

python main.py --conf conf/ENERGY-Wind/oneline_st_nn_energy-wind.json --gpuid 1 --seed 43

python main.py --conf conf/ENERGY-Wind/oneline_st_nn_energy-wind.json --gpuid 1 --seed 44

python main.py --conf conf/ENERGY-Wind/oneline_st_nn_energy-wind.json --gpuid 1 --seed 45



python main.py --conf conf/ENERGY-Wind/oneline_st_an_energy-wind.json --gpuid 1 --seed 42

python main.py --conf conf/ENERGY-Wind/oneline_st_an_energy-wind.json --gpuid 1 --seed 43

python main.py --conf conf/ENERGY-Wind/oneline_st_an_energy-wind.json --gpuid 1 --seed 44

python main.py --conf conf/ENERGY-Wind/oneline_st_an_energy-wind.json --gpuid 1 --seed 45




python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 1 --seed 42

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 1 --seed 43

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 1 --seed 44

python main.py --conf conf/ENERGY-Wind/trafficstream.json --gpuid 1 --seed 45



python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 42

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 43

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 44

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 45






python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 42

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 43 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.88.pkl"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 44 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.8834.pkl"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 45 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.8834.pkl"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 46 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.88.pkl" 
