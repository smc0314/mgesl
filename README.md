# MGESL

This is the initial version of MGESL model and we will continue to optimize the structure of the code in the future.


### Installation
```
conda create -n mgesl python=3.7

conda activate mgesl

pip install -r requirements.txt
```



## Train Models
#### Configure your own path
```
main.py (line 240-256)、load_data.py (line 8-9, 33-34)、knowledge_graph (line 161-162)、save_data (line 48-60)
```


#### Process data

For all the datasets, the following command can be used to get the two kinds of fine-grained history and the coarse-grained history of their entities.
```
cd src
python get_history.py --dataset ICEWS14
python get_1hop_history.py --dataset ICEWS14

cd hgls
python generate_data.py --data=ICEWS14
python save_data.py --data=ICEWS14
```



#### Train models under candidate entity unknown setting

Then the following commands can be used to train MGESL under the candidate entity unknown setting.


```
python main.py -d ICEWS14  --train-history-len 5 --test-history-len 5  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --use-prelearning --gpu 0 --save checkpoint
```



#### Evaluate models

The following commands can be used to evaluate MGESL.

```
python main.py -d ICEWS14  --train-history-len 5 --test-history-len 5  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --use-prelearning --gpu 0 --save checkpoint --test 
```

#### Train models under candidate entity known setting

Configure the save_data.py
```
save_data (line 48-60)
change 'noleak_length_' in each path to 'leak_length'
```

Configure the src/hgls/utils.py
```
Uncomment line 108、132、156
Comment out line 110、134-138、158-163、
```

Configure the main.py
```
main.py (line 240-256)
change 'noleak_length_' in each path to 'leak_length'
```

Then the following commands can be used to train MGESL under the candidate entity known setting (the same as the candidate entity unknown setting).

```
python main.py -d ICEWS14  --train-history-len 5 --test-history-len 5  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --use-prelearning --gpu 0 --save checkpoint
```

The following commands can be used to evaluate MGESL (the same as the candidate entity unknown setting).
```
python main.py -d ICEWS14  --train-history-len 5 --test-history-len 5  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction  --use-prelearning --gpu 0 --save checkpoint --test
```
## Acknowledge

Our code is based on TiRGN(https://github.com/Liyyy2122/TiRGN) and HGLS(https://github.com/CRIPAC-DIG/HGLS)