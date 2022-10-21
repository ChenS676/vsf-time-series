Code for the paper - [Multi-Variate Time Series Forecasting on Variable Subsets]() accepted at KDD 2022 Research Track.

# ToDo List
- metr-la， ecg,  exchange rate, solar energy dataset doesn't work with script "train_multi_step"
- pems_bay too 
- already works datasets: traffic, solar-energy, electricity, metr-la
- Adapt current train script into kiglis, what is num_dim and utw. 
- add information print to visualize the architecture of the model 


# Download dataset 
Multivariate time series datasets
Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them and move them to the data folder.

# Running the model

Datasets - METR-LA, SOLAR, TRAFFIC, ECG. This code provides a running example with all components on [MTGNN](https://github.com/nnzhan/MTGNN) model (we acknowledge the authors of the work).

## Standard Training
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --step_size1 {3} --mask_remaining {4}
```
Here, <br />
{0} - refers to the dataset directory: ./data/{ECG/TRAFFIC/METR-LA/SOLAR} <br />
{1} - refers to the model name <br />
{2} - refers to the manually assigned "ID" of the experiment  <br />
{3} - step_size1 is 2500 for METR-LA and SOLAR, 400 for ECG, 1000 for TRAFFIC <br />
{4} - inference post training in the partial setting, set to true or false. Note - mask_remaining is the alias for "Partial" setting in the paper
* random_node_idx_split_runs - the number of randomly sampled subsets per trained model run
* lower_limit_random_node_selections and upper_limit_random_node_selections - the percentage of variables in the subset **S**.


### Training with predefined subset S, the S apriori setting
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 50 --predefined_S --random_node_idx_split_runs 1 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --step_size1 {3}
```


### Training the model with Identity matrix as Adjacency
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --adj_identity_train_test --random_node_idx_split_runs 100 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --step_size1 {3}
```


## Inference

### Partial setting inference
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --mask_remaining True
```
* Note that epochs are set to 0 and mask_remaining (alias of "Partial" setting in the paper) to True


### Oracle setting inference
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 100 --upper_limit_random_node_selections 100 --do_full_set_oracle true --full_set_oracle_lower_limit 15 --full_set_oracle_upper_limit 15
```


## Our Wrapper Technique
```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 0 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --borrow_from_train_data true --num_neighbors_borrow 5 --dist_exp_value 0.5 --neighbor_temp 0.1 --use_ewp True
```


## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt


## Data Preparation


### Multivariate time series datasets

Download Solar and Traffic datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.

Download the METR-LA dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git). Move them into the data folder. (Optinally - download the adjacency matrix for META-LA from [here](https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl) and put it as ./data/sensor_graph/adj_mx.pkl , as shown below):
```
wget https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl
mkdir data/sensor_graph
mv adj_mx.pkl data/sensor_graph/
```

Download the ECG5000 dataset from [time series classification](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).

```

# Create data directories
mkdir -p data/{METR-LA,SOLAR,TRAFFIC,ECG}

# for any dataset, run the following command
python generate_training_data.py --ds_name {0} --output_dir data/{1} --dataset_filename data/{2}
```
Here <br />
{0} is for the dataset: metr-la, solar, traffic, ECG <br />
{1} is the directory where to save the train, valid, test splits. These are created from the first command <br />
{2} the raw data filename (the downloaded file), such as - ECG_data.csv, metr-la.hd5, solar.txt, traffic.txt


## Citation

```
@inproceedings{10.1145/3534678.3539394,
author = {Chauhan, Jatin and Raghuveer, Aravindan and Saket, Rishi and Nandy, Jay and Ravindran, Balaraman},
title = {Multi-Variate Time Series Forecasting on Variable Subsets},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539394},
doi = {10.1145/3534678.3539394},
abstract = {},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {76–86},
numpages = {11},
keywords = {multivariate time series forecasting, variable subsets, partial inference, retrieval model},
location = {Washington DC, USA},
series = {KDD '22}
}
```

```
gunzip  solar-energy/solar_AL.txt.gz

gunzip  traffic/traffic.txt.gz 

python generate_training_data.py --ds_name data/electricity --output_dir data/electricity --dataset_filename data/electricity/electricity.txt 

python generate_training_data.py --ds_name data/solar-energy --output_dir data/solar-energy --dataset_filename data/solar-energy/solar_AL.txt

python generate_training_data.py --ds_name data/traffic  --output_dir data/traffic --dataset_filename data/traffic/traffic.txt

python generate_training_data.py --ds_name metr-la  --output_dir data/metr-la --dataset_filename data/metr-la/metr-la.h5




python train_multi_step.py --data ./data/electricity --model_name mtgnn --device cuda:0 --expid 3242 --epochs 100 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --step_size1 1000 --mask_remaining true



python train_multi_step.py --data ./data/traffic --model_name mtgnn --device cuda:0 --expid 3242 --epochs 100 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --step_size1 1000 --mask_remaining true
```

| Datasets    | Description |
| ----------- | ----------- |
| Metr-La   | This dataset contains average traffic speed measured by 207 loop detectors on the highways of Los Angeles ranging from Mar 2012 to Jun 2012.|
| Solar   |  This dataset contains the solar power output that was collected from 137 plants in Alabama State in 2007     |
| Traffic | This dataset contains road occupancy rates that were measured by 862 sensors in San Francisco Bay area during 2015 and 2016 . Since the default scale of a substantial fraction of values is of the order of 1e −3, we upscale (multiply the variable values) by 1e3. |
| ECG5000| This dataset from the UCR time-series Classification Archive consists of 140 electrocardiograms (ECG) with a length of 5000 each (we use it for the purpose of forecasting, as done by [5]). |
| ----------- | ----------- |
| ----------- | ----------- |

https://github.com/nnzhan/MTGNN 

https://ojs.aaai.org/index.php/AAAI/article/view/3881 


## R eferences
https://github.com/liyaguang/DCRNN 

https://github.com/microsoft/StemGNN 


[16] Modeling Long and Short-Term Temporal Patterns with Deep Neural Networks git@github.com:laiguokun/LSTNet.git

[8] Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN) https://github.com/guoshnBJTU/ASTGCN-r-pytorch

[25] Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Network 