Extended ACT Policy for use in [RoboTwin 2.0 Repo](https://github.com/robotwin-Platform/robotwin).  

This policy splits the left and right arm training pipelines so they do not share actions. 

# Data Creation

## Collect Demos
RoboTwin has a built in [data creation pipeline](https://robotwin-platform.github.io/doc/usage/collect-data.html). From the RoboTwin root you can run:
```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```
task_name: the name of a task from the [50 Bimanual Task List](https://robotwin-platform.github.io/doc/tasks/index.html). (i.e., handver_block, grab_roller)
task_config: comes from config file under RoboTwin/task_config. You can change the number of episodes or add domain randomization in the config file. (Usually demo_clean)

## Prepare Data 
This data needs to be converted in hdf5 files for the ACT pipeline. You can run:
```
bash policy/ACTOracleSplit/process_data.sh ${task_name} ${task_config} ${num_episodes}
```
num_episodes: number of episodes to process, maxes out at the value defined in the task_config. 

# Training 


```
bash policy/ACTOracleSplit/{train_script}.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
```
train_script: bash script config for training.
expert_data_num: number of training demos to use (Usually equal to num_episodes)
seed: random training seed (Generally set to zero).




# Eval
