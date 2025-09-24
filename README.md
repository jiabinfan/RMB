## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```
Then install the environment. Note that we found error in transformers==4.46.1, please use early versions.
```
pip install -r requirements.txt
```

### Training of reward models
Then, go to the `scripts' folder and train the reward model with the default hyperparameters
```
cd scripts
sh train_hsic_rms.sh  # learn diverse RMs
sh rmb_boosting.sh    $ boosting RMs
```

### RLHF

#### Data Generation

```
cd scripts/rlhf
sh data_generation4rlhf.sh
```

#### BoN
Go to the `scripts/rlhf/bon` folder and run the scripts. For more information about each step, please refer to `rlhf/bon/README.md` .

**Note: please set the path to your dataset and reward model in the corresponding shells.**
```
cd scripts/rlhf/bon
sh step1_train_proxy_reward_model_baseline.sh
sh step1_train_proxy_reward_model_grm.sh
sh step1_train_proxy_hsic_rms.sh
sh step1_train_proxy_rmb_boosting.sh
sh step2_generate_samples.sh
sh step3_obtain_proxy_score.sh
sh step4_choose_best_of_n.sh
sh step5_obtain_bon_gold_score.sh
sh step6_collect.sh
```

#### PPO
Go to the `scripts/rlhf/ppo' folder and train the gemma-2b-it model with the default parameters.

**Note: please set the path to your reward model in the corresponding shells.**
```
cd scripts/rlhf/ppo
sh train_ppo.sh
sh train_ppo.grm.sh
sh train_ppo_ensemble_baseline.sh
sh_train_ppo_rmb.sh
```


## Acknowledgment
This repo is built upon [GRM](https://github.com/YangRui2015/Generalizable-Reward-Model) [transformers](https://github.com/huggingface/transformers) and [trl](https://github.com/huggingface/trl), with also inspiration from [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling). 
