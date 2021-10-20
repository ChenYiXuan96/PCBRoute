
## Dependencies

* Python>=3.6
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib 

## File arrangement

The code is built on top of the [attention model](https://github.com/wouterkool/attention-learn-to-route) that is originally used to solve other problems. As a result, some of the code is no longer used, but still kept. Files regarding the code that are used in this project is to be introduced.

In the base directory, the most important file is __run.py__, which is called in terminal to do all experiments with different arguments. It is used for both training and validating. __train.py__ contains functions to train all the models, including attention model with REINFORCE and imitation learning models, which is called primarily by __run.py__.

The __copt.so__ is the Python module provided by Zuken, which is called in other files.

The __reinforce_baselines.py__ contains code that defines the class of baselines, including exponential baseline and rollout baseline.

The __options.py__ contains code to process arguments input from terminal. It is called by __run.py__.

The __successive_halving.py__ is a file that is used directly rather than calling __run.py__ first. It is used to do hyperparameter tuning using successive halving.

The __gen_pcb_offline_train_data.py__ contains code to produce demos for imitation learning, and validation dataset that is used across many experiments to compare the performance.

The __nets__ package involves the code to build the structure of DNNs. __attention_model.py__ contains the structure of the decoder, and all classes in the file inherit the nn.Module from PyTorch. The structure of the encoder is defined in __graph_encoder.py__. 

The __problems__ package contains the code regarding the problem (i.e. multiple-routing problem). Only the subpackage __pcb_route__ is used in this project. Code in __problem_pcb_route.py__ is used to produce datasets and evaluate solutions. __state_pcb_route.py__ contains code that logs the information regarding the partial solution while in the training process.

The __outputs__ package does not contain the code, but all the trained models that are in the form of .pt file. It contains trained models for all the experiments, thus the results could be reproduced without spending long time to train.

The __pre_gen_data__ package contains the demo data used for imitation learning and validating.

## Instructions on Running the Code

To do hyperparameter tuning:

```bash
python3 successive_halving.py --n_para_sets 128
```

Input this code to terminal to do hyperparameter tuning with 128 randomly selected hyperparameter sets.

To run the vanilla attention model on the five-pair problem:
```bash
    python run.py --problem PcbRoute --graph_size 5 --baseline rollout 
    --val_size 64 --embedding_dim 128 --n_epochs 150
    --eval_batch_size 1 --lr_decay 0.9573
    --n_encode_layers 4 --penalty_per_node 23111.415
    --run_name 'PcbRoute5_optimized'
```
where problem PcbRoute is used to indicate that this runs for the multiple-routing problem of this project. graph_size specifies the number of terminal-pair to connect. n_epochs sets the total epochs to train.

To run the attention model with BC on the five-pair problem:

```bash
    python run.py --problem PcbRoute --graph_size 5 --baseline rollout
    --val_size 64 --embedding_dim 128
    --hidden_dim 128 --n_epochs 150
    --eval_batch_size 1 --lr_decay 0.9573
    --n_encode_layers 4 --penalty_per_node 23111.415
    --run_name 'PcbRoute5_optimized_bc' --use_BC 1
    --BC_demos_path 'pre_gen_data/pcb_5_5k_bruteforce_data.json' 
    --BC_n_epochs 10 --lr_model_BC 0.001
```

where use_BC indicates that BC in used in this run. BC_n_epochs specify the number of epochs to pretrain the model using BC. This must be less than or equal to ten, because only 50 k demos are generated.

To run the attention model with DAR on the five-pair problem:

```bash
    python run.py --problem PcbRoute --graph_size 5
    --baseline rollout --batch_size 64 --epoch_size 4096
    --val_size 64 --embedding_dim 128 --hidden_dim 128 --n_epochs 150
    --eval_batch_size 1 --lr_decay 0.9573 --n_encode_layers 4
    --penalty_per_node 23111.415 --run_name 'PcbRoute5_optimized_dapg'
    --use_BC_DAPG 1 --BC_demos_path 'pre_gen_data/pcb_5_5k_bruteforce_data.json'
    --BC_n_epochs 10 --lr_model_BC 0.001 --DAPG_actor_ratio 0.8
```

where the attention model with DAR is indicated by use_BC_DAPG.

To run experiments using normalised input and reward, append the argument --normalize_input_reward 1 to the corresponding command. To run experiments for the eight-pair problem, modifications on graph_size, run_name, and BC_demos_path need to be made. An example is:

```bash
    python run.py --problem PcbRoute --graph_size 8 --baseline rollout
    --batch_size 64 --epoch_size 4096 --val_size 64 --embedding_dim 128
    --hidden_dim 128 --n_epochs 110 --eval_batch_size 1 --lr_decay
    0.9573 --n_encode_layers 4 --penalty_per_node 23111.415 --run_name
    'PcbRoute8_optimized_normalize_DAR' --use_BC_DAPG 1 --BC_demos_path
    'pre_gen_data/pcb_8_50k_bruteforce_data_9100.json' --BC_n_epochs 10
    --lr_model_BC 0.001 --normalize_input_reward 1
```

This code runs experiment using the attention model with DAR and normalisation on the eight-pair problem. To evaluate models using checkpoints of different models, eval_only, load_path, and val_dataset need to be specified:

```bash
    python run.py --problem PcbRoute --graph_size 8 --baseline rollout
    --batch_size 64 --epoch_size 4096 --val_size 2048 --embedding_dim
    128 --hidden_dim 128 --n_epochs 110 --eval_batch_size 1 --lr_decay
    0.9573 --n_encode_layers 4 --penalty_per_node 23111.415 --run_name
    'eval' --use_BC_DAPG 1 --BC_demos_path
    'pre_gen_data/pcb_8_50k_bruteforce_data_9100.json' --BC_n_epochs 10
    --lr_model_BC 0.001 --normalize_input_reward 1 --eval_only
    --load_path 'outputs/PcbRoute_8/PcbRoute8_optimized_normalize_DAR_
    20200914T221819/epoch-109.pt' --val_dataset
    'pre_gen_data/pcb_8_3k_validate.json'
```