## Usage
### Training
```python
python train.py --config ${config_file} --disable_tqdm
```
- *config*  
file path to yaml config file
- *disable_tqdm*  
suppress tqdm progress bars when specified

### Predicting
```python
python predict.py --config ${config_file} --disable_tqdm
```
- *config*  
file path to yaml config file
- *disable_tqdm*  
suppress tqdm progress bars when specified

### Config
- *seed* (int)  
seed for random, numpy, and torch random number generation
- *deterministic* (bool)  
behaves deterministically without cuDNN backends when specified
- *amp* (bool)  
calculates in mixed precision when specified
- *epochs* (int)  
maximum number of epochs
- *load_checkpoint* (str)  
file path to checkpoint to continue off of
- *save_dir* (str)  
directory path to save to
- *model_name* (str) | *model_args* (dict)  
model to instantiate | instantiation arguments for model
- *train_dataset_args* (dict) | *val_dataset_args* (dict)  
instantiation arguments for train | validation dataset  
batch_size, shuffle, and num_workers used in pytorch's dataloader
- *loss_name* (str) | *loss_args* (dict)  
loss function to instantiate | instantiation arguments for loss function
- *optimizer_name* (str) | *optimizer_args* (dict)  
optimizer to instantiate | instantiation arguments for optimizer
- *scheduler_name* (str) | *scheduler_args* (dict)  
learning rate scheduler to instantiate | instantiation arguments for learning rate scheduler

