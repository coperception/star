# Where to save the created data
# create_data_save_path := /scratch/dm4524/data/V2X-Sim-det
create_data_save_path := /mnt/NAS/data/dekun/V2X-Sim-det

# Path to the created training data
training_data := $(create_data_save_path)/train
test_data := $(create_data_save_path)/test

# com := mean
batch_size := 1
# Where to store the logs
logpath := logs-amortized-ts1-complement=classification
# Train for how many epochs
nepoch := 50
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)
fine_tune := 1

# args for mae
mask := complement
mask_ratio := 0.75
time_stamp := 1
com := ind_mae
mae_model := amortized_ind_patch8 #ind_bev_mae_vit_base_patch16
lr := 1e-3
min_lr := 0.
mae_load_path := logs-amortized-ts1-complement=classification/ind_mae/with_cross/completion_epoch_14.pth
test_batch := 1


train_completion:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr)

train_det:
	python train_codet.py --data $(training_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) --resume_completion $(logpath)/$(com)/with_cross/completion_epoch_1.pth --fine_tune $(fine_tune)

test_completion:
	python test_completion.py --data $(training_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(mae_load_path)
