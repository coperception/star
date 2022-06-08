# Where to save the created data
#create_data_save_path := /scratch/dm4524/data/V2X-Sim-det
create_data_save_path := /mnt/NAS/data/dekun/V2X-Sim-det
test_model_path := /mnt/NAS/data/zjx/ind_fusion/

# Path to the created training data
training_data := $(create_data_save_path)/train
test_data := $(create_data_save_path)/test

# com := mean
batch_size := 1
# Where to store the logs
# logpath := logs-amortized-ts1-complement=classification
# Train for how many epochs
nepoch := 50
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)
fine_tune := 0

# args for mae
mask := complement
mask_ratio := 0.0
time_stamp := 1
recon := ind
com := $(recon)_mae
patch_size := 8
mae_model := amortized_$(recon)_patch$(patch_size)
lr := 1e-3
min_lr := 0.
# mae_load_path := logs-amortized-ts1-complement=classification/ind_mae/with_cross/completion_epoch_20.pth
test_load_path := $(test_model_path)logs-amortized-ts2-complement-ratio0.5-ps8=classification/ind_mae/no_cross/completion_epoch_3.pth
test_batch := 1
# logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)-convk11=classification
logpath := logs-amortized-ts$(time_stamp)-$(mask)=classification
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)

train_completion:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(auto_resume_path)

train_completion_with_cross:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --auto_resume_path $(auto_resume_path)

train_det:
	python train_codet.py --data $(training_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) --resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth --fine_tune $(fine_tune) --no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)

test_completion:
	python test_completion.py --data $(test_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(test_load_path) --no_cross_road

test_completion_save:
	python test_completion.py --data $(test_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(test_load_path) --no_cross_road --save_vis

test_completion_with_cross:
	python test_completion.py --data $(training_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(mae_load_path)

