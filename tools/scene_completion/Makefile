# Where to save the created data
#create_data_save_path := /scratch/dm4524/data/V2X-Sim-det
create_data_save_path := /mnt/NAS/data/dekun/V2X-Sim-det
seg_data_path := /mnt/NAS/data/dekun/V2X-Sim-seg
test_model_path := /mnt/NAS/data/zjx/ind_fusion/
upperbound_det_path := /mnt/NAS/data/dekun/upperbound_det_logs/no_cross/epoch_100.pth
upperbound_seg_path := /mnt/NAS/data/dekun/upperbound_seg_logs/no_cross/epoch_100.pth

# Path to the created training data
training_data := $(create_data_save_path)/train
test_data := $(create_data_save_path)/test
seg_train_data := $(seg_data_path)/train
seg_test_data := $(seg_data_path)/test

# com := mean
batch_size := 4
# Where to store the logs
# logpath := logs-amortized-ts1-complement=classification
# Train for how many epochs
nepoch := 100
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)
fine_tune := 0

# args for mae
mask := complement
mask_ratio := 0
time_stamp := 1
recon := ind
com := $(recon)_mae
patch_size := 8
mae_model := amortized_$(recon)_patch$(patch_size)
lr := 1e-3
min_lr := 0.
exp := classification
test_epoch := 3
# mae_load_path := logs-amortized-ts1-complement=classification/ind_mae/with_cross/completion_epoch_20.pth
test_load_path := $(test_model_path)logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=$(exp)/$(com)/no_cross/completion_epoch_$(test_epoch).pth
test_batch := 1
# logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=main
enc_logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=ablate-enc
dec_logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=ablate-dec
temp_logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=ablate-temp
logpath := logs-amortized-ts$(time_stamp)-$(mask)=classification
# If given, the model will resume from its most recent (by modification time) check point
det_logpath := logs-amortized-ts$(time_stamp)-$(mask)=det
auto_resume_path := $(logpath)

train_completion:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(auto_resume_path)

train_completion_with_cross:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --auto_resume_path $(auto_resume_path)

train_det:
	python train_codet.py --data $(training_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(det_logpath) --resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth --fine_tune $(fine_tune) --no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)

train_seg:
	python train_coseg.py --data $(seg_train_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) --resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth --fine_tune $(fine_tune) --no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)

test_completion:
	python test_completion.py --data $(test_data) --com $(com) --log --batch $(test_batch) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(test_load_path) --no_cross_road

test_completion_save:
	python test_completion.py --data $(test_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(test_load_path) --no_cross_road --save_vis

test_completion_with_cross:
	python test_completion.py --data $(training_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(mae_load_path)

test_det:
	python test_codet.py --data $(test_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) \
	--resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth \
	--resume $(upperbound_det_path) \
	--no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) \
	--visualization 0

test_seg:
	python test_coseg.py --data $(seg_test_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) \
	--resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth \
	--resume logs-amortized-ts1-complement=classification/mae/no_cross/seg_epoch_50.pth \
	--no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)


# ablation study
train_completion_ablate_encoder:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(enc_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(enc_logpath) --encode_partial

train_completion_ablate_decoder:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(dec_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(dec_logpath) --decode_singletemp

train_completion_ablate_temp:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(temp_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(temp_logpath) --no_temp_emb
