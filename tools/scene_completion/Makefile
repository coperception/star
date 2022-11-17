# Where to save the created data
#create_data_save_path := /scratch/dm4524/data/V2X-Sim-det
create_data_save_path := /mnt/NAS/data/dekun/V2X-Sim-det
seg_data_path := /mnt/NAS/data/dekun/V2X-Sim-seg
test_model_path := /mnt/NAS/data/zjx/ind_fusion/
upperbound_det_path := /mnt/NAS/data/dekun/upperbound_det_logs/no_cross/epoch_100.pth
lowerbound_det_path := /mnt/NAS/data/dekun/lowerbound_det_logs/no_cross/epoch_100.pth
upperbound_det_logpath := /mnt/NAS/data/dekun/upperbound_det_logs/no_cross/
upperbound_seg_path := /mnt/NAS/data/dekun/upperbound_seg_logs/no_cross/epoch_100.pth
lowerbound_seg_path := /mnt/NAS/data/dekun/lowerbound_seg_logs/no_cross/epoch_100.pth

# Path to the created training data
training_data := $(create_data_save_path)/train
test_data := $(create_data_save_path)/test
seg_train_data := $(seg_data_path)/train
seg_test_data := $(seg_data_path)/test

# com := mean
batch_size := 2
# Where to store the logs
# logpath := logs-amortized-ts1-complement=classification
# Train for how many epochs
nepoch := 20
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
mae_model := amortized_$(recon)_patch$(patch_size)_light
lr := 2e-4
min_lr := 0.
exp := main
test_epoch := 20
# args for focal
# args for vqmae
decay := 0.
commitment_cost := 0.25
num_vq_embeddings := 8192
vq_embedding_dim := 32
# paths
test_load_path := logs-amortized-ts1-complement-8=light/ind_mae/no_cross/completion_epoch_20.pth
# test_load_path := $(test_model_path)logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=$(exp)/$(com)/no_cross/completion_epoch_$(test_epoch).pth
cnn_test_load_path := /mnt/NAS/data/zjx/ind_fusion/logs=cnn/late/no_cross/completion_epoch_20.pth
test_batch := 1
# logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=main
enc_logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=ablate-enc
dec_logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=ablate-dec
temp_logpath := logs-amortized-ts$(time_stamp)-$(mask)-ratio$(mask_ratio)-ps$(patch_size)=ablate-temp
# logpath := logs-amortized-ts$(time_stamp)-$(mask)=classification
logpath := logs-amortized-ts$(time_stamp)-$(mask)-$(patch_size)=light
# If given, the model will resume from its most recent (by modification time) check point
det_logpath := logs-amortized-ts$(time_stamp)-$(mask)=det
auto_resume_path := $(logpath)
cnn_logpath := /mnt/NAS/data/zjx/ind_fusion/logs=cnn/
vqvae_logpath := /mnt/NAS/data/zjx/ind_fusion/logs=vqvae-debug/
vqvae_test_load_path := /mnt/NAS/data/zjx/ind_fusion/logs=vqvae-vae/vqvae/no_cross/completion_epoch_12.pth
vqstar_logpath := logs-vqstar-ts$(time_stamp)-$(mask)-$(patch_size)-vq$(num_vq_embeddings)dim$(vq_embedding_dim)=light+adam+minlr0+recon
vqstar_test_load_path := logs-vqstar-ts$(time_stamp)-$(mask)-$(patch_size)-vq$(num_vq_embeddings)dim$(vq_embedding_dim)=light+adam+minlr0+kmeansfly+decopos-b-vq/vqstar/no_cross/completion_epoch_10.pth

debug_completion:
	python train_completion.py --data $(training_data) --com $(com) --log --batch 1 --nepoch 1 --logpath logs --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road 

train_completion:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(auto_resume_path)

train_completion_cnn:
	python train_completion.py --data $(training_data) --com late --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(cnn_logpath) --lr $(lr) --no_cross_road  --auto_resume_path $(cnn_logpath)

train_completion_vqstar:
	python train_completion.py --data $(training_data) --com vqstar --log --batch $(batch_size) --nepoch $(nepoch) \
	--logpath $(vqstar_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) \
	--decay $(decay) --num_vq_embeddings $(num_vq_embeddings) --commitment_cost $(commitment_cost) --vq_embedding_dim $(vq_embedding_dim) \
	--lr $(lr) --no_cross_road  --auto_resume_path $(vqstar_logpath) --wandb

train_completion_vqvae:
	python train_completion.py --data $(training_data) --com vqvae --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(vqvae_logpath) --lr $(lr) --no_cross_road  --auto_resume_path $(vqvae_logpath) --wandb

train_completion_with_cross:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --auto_resume_path $(auto_resume_path)

train_det:
	python train_codet.py --data $(training_data) --com $(com) --log --batch $(batch_size) --resume $(upperbound_det_path) --nepoch $(nepoch) --logpath $(upperbound_det_logpath) --resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth --fine_tune $(fine_tune) --no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)

train_seg:
	python train_coseg.py --data $(seg_train_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) --resume_completion $(logpath)/$(com)/with_cross/completion_epoch_20.pth --fine_tune $(fine_tune) --no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)

test_completion:
	python test_completion.py --data $(test_data) --com $(com) --log --batch $(test_batch) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(test_load_path) --no_cross_road

test_completion_cnn:
	python test_completion.py --data $(test_data) --com late --log --batch $(test_batch) --load_path $(test_load_path) --no_cross_road

test_completion_vqvae:
	python test_completion.py --data $(test_data) --com vqvae --log --batch $(test_batch) --load_path $(vqvae_test_load_path) --no_cross_road --save_vis

test_completion_vqstar:
	python test_completion.py --data $(test_data) --com vqstar --log --batch $(test_batch) \
	--load_path $(vqstar_test_load_path) --no_cross_road --save_vis \
	--decay $(decay) --num_vq_embeddings $(num_vq_embeddings) --commitment_cost $(commitment_cost) --vq_embedding_dim $(vq_embedding_dim)

test_completion_save:
	python test_completion.py --data $(test_data) --com $(com) --log --batch $(test_batch) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(test_load_path) --no_cross_road  --save_vis

test_completion_with_cross:
	python test_completion.py --data $(training_data) --com $(com) --log --batch $(test_batch) --logpath $(logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --load_path $(mae_load_path)

test_det:
	python test_codet.py --data $(test_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath logs-amortized-ts1-complement-8=light+trydec1/ind_mae/no_cross/ \
	--resume_completion $(test_load_path) \
	--resume $(lowerbound_det_path) \
	--no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) \
	--visualization 0

test_det_cnn:
	python test_codet.py --data $(test_data) --com late --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(cnn_logpath) \
	--resume_completion $(cnn_test_load_path) \
	--resume /mnt/NAS/data/dekun/lowerbound_det_logs/no_cross/epoch_100.pth \
	--no_cross_road --visualization 1

test_det_vqvae:
	python test_codet.py --data $(test_data) --com vqvae --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(vqvae_logpath) \
	--resume_completion $(vqvae_test_load_path) \
	--resume /mnt/NAS/data/dekun/lowerbound_det_logs/no_cross/epoch_100.pth \
	--no_cross_road --visualization 0

test_seg:
	python test_coseg.py --data $(seg_test_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath logs-amortized-ts1-complement-8=light-a075-g3 \
	--resume_completion $(test_load_path) \
	--resume $(lowerbound_seg_path) --vis \
	--no_cross_road --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model)

test_seg_cnn:
	python test_coseg.py --data $(seg_test_data) --com late --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath) \
	--resume_completion $(cnn_test_load_path) \
	--resume $(lowerbound_seg_path) \
	--no_cross_road

test_seg_vqvae:
	python test_coseg.py --data $(seg_test_data) --com vqvae --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(vqvae_logpath) \
	--resume_completion $(vqvae_test_load_path) \
	--resume $(lowerbound_seg_path) \
	--no_cross_road

# ablation study
train_completion_ablate_encoder:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(enc_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(enc_logpath) --encode_partial

train_completion_ablate_decoder:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(dec_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(dec_logpath) --decode_singletemp

train_completion_ablate_temp:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(temp_logpath) --mask $(mask) --mask_ratio $(mask_ratio) --time_stamp $(time_stamp) --mae_model $(mae_model) --lr $(lr) --no_cross_road  --auto_resume_path $(temp_logpath) --no_temp_emb
