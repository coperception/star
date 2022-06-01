# Where to save the created data
create_data_save_path := /scratch/dm4524/data/V2X-Sim-det

# Path to the created training data
training_data := $(create_data_save_path)/train

com := mean
batch_size := 4
# Where to store the logs
logpath := logs
# Train for how many epochs
nepoch := 20
# If given, the model will resume from its most recent (by modification time) check point
auto_resume_path := $(logpath)
teacher_trainable := 1


train_completion:
	python train_completion.py --data $(training_data) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --logpath $(logpath)

train_det:
	python train_codet.py --data $(training_data) --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(nepoch) --logpath $(logpath) --resume_teacher $(logpath)/$(com)/with_cross/teacher_epoch_1.pth --teacher_trainable $(teacher_trainable)
