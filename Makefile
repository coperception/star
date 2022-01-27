training_script := train_codet.py
testing_script := test_codet.py
training_data = '/mnt/NAS/home/yiming/NeurIPS2021-DiscoNet/V2X-Sim-1.0-trainval/train'
testing_data = 'test_data'
bound := 'lowerbound'
com := 'sum'
batch_size := 4
epoch_num := 0
nepoch := 100
resume := './logs/$(com)/epoch_$(epoch_num).pth'

train:
	CUDA_VISIBLE_DEVICES=2 python $(training_script) --data $(training_data) --bound $(bound) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch)

train_upperbound:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --nepoch $(nepoch)

train_disco:
	python $(training_script) --data $(training_data) --bound $(bound) --com $(com) --log --batch $(batch_size) --nepoch $(nepoch) --kd_flag 1 --resume_teacher ./logs/upperbound/epoch_$(nepoch).pth

test:
	python $(testing_script) --data $(testing_data) --bound $(bound) --com $(com) --resume ./logs/$(com)/epoch_$(nepoch).pth --tracking

test_when2com:
	 python $(testing_script) --data $(test_data) --bound $(bound) --com $(com) --resume ./logs/$(com)/epoch_$(nepoch).pth --tracking --inference activated
