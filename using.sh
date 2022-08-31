python train.py --model (model name) --dataset (dataset name) --exp (exp name)
python train.py --model (model name) --dataset (dataset name) --exp (exp name)
python train.py --model dehazeformer-t --dataset RESIDE-IN --exp indoor
python train.py --model dehazeformer-t --dataset RESIDE-IN --exp indoor --gpu '0'
python train.py --model dehazeformer-t --dataset RESIDE-IN --exp indoor --gpu '0,1,2'
python test.py --model dehazeformer-t --dataset RESIDE-OUT --exp outdoor
python train.py --model dehazeformer-t --dataset RESIDE-OUT --exp outdoor --gpu '0'