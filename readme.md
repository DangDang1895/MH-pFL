cnn:

python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True --topk True
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True --cluster-flag True

more model:

python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn False
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn False --topk True
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn False --cluster-flag True

new_client:
cnn:

TRAIN
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn True --save-model True
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn True --cluster-flag True --save-model True

TEST
python run_test.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --test-clients 20 --only-cnn True --hynet-dir Output/cnn_cifar100_cn_100_tr_80_0hynet.pt --fc-dir Output/cnn_cifar100_cn_100_tr_80_0fc.pt 

more model:

TRAIN
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn False --save-model True
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn False --cluster-flag True --save-model True

TEST
python run_test.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --test-clients 20 --test-more-model True --hynet-dir Output/more_model_cifar100_cn_100_tr_80_0hynet.pt 

python run_test.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --test-clients 20 --only-cnn False --test-more-model False --hynet-dir Output/more_model_cifar100_cn_100_tr_80_0hynet.pt --fc-dir Output/more_model_cifar100_cn_100_tr_80_0fc.pt 

