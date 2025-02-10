you can run MH-pFedHN or MH-pFedHNGD:
python MH-pFedHN.py --data-path data --data-name cifar100 --num-nodes 50 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True
python MH-pFedHNGD.py --data-path data --data-name cifar100 --num-nodes 50 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True

TEST
python run.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn True --save-model True

python run_test.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --test-clients 20 --only-cnn True --hynet-dir Output/cnn_cifar100_cn_100_tr_80_0hynet.pt --fc-dir Output/cnn_cifar100_cn_100_tr_80_0fc.pt 


