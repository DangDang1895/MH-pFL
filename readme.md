# MH-pFedHN and MH-pFedHNGD Usage Guide  

## Running MH-pFedHN and MH-pFedHNGD (Only CNN Model)  

To run the models with only CNN, use the following commands:  

```bash  
python MH-pFedHN.py --data-path data --data-name cifar100 --num-nodes 50 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True  
python MH-pFedHNGD.py --data-path data --data-name cifar100 --num-nodes 50 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn True
```

## Running MH-pFedHN and MH-pFedHNGD (More Models)
```bash  
python MH-pFedHN.py --data-path data --data-name cifar100 --num-nodes 50 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn False  
python MH-pFedHNGD.py --data-path data --data-name cifar100 --num-nodes 50 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --only-cnn False
```

## Running MH-pFedHN and MH-pFedHNGD for Testing (Only CNN Model)
```bash 
python MH-pFedHN.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn True --save-model True  
python MH-pFedHNGD.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn True --save-model True
```
## Running MH-pFedHN and MH-pFedHNGD for Testing (More Models)
```bash 
python MH-pFedHN.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn False --save-model True  
python MH-pFedHNGD.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --train-clients 80 --only-cnn False --save-model True
```

## Generalization to Novel Clients
```bash 
python run_test.py --data-path data --data-name cifar100 --num-nodes 100 --num-steps 500 --cuda 0 --optim adam --lr 2e-4 --test-clients 20 --only-cnn True --hynet-dir Output/cnn_cifar100_cn_100_tr_80_0hynet.pt --fc-dir Output/cnn_cifar100_cn_100_tr_80_0fc.pt
```