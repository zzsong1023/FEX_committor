import os
import time
import random

gpus = [1,7,8,9,2,3,4,5,6]*5
idx = 0


rho = 200
epoch = 200
finetune = 20000
for i in range(1):
    for dim in [3]:
        gpu = 0
        os.system('python controller_butane.py --rho '+str(rho)+' --finetune '+str(finetune)+' --epoch '+str(epoch)+' --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt 05_17_2024_pwell'+str(dim)+'_T_300K_depth2sub --tree depth2_sub --random_step 3 --lr 0.002 --input_dim '+str(dim)+' --base 20000')
        time.sleep(5)
