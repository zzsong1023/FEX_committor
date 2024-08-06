import os
import time
import random

gpus = [1,7,8,9,2,3,4,5,6]*5
idx = 0

T = 0.05
rho = 0.5
epoch = 1000
finetune = 20000
for i in range(1):
    for dim in [10]:
        gpu = random.randint(0,4)
        os.system('python controller_pwell.py --T '+str(T)+' --rho '+str(rho)+' --finetune '+str(finetune)+' --epoch '+str(epoch)+' --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt 01_26_2024_pwell'+str(dim)+'_T_'+str(T)+'_depth2sub --tree depth2_sub --random_step 3 --lr 0.002 --input_dim '+str(dim)+' --base 20000 --N 20000 --N_bound 1000')
        time.sleep(500)
