import os
import time
import random

gpus = [1,7,8,9,2,3,4,5,6]*5
idx = 0
finetune = 20000
epoch = 1000
tree = 'depth2_sub'
batch_size = 10
lr_schedule = 'cos'
for i in range(1):
    for dim in [6]:
        gpu = random.randint(0,4)
        os.system('python controller_coecentric.py --lr_schedule '+lr_schedule+' --epoch '+str(epoch)+' --finetune '+str(finetune)+' --bs '+str(batch_size)+' --greedy 0.1 --gpu '+str(gpu)+' --ckpt 01_26_2024_coecentric_'+str(dim)+'_tree_'+tree+' --tree '+tree+' --random_step 3 --lr 0.002 --input_dim '+str(dim)+' --base 20000 --N 30000 --N_bound 1000')
        time.sleep(500)
