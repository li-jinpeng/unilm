import schedule
import time
import os
 
def run():
    os.system('conda activate layoutlmft')
    os.system('python -m torch.distributed.launch --nproc_per_node=8 /home/lijinpeng/unilm/layoutlmft/examples/run_xfun_ser.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16')
 
schedule.every(45).minutes.do(run)    # 每隔45分钟执行一次任务
 
while True:
    schedule.run_pending()  # run_pending：运行所有可以运行的任务