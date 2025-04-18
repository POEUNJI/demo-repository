out_dir = 'out-eunji'
eval_interval = 10
log_interval = 1

always_save_checkpoint = True

dataset = '은지말'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 10
lr_decay_iters = 500
dropout = 0.1

learning_rate = 1e-3
device = 'cpu'  # 은지는 지금 IPU 세션이니까 CPU로 지정해
compile = False
eval_iters = 20


