"""
train_ockham.py - Training with Ockham's Razor principles

This script demonstrates how to train a nanoGPT model using the OckhamLearner
for intelligent, minimal adaptation during test-time training.

Usage:
    # Standard training with Ockham regularization
    python train_ockham.py --config=config/train_shakespeare_char.py
    
    # Training with surprise gate (only update when necessary)
    python train_ockham.py --config=config/train_shakespeare_char.py --surprise_threshold=2.0
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from ockham_learner import OckhamLearner
from ockham_memory import OckhamMemory

# -----------------------------------------------------------------------------
# default config values designed to train a small GPT on Shakespeare
# I/O
out_dir = 'out-ockham'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False
wandb_project = 'ockham-gpt'
wandb_run_name = 'ockham-run'
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
# model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = False
# adamw optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 0.0  # Set to 0 to avoid double regularization with Ockham
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
# Ockham-specific parameters
use_ockham = True  # Enable Ockham regularization
lambda_ockham = 0.01  # Strength of Ockham penalty
surprise_threshold = None  # Minimum loss to trigger update (None = always update)
consolidate_interval = 1000  # How often to consolidate the anchor (None = never)
use_ockham_memory = False  # Enable OckhamMemory for model selection
# DDP settings
backend = 'nccl'
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Set to False for easier debugging
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# model init
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap with DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Initialize OckhamLearner if enabled
ockham_learner = None
if use_ockham and master_process:
    print(f"\n{'='*80}")
    print("OCKHAM'S RAZOR MODE ENABLED")
    print(f"{'='*80}")
    print(f"  Lambda (regularization strength): {lambda_ockham}")
    print(f"  Surprise threshold: {surprise_threshold if surprise_threshold else 'None (always update)'}")
    print(f"  Consolidation interval: {consolidate_interval if consolidate_interval else 'None (never consolidate)'}")
    print(f"{'='*80}\n")
    
    raw_model = model.module if ddp else model
    ockham_learner = OckhamLearner(
        model=raw_model,
        optimizer=optimizer,
        lambda_ockham=lambda_ockham,
        surprise_threshold=surprise_threshold,
        device=device
    )

# Initialize OckhamMemory if enabled
ockham_memory = None
if use_ockham_memory and master_process:
    ockham_memory = OckhamMemory(memory_dir=os.path.join(out_dir, 'ockham_memory'))
    print(f"[OckhamMemory] Initialized at {ockham_memory.memory_dir}")

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define loss function for OckhamLearner
def loss_fn(logits, targets):
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        
        # Save to OckhamMemory if enabled
        if use_ockham_memory and losses['val'] < best_val_loss:
            model_id = f"iter_{iter_num:06d}"
            ockham_memory.add_candidate(
                model=raw_model,
                val_loss=losses['val'].item(),
                train_loss=losses['train'].item(),
                config=model_args,
                model_id=model_id
            )
            print(ockham_memory.get_frontier_summary())
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    if use_ockham and ockham_learner is not None:
        # OckhamLearner handles everything internally
        X, Y = get_batch('train')
        with ctx:
            logits = model(X)
        step_metrics = ockham_learner.adapt(X, Y, loss_fn)
        loss = torch.tensor(step_metrics['total_loss'])  # For logging
    else:
        # Standard training with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
    
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps if not use_ockham else loss
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        if use_ockham and ockham_learner is not None:
            metrics = ockham_learner.get_metrics()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, "
                  f"updates {metrics['updates_performed']}/{metrics['total_batches']} "
                  f"({metrics['update_rate']*100:.1f}%), "
                  f"avg_complexity {metrics['avg_complexity_cost']:.4f}")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    # Consolidate anchor if interval is set
    if use_ockham and ockham_learner is not None and consolidate_interval is not None:
        if iter_num > 0 and iter_num % consolidate_interval == 0:
            ockham_learner.consolidate()
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if master_process and use_ockham and ockham_learner is not None:
    print("\n" + "="*80)
    print("FINAL OCKHAM METRICS")
    print("="*80)
    metrics = ockham_learner.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("="*80)
