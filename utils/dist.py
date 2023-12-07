'''
distributed training
ref: https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/utils/distributed.py
NOTE: dist is not implemented and tested in current codebase
'''

import torch
import torch.nn as nn
import subprocess
import sys
import os

import ipdb


def is_distributed():
    return torch.distributed.is_initialized()

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def all_reduce_numpy(array):
    tensor = torch.from_numpy(array).cuda()
    torch.distributed.all_reduce(tensor) # torch.distributed.all_reduce: reduces the tensor data across all machines in such a way that all get the final result.
    return tensor.cpu().numpy()

def handle_distributed(args, main_file, logger=print):
    if not args.distributed:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))   
        return

    if args.local_rank >= 0:
        _setup_process_group(args)
        return

    current_env = os.environ.copy()
    if current_env.get('CUDA_VISIBLE_DEVICES') is None:
        current_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
        world_size = len(args.gpu)
    else:
        world_size = len(current_env['CUDA_VISIBLE_DEVICES'].split(','))

    current_env['WORLD_SIZE'] = str(world_size)

    print('World size:', world_size)
    # Logic for spawner
    python_exec = sys.executable
    command_args = sys.argv
    logger('{}'.format(command_args))

    command_args = command_args[1:]
    # print(command_args)
    command_args = [
        python_exec, '-u',
        '-m', 'torch.distributed.launch',
        '--nproc_per_node', str(world_size),
        '--master_port', str(29961),
        main_file,
    ] + command_args
    process = subprocess.Popen(command_args, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=command_args)    
    sys.exit(process.returncode)

def _setup_process_group(args):
    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        # rank=local_rank
    )