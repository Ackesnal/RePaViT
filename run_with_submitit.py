import argparse
import os
from pathlib import Path

import main
import submitit
import json

def parse_args():
    parser = argparse.ArgumentParser("Submitit for RePaViT", parents=[main.get_args_parser()])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--ncpus", default=20, type=int, help="Number of cpus per to request in each task")
    parser.add_argument("--nodes", default=8, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=120, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="h2gpu", type=str, help="Partition where to submit")
    parser.add_argument("--master_port", default="12345", type=str, help="Master port for the distributed training")
    parser.add_argument("--additional_slurm_params", type=json.loads, help="Additional slurm parameters in JSON format")
    return parser.parse_args()


def get_shared_folder() -> Path:
    working_dir = os.path.abspath(__file__)
    print(os.path.join(working_dir, "output/"))
    raise RuntimeError("No shared folder available")

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main

        self._setup_gpu_args()
        main.main(self.args)

    def checkpoint(self):
        import os
        import submitit
        from pathlib import Path
        import torch
        import glob
        import re

        output_dir = Path(self.args.output_dir)
        checkpoint = None
        current_checkpoint_path = None
    
        # Define the checkpoint filenames with the prefix path
        checkpoint_path = output_dir / 'checkpoint.pth'
        best_checkpoint_path = output_dir / 'best_checkpoint.pth'

        # Try to load 'checkpoint.pth'
        try:
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded '{checkpoint_path}'")
            current_checkpoint_path = checkpoint_path
        except Exception as e1:
            print(f"Failed to load '{checkpoint_path}': {e1}")
            
            # Try to load 'best_checkpoint.pth'
            try:
                checkpoint = torch.load(best_checkpoint_path)
                print(f"Loaded '{best_checkpoint_path}'")
                current_checkpoint_path = best_checkpoint_path
            except Exception as e2:
                print(f"Failed to load '{best_checkpoint_path}': {e2}")
                
                # Find all 'checkpoint{number}epoch.pth' files in the output directory
                # Specifically search for files with numbers in their filenames
                pattern = str(output_dir / 'checkpoint*epoch.pth')
                files = glob.glob(pattern)
                files_with_numbers = []
                
                # Extract numbers from filenames and store them with filenames
                for f in files:
                    filename = Path(f).name  # Extract the filename from the full path
                    match = re.search(r'checkpoint(\d+)epoch\.pth', filename)
                    if match:
                        epoch_num = int(match.group(1))
                        files_with_numbers.append((epoch_num, f))
                
                # Sort files by epoch number in descending order
                files_with_numbers.sort(reverse=True)
                
                # Try loading the checkpoints starting from the largest number
                for epoch_num, f in files_with_numbers:
                    try:
                        checkpoint = torch.load(f)
                        current_checkpoint_path = f
                        print(f"Loaded '{f}'")
                    except Exception as e3:
                        print(f"Failed to load '{f}': {e3}")
        
        if checkpoint is None:
            print("No checkpoint found, unable to resume training.")
            return None
        else:
            if 'epoch' in checkpoint and checkpoint['epoch'] >= 299:
                print("Training completed, no need to resubmit.")
                return None
            else:
                print(f"Checkpoint exists at {str(current_checkpoint_path)}, resuming training...")
                self.args.resume = str(current_checkpoint_path)
                return submitit.helpers.DelayedSubmission(type(self)(self.args))
            

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        
        node_list = job_env.hostnames
        num_nodes = len(node_list)
        node_0 = node_list[0]
        os.environ['MASTER_ADDR'] = node_0
        os.environ['MASTER_PORT'] = self.args.master_port
        
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

def run():
    args = parse_args()
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    num_cpus_per_task = args.ncpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}



    executor.update_parameters(
        mem_gb=50 * num_gpus_per_node,
        #gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=num_cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_additional_parameters={"gres": f"gpu:{num_gpus_per_node}", "account": "OD-221915"},
        **kwargs
    )

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    run()
