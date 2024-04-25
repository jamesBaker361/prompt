import sys
sys.path += ['.']
import sys
sys.path.append("/home/jlb638/Desktop/prompt")
import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
os.environ["WANDB_DIR"]="/scratch/jlb638/wandb"
os.environ["WANDB_CACHE_DIR"]="/scratch/jlb638/wandb_cache"
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from diffusers.pipelines import BlipDiffusionPipeline
from PIL import Image
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"
from static_globals import *
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from aesthetic_reward import get_aesthetic_scorer
import argparse
from run_and_evaluate import evaluate_one_sample
from datasets import load_dataset
import wandb
from gpu import print_details
import gc

parser=argparse.ArgumentParser()

parser.add_argument("--limit",type=int,default=50)
parser.add_argument("--method_name",type=str,default=BLIP_DIFFUSION)
parser.add_argument("--src_dataset",type=str,default="jlbaker361/league-hard-prompt")
parser.add_argument("--num_inference_steps",type=int,default=30)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/prompt-images")
parser.add_argument("--convergence_scale",type=float,default=0.75)
parser.add_argument("--n_img_chosen",type=int,default=64)
parser.add_argument("--target_cluster_size",type=int,default=10)
parser.add_argument("--min_cluster_size",type=int,default=5)
parser.add_argument("--inf_config", type=str, default="dvlab/configs/rival_variation.json")
parser.add_argument("--inner_round", type=int, default=1, help="number of images per reference")
parser.add_argument("--is_half", type=bool, default=False)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--editing_early_steps", type=int, default=1000)
parser.add_argument("--train_text_encoder",action="store_true",help="ddpo train text encoder")
parser.add_argument("--train_text_encoder_embeddings",action="store_true",help="ddpo train text encoder like for text inversion")
parser.add_argument("--train_unet",action="store_true",help="ddpo whether to train unet")
parser.add_argument("--use_lora_text_encoder",action="store_true",help="ddpo use lora if train_text_encoder=True")
parser.add_argument("--use_lora",action="store_true",help="unet ddpo use lora")

def main(args):
    accelerator=Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="one_shot")
    dataset=load_dataset(args.src_dataset,split="train")
    print('dataset.column_names',dataset.column_names)
    aggregate_dict={
        metric:[] for metric in METRIC_LIST
    }
    evaluation_prompt_list=[
        "  {} at the beach",
        "  {} in the jungle",
        "  {} in the snow",
        "  {} in the street",
        "  {} with a city in the background",
        "  {} with a mountain in the background",
        "  {} with the Eiffel Tower in the background",
        "  {} near the Statue of Liberty",
        "  {} near the Sydney Opera House",
        "  {} floating on top of water",
        "  {} eating a burger",
        "  {} drinking a beer",
        "  {} wearing a blue hat",
        "  {} wearing sunglasses",
        "  {} playing with a ball",
        "  {} as a police officer"
    ]
    len_dataset=len([r for r in dataset])
    print("len",len_dataset)
    for j,row in enumerate(dataset):
        gc.collect()
        accelerator.free_memory()
        torch.cuda.empty_cache()
        if j>args.limit:
            print("reached limit")
            break
        subject=row["subject"]
        label=row["label"]
        src_image=row["splash"]
        text_prompt=row["optimal_prompt"]
        metric_dict,evaluation_image_list=evaluate_one_sample(args.method_name,
                                                              src_image,
                                                              text_prompt,
                                                              evaluation_prompt_list,
                                                              accelerator,subject,
                                                              args.num_inference_steps,
                                                              args.n_img_chosen,
                                                              args.target_cluster_size,
                                                              args.min_cluster_size,
                                                              args.convergence_scale,
                                                              args.inf_config,
                                                                args.is_half,
                                                                args.seed,
                                                                args.inner_round,
                                                                args.editing_early_steps)
        os.makedirs(f"{args.image_dir}/{label}/",exist_ok=True)
        for i,image in enumerate(evaluation_image_list):
            path=f"{args.image_dir}/{label}/{args.method_name}_{i}.png"
            image.save(path)
            accelerator.log({
                f"{label}/{args.method_name}_{i}":wandb.Image(path)
            })
        for metric,value in metric_dict.items():
            aggregate_dict[metric].append(value)
        print(f"after {j} samples:")
        for metric,value_list in aggregate_dict.items():
            print(f"\t{metric} {np.mean(value_list)}")
    columns=METRIC_LIST
    data=np.transpose([v for v in aggregate_dict.values()])
    accelerator.get_tracker("wandb").log({
        "result_table":wandb.Table(columns=columns,data=data)
    })
if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!")