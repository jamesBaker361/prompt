import sys
sys.path += ['.']
import sys
sys.path.append("/home/jlb638/Desktop/prompt")
import os
cache_dir="/scratch/jlb638/trans_cache"
os.environ["TRANSFORMERS_CACHE"]=cache_dir
os.environ["HF_HOME"]=cache_dir
os.environ["HF_HUB_CACHE"]=cache_dir
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from PIL import Image
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"
from static_globals import *

def evaluate_one_sample(
        method_name:str,
        src_image: Image,
        text_prompt:str,
        evaluation_prompt_list:list
)->dict:
    if method_name == BLIP_DIFFUSION:
        pass
    elif method_name==ELITE:
        pass
    elif method_name==RIVAL:
        pass
    elif method_name==IP_ADAPTER:
        pass
    elif method_name==CHOSEN:
        pass
    else:
        raise Exception(f"no support for {method_name}")
    return {}