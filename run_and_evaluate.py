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
from numpy.linalg import norm
import torch
torch.hub.set_dir("/scratch/jlb638/torch_hub_cache")
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline
from PIL import Image
import ImageReward as image_reward
reward_cache="/scratch/jlb638/ImageReward"
from static_globals import *
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel
import numpy as np
from peft import LoraConfig, get_peft_model
from aesthetic_reward import get_aesthetic_scorer
from chosen_helpers import get_hidden_states,get_best_cluster_kmeans,get_init_dist,loop
import gc
from dvlab.rival.test_variation_sdv1 import make_eval_image
from instant.infer import instant_generate_one_sample

def cos_sim(vector_i,vector_j)->float:
    return np.dot(vector_i,vector_j)/(norm(vector_i)*norm(vector_j))

def evaluate_one_sample(
        method_name:str,
        src_image: Image,
        text_prompt:str,
        evaluation_prompt_list:list,
        accelerator:Accelerator,
        subject:str,
        num_inference_steps:int,
        n_img_chosen:int,
        target_cluster_size:int,
        min_cluster_size:int,
        convergence_scale:float,
        inf_config:str,
        is_half:bool,
        seed:int,
        inner_round:int,
        editing_early_steps:int
)->dict:
    method_name=method_name.strip()
    if method_name == BLIP_DIFFUSION:
        blip_diffusion_pipe=BlipDiffusionPipeline.from_pretrained(
            "Salesforce/blipdiffusion", torch_dtype=torch.float32)
        blip_diffusion_pipe=accelerator.prepare(blip_diffusion_pipe)
        evaluation_image_list=[
            blip_diffusion_pipe(
                evaluation_prompt.format(subject),
                src_image,
                subject,
                subject,
                guidance_scale=7.5,
                num_inference_steps=num_inference_steps,
                neg_prompt=NEGATIVE,
                height=512,
                width=512,
                ).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==ELITE:
        pass
    elif method_name==RIVAL:
        evaluation_image_list=[
            make_eval_image(inf_config,accelerator,is_half,
                            "runwayml/stable-diffusion-v1-5",
                            evaluation_prompt.format(subject),
                            NEGATIVE,src_image,seed,inner_round,
                            num_inference_steps,editing_early_steps) for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==INSTANT:
        evaluation_image_list=[
            instant_generate_one_sample(src_image,evaluation_prompt.format(subject),
                                        NEGATIVE, num_inference_steps, 
                                        accelerator ) for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==IP_ADAPTER or method_name==FACE_IP_ADAPTER:
        pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None)
        unet=pipeline.unet
        vae=pipeline.vae
        tokenizer=pipeline.tokenizer
        text_encoder=pipeline.text_encoder
        pipeline("none",num_inference_steps=1) #things initialize weird if we dont do it once
        if method_name==IP_ADAPTER:
            pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        elif method_name==FACE_IP_ADAPTER:
            pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
        image_encoder=pipeline.image_encoder
        unet,text_encoder,vae,tokenizer,image_encoder = accelerator.prepare(
            unet,text_encoder,vae,tokenizer,image_encoder
        )
        for model in [vae,unet,text_encoder, image_encoder]:
            model.requires_grad_(False)
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(subject),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None,
                    ip_adapter_image=src_image).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    elif method_name==CHOSEN:
        vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        vit_model = ViTModel.from_pretrained('facebook/dino-vitb16')
        pipeline=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker=None)
        unet=pipeline.unet
        vae=pipeline.vae
        tokenizer=pipeline.tokenizer
        text_encoder=pipeline.text_encoder
        for model in [vae,unet,text_encoder]:
            model.requires_grad_(False)
        config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none")
        unet = get_peft_model(unet, config)
        unet.train()
        unet.print_trainable_parameters()
        trainable_parameters=[]
        for model in [vae,unet,text_encoder]:
            trainable_parameters+=[p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=0.00000001)
        unet,text_encoder,vae,tokenizer = accelerator.prepare(
            unet,text_encoder,vae,tokenizer
        )
        n_clusters=n_img_chosen // target_cluster_size
        image_list=[
                pipeline(text_prompt,negative_prompt=NEGATIVE,num_inference_steps=num_inference_steps,safety_checker=None).images[0] for _ in range(n_img_chosen)]
        print("generated initial sets of images")
        last_hidden_states=get_hidden_states(image_list,vit_processor,vit_model)
        print("last hidden staes")
        init_dist=get_init_dist(last_hidden_states)
        print("init_dist")
        pairwise_distances=init_dist
        iteration=0
        while pairwise_distances>=convergence_scale*init_dist and iteration<10:
            iteration+=1
            valid_image_list, pairwise_distances=get_best_cluster_kmeans(image_list, n_clusters, min_cluster_size, vit_processor, vit_model)
            text_prompt_list=[text_prompt]*len(valid_image_list)
            pipeline=loop(
                valid_image_list,
                text_prompt_list,
                pipeline,
                0,
                optimizer,
                accelerator,
                1,
                num_inference_steps,
                size=512,
                train_batch_size=2,
                noise_offset=0.0,
                max_grad_norm=1.0
            )
            image_list=[pipeline(text_prompt,num_inference_steps=num_inference_steps,safety_checker=None).images[0] for _ in range(n_img_chosen) ]
        evaluation_image_list=[
            pipeline(evaluation_prompt.format(text_prompt),
                    num_inference_steps=num_inference_steps,
                    negative_prompt=NEGATIVE,
                    safety_checker=None).images[0] for evaluation_prompt in evaluation_prompt_list
        ]
    else:
        message=f"no support for {method_name} try one of "+" ".join(METHOD_LIST)
        raise Exception(message)

    print(evaluation_image_list)
    #METRIC_LIST=[PROMPT_SIMILARITY, IDENTITY_CONSISTENCY, TARGET_SIMILARITY, AESTHETIC_SCORE, IMAGE_REWARD]
    metric_dict={}
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_inputs=clip_processor(text=evaluation_prompt_list, images=evaluation_image_list+[src_image], return_tensors="pt", padding=True)

    outputs = clip_model(**clip_inputs)
    text_embed_list=outputs.text_embeds.detach().numpy()
    image_embed_list=outputs.image_embeds.detach().numpy()[:-1]
    src_image_embed=outputs.image_embeds.detach().numpy()[-1]
    ir_model=image_reward.load("ImageReward-v1.0",download_root=reward_cache)

    identity_consistency_list=[]
    target_similarity_list=[]
    prompt_similarity_list=[]
    for i in range(len(image_embed_list)):
        image_embed=image_embed_list[i]
        text_embed=text_embed_list[i]
        target_similarity_list.append(cos_sim(image_embed,src_image_embed))
        prompt_similarity_list.append(cos_sim(image_embed, text_embed))
        for j in range(i+1, len(image_embed_list)):
            vector_j=image_embed_list[j]
            identity_consistency_list.append(cos_sim(image_embed,vector_j))


    metric_dict[IDENTITY_CONSISTENCY]=np.mean(identity_consistency_list)
    metric_dict[TARGET_SIMILARITY]=np.mean(target_similarity_list)
    metric_dict[PROMPT_SIMILARITY]=np.mean(prompt_similarity_list)
    #for evaluation_image,evaluation_prompt in zip(evaluation_image_list, evaluation_prompt_list):
    metric_dict[IMAGE_REWARD]=np.mean(
        [ir_model.score(evaluation_prompt.format(subject),evaluation_image) for evaluation_prompt,evaluation_image in zip(evaluation_prompt_list, evaluation_image_list) ]
    )
    aesthetic_scorer=get_aesthetic_scorer()
    metric_dict[AESTHETIC_SCORE]=np.mean(
        [aesthetic_scorer(evaluation_image).cpu().numpy()[0] for evaluation_image in evaluation_image_list]
    )
    for metric in METRIC_LIST:
        if metric not in metric_dict:
            metric_dict[metric]=0.0
    accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()
    unet=vae=tokenizer=text_encoder=image_encoder=blip_diffusion_pipe=pipeline=clip_model=optimizer=None
    return metric_dict,evaluation_image_list