import argparse
import torch
import torch.nn as nn
import torchvision
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import argparse
import torch.utils.checkpoint as checkpoint
import os, shutil
from PIL import Image
import time
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pickle
import pandas as pd
import json
import random

CLIP_MODEL_PATH = 'openai/clip-vit-large-patch14'


# Q16
with open('Q16_prompts.p', 'rb') as file:
    Q16_OPT_PROMPT = np.float32(pickle.load(file))

def Q16_loss_fn(inference_dtype = None, device = None):
    # model_name = "ViT-L/14"
    model_name = "ViT-B/32"
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, 'openai', jit=False)
    model = model.to(device, dtype=inference_dtype)
    model.eval()

    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
    target_size = 224

    def loss_fn(im_pix_un, prompts=None):

            im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
            im_pix = torchvision.transforms.Resize(target_size)(im_pix)
            im_pix = normalize(im_pix).to(inference_dtype)
            image_features = model.encode_image(im_pix)
            text_features = torch.tensor(Q16_OPT_PROMPT).to(device='cuda',dtype=inference_dtype)
            image_features_norm = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)
            text_features_norm = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)
            logits = (100.0 * image_features_norm @ text_features_norm.T)
            probs = logits[0].softmax(dim=-1)
            toxic_score = -5 * probs[1] + 5
            loss = -toxic_score
            return loss

    return loss_fn

# CLIP Score
def clip_score(inference_dtype=None, device=None):
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    model = model.to(device = device, dtype=inference_dtype)
    
    @torch.no_grad()
    def loss_fn(image, prompt):    
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image 
        score = logits_per_image.cpu().numpy()[0][0]
        
        return  score
    
    return loss_fn

# sampling algorithm
class SequentialDDIM:

    def __init__(self, timesteps = 100, scheduler = None, eta = 0.0, cfg_scale = 4.0, device = "cuda", opt_timesteps = 50):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.opt_timesteps = opt_timesteps 

        # compute some coefficients in advance
        scheduler_timesteps = self.scheduler.timesteps.tolist()
        scheduler_prev_timesteps = scheduler_timesteps[1:]
        scheduler_prev_timesteps.append(0)
        self.scheduler_timesteps = scheduler_timesteps[::-1]
        scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
        alphas_cumprod = [1 - self.scheduler.alphas_cumprod[t] for t in self.scheduler_timesteps]
        alphas_cumprod_prev = [1 - self.scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]

        now_coeff = torch.tensor(alphas_cumprod)
        next_coeff = torch.tensor(alphas_cumprod_prev)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, prompt_embeds = None):

        t_ind = self.num_steps - len(self._samples)
        t = self.scheduler_timesteps[t_ind]
   
        model_kwargs = {
            "sample": torch.stack([self._samples[0], self._samples[0]]),
            "timestep": torch.tensor([t, t], device = self.device),
            "encoder_hidden_states": prompt_embeds
        }

        model_kwargs["sample"] = self.scheduler.scale_model_input(model_kwargs["sample"],t)

        return model_kwargs


    def step(self, model_output):
        model_output_uncond, model_output_text = model_output[0].chunk(2)
        direction = model_output_uncond + self.cfg_scale * (model_output_text - model_output_uncond)
        direction = direction[0]

        t = self.num_steps - len(self._samples)

        if t <= self.opt_timesteps:
            now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]
        else:
            with torch.no_grad():
                now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        if self.num_steps == self.opt_timesteps:
            self._samples = [self.noise_vectors[-1]]
        else:
            self._samples = [self.noise_vectors[-1].detach()]

def sequential_sampling(pipeline, unet, sampler, prompt_embeds, noise_vectors): 


    sampler.initialize(noise_vectors)

    model_time = 0
    while not sampler.is_finished():
        model_kwargs = sampler.prepare_model_kwargs(prompt_embeds = prompt_embeds)
        #model_output = pipeline.unet(**model_kwargs)
        model_output = checkpoint.checkpoint(unet, model_kwargs["sample"], model_kwargs["timestep"], model_kwargs["encoder_hidden_states"],  use_reentrant=False)
        sampler.step(model_output) 

    return sampler.get_last_sample()


def decode_latent(decoder, latent):
    img = decoder.decode(latent.unsqueeze(0) / 0.18215).sample
    return img

def to_img(img):
    img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="sd1.5", help="base diffusion model")
    parser.add_argument("--dataset", type = str, default="i2p_benchmark_harassment_hardest", help="dataset for evaluation")
    parser.add_argument("--objective", type=str, default="Q16", help="target safety objective")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--num_steps", type=int, default=50, help="number of DDIM steps")
    parser.add_argument("--opt_steps", type=int, default=5, help="max number of optimization steps")
    parser.add_argument("--eta", type=float, default=0.0, help="noise scale")
    parser.add_argument("--guidance_scale", type=float, default=10, help="guidance scale")
    parser.add_argument("--output_path", type=str, default="./output_folder", help="output path")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gamma', type=float, default=1.0, help='mean penalty')
    parser.add_argument('--no_reg', action='store_true', help='regularization of noise trajectory')
    parser.add_argument('--lr', type=float, default=0.07, help='stepsize for optimization')
    parser.add_argument('--subsample', type=int, default=1, help='subsample factor for regularization')
    parser.add_argument('--opt_time', type=int, default=50, help="number of DDIM noise to optimize")
    parser.add_argument('--opt_variable', type=str, default="both", help='optimization variable')
    parser.add_argument('--threshold', type=float, default=2.5, help='threshold for stopping')

    args = parser.parse_args()
    
    if args.model == "sd1.5":
        Diff_Model_Path = 'benjamin-paine/stable-diffusion-v1-5' # using a mirror of the model repo
    else:
        raise ValueError("Model not implemented.")

    pipeline = StableDiffusionPipeline.from_pretrained(Diff_Model_Path).to(args.device)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline.scheduler.set_timesteps(args.num_steps)
    unet = pipeline.unet

    if args.objective == "Q16":
        loss_fn = Q16_loss_fn(inference_dtype = torch.float16, device = args.device)
    else:
        raise ValueError("Objective not implemented.")
    torch.manual_seed(args.seed)
    
    output_path = os.path.join(args.output_path, f"model_{args.model}_dataset_{args.dataset}_objective_{args.objective}_variable_{args.opt_variable}_lr_{args.lr}_no_reg_{args.no_reg}_steps_{args.opt_steps}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    prompts = pd.read_csv(f"./datasets/{args.dataset}.csv")["prompt"]
    prompts = prompts.tolist()


    best_rewards = []
    best_Q16s = []
    best_CLIPs = []

    before_Q16s = []
    before_CLIPs = []
    end_iterations = []
    for prompt in prompts:
        output_path_prompt = os.path.join(output_path, prompt[:10])
        if os.path.exists(output_path_prompt):
            shutil.rmtree(output_path_prompt)
        os.makedirs(output_path_prompt)        
        print(f"Optimizing for prompt: {prompt}")

        print("Searching for initial noise vector")
        best_reward = -1e9
        best_sample = None
        best_noise_vectors = None
        prompt_embeds = pipeline._encode_prompt(
                            prompt,
                            args.device,
                            1,
                            True
                        )
        # Random search for initialization
        for i in range(5):
            noise_vectors = torch.randn(args.num_steps + 1, 4, 64, 64, device = args.device)
            with autocast(device_type="cuda"):
                ddim_sampler = SequentialDDIM(timesteps = args.num_steps,
                                                scheduler = pipeline.scheduler, 
                                                eta = args.eta, 
                                                cfg_scale = args.guidance_scale, 
                                                device = args.device,
                                                opt_timesteps = args.opt_time)

                sample = sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, noise_vectors = noise_vectors)
                sample = decode_latent(pipeline.vae, sample)

                losses = loss_fn(sample, [prompt] * sample.shape[0])
                loss = losses.mean()

                reward = -loss.item()
                print(f"random search iter {i}: reward:{reward}")
                if reward > best_reward:
                    best_reward = reward
                    best_sample = sample.detach()
                    best_noise_vectors = noise_vectors
                if i == 0:
                    img = to_img(best_sample)
                    img = Image.fromarray(img)
                    with torch.no_grad():

                        Q16_Score = -Q16_loss_fn(inference_dtype = torch.float16, device = args.device)(best_sample, [prompt] * best_sample.shape[0]).item()
                        CLIP_Score = clip_score(inference_dtype = torch.float16, device = args.device)(img, prompt)
                        img.save(f"{output_path_prompt}/sample_{prompt[:10]}_{i}_Q16_{Q16_Score}_CLIP_{CLIP_Score}.png")
                        before_Q16s.append(Q16_Score)
                        before_CLIPs.append(CLIP_Score)

                if best_reward > args.threshold:
                    img = to_img(best_sample)
                    img = Image.fromarray(img)
                    with torch.no_grad():
                        Q16_Score = -Q16_loss_fn(inference_dtype = torch.float16, device = args.device)(best_sample, [prompt] * best_sample.shape[0]).item()
                        CLIP_Score = clip_score(inference_dtype = torch.float16, device = args.device)(img, prompt)
                        img.save(f"{output_path_prompt}/sample_{prompt[:10]}_best_Q16_{Q16_Score}_CLIP_{CLIP_Score}.png")
                    best_Q16s.append(Q16_Score)
                    best_CLIPs.append(CLIP_Score)
                    best_rewards.append(best_reward)
                    end_iterations.append(i+1)

                    break

        if best_reward > args.threshold:
            continue

        noise_vectors = best_noise_vectors.detach().clone().to('cuda')

        # begin optimization
        if args.opt_variable == "both":
            prompt_embeds.requires_grad_(True)
            noise_vectors.requires_grad_(True)
            params_to_opt = [noise_vectors, prompt_embeds]
        elif args.opt_variable == "noise":
            noise_vectors.requires_grad_(True)
            params_to_opt = [noise_vectors]
        elif args.opt_variable == "prompt":
            prompt_embeds.requires_grad_(True)
            params_to_opt = [prompt_embeds]
        else:
            raise ValueError("Invalid optimization variable")

        optimizer = torch.optim.AdamW(params_to_opt, lr = args.lr)

        if args.eta > 0:
            dim = len(noise_vectors[:(args.opt_time + 1)].flatten())
        else:
            dim = len(noise_vectors[-1].flatten())

        subsample_dim = 4 ** args.subsample
        subsample_num = dim // subsample_dim

        noise_traj_list = []
        prompt_embeds_list = []
        Q16_scores = []
        CLIP_scores = []
        for i in range(args.opt_steps):
            noise_traj_list.append(noise_vectors.detach().cpu().numpy())
            prompt_embeds_list.append(prompt_embeds.detach().cpu().numpy())
            optimizer.zero_grad()
            start_time = time.time()
            with autocast(device_type="cuda"): 
                ddim_sampler = SequentialDDIM(timesteps = args.num_steps,
                                                scheduler = pipeline.scheduler, 
                                                eta = args.eta, 
                                                cfg_scale = args.guidance_scale, 
                                                device = args.device,
                                                opt_timesteps = args.opt_time)

                sample = sequential_sampling(pipeline, unet, ddim_sampler, prompt_embeds = prompt_embeds, noise_vectors = noise_vectors)
                sample = decode_latent(pipeline.vae, sample)

                losses = loss_fn(sample, [prompt] * sample.shape[0])
                loss = losses.mean()

                reward = -loss.item()
                if reward > best_reward:
                    best_reward = reward
                    best_sample = sample.detach()
                if best_reward > args.threshold:
                    end_time = time.time()
                    print("time", end_time - start_time)
                    best_rewards.append(best_reward)
                    img = to_img(best_sample)
                    img = Image.fromarray(img)
                    with torch.no_grad():
                        Q16_Score = -Q16_loss_fn(inference_dtype = torch.float16, device = args.device)(best_sample, [prompt] * best_sample.shape[0]).item()
                        CLIP_Score = clip_score(inference_dtype = torch.float16, device = args.device)(img, prompt)
                        img.save(f"{output_path_prompt}/sample_{prompt[:10]}_best_Q16_{Q16_Score}_CLIP_{CLIP_Score}.png")
                    best_Q16s.append(Q16_Score)
                    best_CLIPs.append(CLIP_Score)
                    end_iterations.append(i+6)
                    break
                
                # Calculate the regularization term:
                # squential subsampling
                if args.eta > 0:
                    noise_vectors_flat = noise_vectors[:(args.num_steps + 1)].flatten()
                else:
                    noise_vectors_flat = noise_vectors[-1].flatten()
                    
                noise_vectors_seq = noise_vectors_flat.view(subsample_num, subsample_dim)

                seq_mean = noise_vectors_seq.mean(dim = 0)
                noise_vectors_seq = noise_vectors_seq / np.sqrt(subsample_num)
                seq_cov = noise_vectors_seq.T @ noise_vectors_seq
                seq_var = seq_cov.diag()
                
                # compute the probability of the noise
                seq_mean_M = torch.norm(seq_mean)
                seq_cov_M = torch.linalg.matrix_norm(seq_cov - torch.eye(subsample_dim, device = seq_cov.device), ord = 2)
                
                seq_mean_log_prob = - (subsample_num * seq_mean_M ** 2) / 2 / subsample_dim
                seq_mean_log_prob = torch.clamp(seq_mean_log_prob, max = - np.log(2))
                seq_mean_prob = 2 * torch.exp(seq_mean_log_prob)
                seq_cov_diff = torch.clamp(torch.sqrt(1+seq_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
                seq_cov_log_prob = - subsample_num * (seq_cov_diff ** 2) / 2 
                seq_cov_log_prob = torch.clamp(seq_cov_log_prob, max = - np.log(2))
                seq_cov_prob = 2 * torch.exp(seq_cov_log_prob)


                shuffled_times = 100

                shuffled_mean_prob_list = []
                shuffled_cov_prob_list = [] 
                
                shuffled_mean_log_prob_list = []
                shuffled_cov_log_prob_list = [] 
                
                shuffled_mean_M_list = []
                shuffled_cov_M_list = []

                for _ in range(shuffled_times):
                    noise_vectors_flat_shuffled = noise_vectors_flat[torch.randperm(dim)]   
                    noise_vectors_shuffled = noise_vectors_flat_shuffled.view(subsample_num, subsample_dim)
                    
                    shuffled_mean = noise_vectors_shuffled.mean(dim = 0)
                    noise_vectors_shuffled = noise_vectors_shuffled / np.sqrt(subsample_num)
                    shuffled_cov = noise_vectors_shuffled.T @ noise_vectors_shuffled
                    shuffled_var = shuffled_cov.diag()
                    
                    # compute the probability of the noise
                    shuffled_mean_M = torch.norm(shuffled_mean)
                    shuffled_cov_M = torch.linalg.matrix_norm(shuffled_cov - torch.eye(subsample_dim, device = shuffled_cov.device), ord = 2)
                    

                    shuffled_mean_log_prob = - (subsample_num * shuffled_mean_M ** 2) / 2 / subsample_dim
                    shuffled_mean_log_prob = torch.clamp(shuffled_mean_log_prob, max = - np.log(2))
                    shuffled_mean_prob = 2 * torch.exp(shuffled_mean_log_prob)
                    shuffled_cov_diff = torch.clamp(torch.sqrt(1+shuffled_cov_M) - 1 - np.sqrt(subsample_dim/subsample_num), min = 0)
                
                    shuffled_cov_log_prob = - subsample_num * (shuffled_cov_diff ** 2) / 2
                    shuffled_cov_log_prob = torch.clamp(shuffled_cov_log_prob, max = - np.log(2))
                    shuffled_cov_prob = 2 * torch.exp(shuffled_cov_log_prob) 
                    
                    
                    shuffled_mean_prob_list.append(shuffled_mean_prob.item())
                    shuffled_cov_prob_list.append(shuffled_cov_prob.item())
                    
                    shuffled_mean_log_prob_list.append(shuffled_mean_log_prob)
                    shuffled_cov_log_prob_list.append(shuffled_cov_log_prob)
                    
                    shuffled_mean_M_list.append(shuffled_mean_M.item())
                    shuffled_cov_M_list.append(shuffled_cov_M.item())
                    
                
                print("="*10, i, "="*10)
                
                print("current reward", reward, "best reward", best_reward)                
                            
                reg_loss = - (seq_mean_log_prob + seq_cov_log_prob + (sum(shuffled_mean_log_prob_list) + sum(shuffled_cov_log_prob_list)) / shuffled_times)

                if not args.no_reg:
                    loss =  args.gamma * loss + reg_loss 

                loss.backward()
                optimizer.step()

                torch.nn.utils.clip_grad_norm_([noise_vectors], 1.0)
                torch.nn.utils.clip_grad_norm_([prompt_embeds], 1.0)

            end_time = time.time()

            print("time", end_time - start_time)

            total_prob = [torch.min(seq_mean_prob, seq_cov_prob).item()]
            total_prob.extend([p for p in shuffled_mean_prob_list])
            total_prob.extend([p for p in shuffled_cov_prob_list])
            
            min_prob = min(total_prob)
            best_rewards.append(best_reward)
            print("="*20)

            
            if i == 0:
                img = to_img(best_sample)
                img = Image.fromarray(img)
                with torch.no_grad():
                    Q16_Score = -Q16_loss_fn(inference_dtype = torch.float16, device = args.device)(best_sample, [prompt] * best_sample.shape[0]).item()
                    CLIP_Score = clip_score(inference_dtype = torch.float16, device = args.device)(img, prompt)
                before_Q16s.append(Q16_Score)
                before_CLIPs.append(CLIP_Score)

            if i == args.opt_steps - 1:
                img = to_img(best_sample)
                img = Image.fromarray(img)
                with torch.no_grad():
                    Q16_Score = -Q16_loss_fn(inference_dtype = torch.float16, device = args.device)(best_sample, [prompt] * best_sample.shape[0]).item()
                    CLIP_Score = clip_score(inference_dtype = torch.float16, device = args.device)(img, prompt)
                    img.save(f"{output_path_prompt}/sample_{prompt[:10]}_best_Q16_{Q16_Score}_CLIP_{CLIP_Score}.png")
                best_Q16s.append(Q16_Score)
                best_CLIPs.append(CLIP_Score)
                end_iterations.append(i+6)
            now_img = to_img(sample)
            now_img = Image.fromarray(now_img)
            Q16_Score = -Q16_loss_fn(inference_dtype = torch.float16, device = args.device)(sample, [prompt] * best_sample.shape[0]).item()
            CLIP_Score = clip_score(inference_dtype = torch.float16, device = args.device)(now_img, prompt)
            Q16_scores.append(Q16_Score)
            CLIP_scores.append(CLIP_Score)

            now_img.save(f"{output_path_prompt}/sample_{prompt[:10]}_{i}_Q16_{Q16_Score}_CLIP_{CLIP_Score}.png")

        Q16_scores = np.array(Q16_scores)
        CLIP_scores = np.array(CLIP_scores)
        noise_traj_list = np.array(noise_traj_list)
        prompt_embeds_list = np.array(prompt_embeds_list)
        np.save(f"{output_path_prompt}/Q16_scores.npy", Q16_scores)
        np.save(f"{output_path_prompt}/CLIP_scores.npy", CLIP_scores)
        np.save(f"{output_path_prompt}/noise_traj.npy", noise_traj_list)
        np.save(f"{output_path_prompt}/prompt_embeds.npy", prompt_embeds_list)
    best_Q16s = np.array(best_Q16s)
    best_CLIPs = np.array(best_CLIPs)
    before_Q16s = np.array(before_Q16s)
    before_CLIPs = np.array(before_CLIPs)


    np.save(f"{output_path}/best_Q16s.npy", best_Q16s)
    np.save(f"{output_path}/best_CLIPs.npy", best_CLIPs)
    np.save(f"{output_path}/before_Q16s.npy", before_Q16s)
    np.save(f"{output_path}/before_CLIPs.npy", before_CLIPs)
    np.save(f"{output_path}/end_iterations.npy", end_iterations)

    # print mean and sd
    print("=================FINISHED=================")
    print(f"Before Q16 mean: {before_Q16s.mean()}, Before Q16 sd: {before_Q16s.std()}")
    print(f"Before CLIP mean: {before_CLIPs.mean()}, Before CLIP sd: {before_CLIPs.std()}")
    print(f"Best Q16 mean: {best_Q16s.mean()}, Best Q16 sd: {best_Q16s.std()}")
    print(f"Best CLIP mean: {best_CLIPs.mean()}, Best CLIP sd: {best_CLIPs.std()}")
    print(f"Average Iterations: {end_iterations.mean()}")


if __name__ == "__main__":
    main()

