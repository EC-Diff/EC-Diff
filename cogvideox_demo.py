import argparse
from typing import Literal, Optional
import time
import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
)
from model.scheduler import DDIMScheduler
from diffusers.utils import export_to_video, load_image, load_video
from model.cogvideo import CogVideoXPipeline,EC_Diff


def generate_video(
    prompt: str,
    cloud_model_path: str,
    edge_model_path: str,
    num_frames: int = 49,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
):
    width, height = (720, 480)
   
    cloud = CogVideoXPipeline.from_pretrained(cloud_model_path, torch_dtype=dtype)
    edge = CogVideoXPipeline.from_pretrained(edge_model_path, torch_dtype=dtype)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.

    cloud.scheduler = CogVideoXDDIMScheduler.from_config(cloud.scheduler.config, timestep_spacing="trailing")
    edge.scheduler = CogVideoXDDIMScheduler.from_config(edge.scheduler.config, timestep_spacing="trailing")

    cloud.to("cuda")
    edge.to("cuda")
    
    cloud.vae.enable_slicing()
    cloud.vae.enable_tiling()
    edge.vae.enable_slicing()
    edge.vae.enable_tiling()
    
    pipe = EC_Diff(cloud=cloud,edge=edge)

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    start = time.time()

    '''video_generate = pipe.one_model(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        use_model = 'edge',
    ).frames[0]'''
    '''video_generate = pipe.hybridsd(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        k = 40,
    ).frames[0]'''
    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)
    
    end = time.time()
    use_time = end - start
    print('using time: ',use_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--cloud_model_path", type=str, default="./checkpoint/CogVideoX-5b")
    parser.add_argument(
        "--edge_model_path", type=str, default="./checkpointCogVideoX-2b")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="float16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        cloud_model_path=args.cloud_model_path,
        edge_model_path=args.edge_model_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
    )