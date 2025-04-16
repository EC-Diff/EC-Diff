import torch
from model.sdxl import StableDiffusionXLPipeline,EC_Diff
from diffusers import DDIMScheduler

generator = torch.Generator(device="cuda:0").manual_seed(18)

cloud = StableDiffusionXLPipeline.from_pretrained("./checkpoint/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda:1")
cloud.scheduler = DDIMScheduler.from_config(cloud.scheduler.config)

edge = StableDiffusionXLPipeline.from_pretrained("./checkpoint/SSD-1B", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda:1")
edge.scheduler = DDIMScheduler.from_config(edge.scheduler.config)

pipe = EC_Diff(cloud=cloud,edge=edge)

prompt = "a cat"

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 50

'''image = pipe.one_model(prompt=prompt,num_inference_steps=n_steps,generator=generator,use_model='cloud').images
im = image[0]
im.save("cloud.png")'''

'''image = pipe.hybridsd(prompt=prompt,num_inference_steps=n_steps,generator=generator,k=40).images
im = image[0]
im.save("hybrid.png")'''

image = pipe(prompt=prompt,num_inference_steps=n_steps,generator=generator).images
im = image[0]
im.save("ec-diff.png")
