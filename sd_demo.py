import torch
from model.sd import StableDiffusionPipeline,EC_Diff
from diffusers import DDIMScheduler

generator = torch.Generator(device="cuda:0").manual_seed(42)

cloud = StableDiffusionPipeline.from_pretrained("./checkpoint/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda:0")
cloud.scheduler = DDIMScheduler.from_config(cloud.scheduler.config)

edge = StableDiffusionPipeline.from_pretrained("./checkpoint/bk-sdm-tiny", torch_dtype=torch.float16).to("cuda:0")
edge.scheduler = DDIMScheduler.from_config(edge.scheduler.config)

pipe = EC_Diff(cloud=cloud,edge=edge)

prompt = "a cat"

#test cloud or edge model result
'''image = pipe.one_model(prompt,generator=generator,use_model='edge').images
im = image[0]
im.save("cloud.png")'''

#test hybridsd result
'''image = pipe.hybridsd(prompt,generator=generator,k=40).images
im = image[0]
im.save("hybrid.png")'''

#test ec-diff result
image = pipe(prompt,generator=generator).images
im = image[0]
im.save("ec-diff.png")