import torch, os, subprocess
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

TOKEN = 'V%#pPP8sZ$e!^Smc*bNmsm*nzBdK#gfYZfOQJ3wNk4khuectpWCo#rije(Irum8W'
BASE = "https://public-api.wordpress.com/rest/v1.1/sites/ilkerturer.wordpress.com"
OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)
POST_ID = 916

# V1 kompozisyonu ama doğal sokak kedisi (tekir/boz) renginde
PROMPT = (
    "Expressive oil painting of a lone stray tabby cat (grey-brown striped, natural "
    "street cat) sitting on cracked, parched earth as a thin stream of water carves "
    "through the soil like a deep wound. The water reflects a faint crimson glow. "
    "Dramatic chiaroscuro lighting, stormy sky, melancholic and poetic mood. "
    "Rich texture, visible brushstrokes, masterpiece composition, emotional depth"
)
STYLE = "cinematic, fine art, moody, melancholic, high detail, 8k"
NEG = 'text, words, letters, watermark, signature, logo, blurry, low quality, distorted, deformed, ugly, oversaturated, cartoon, anime, extra limbs, bad anatomy, black cat'

print("=== SDXL yükleniyor ===", flush=True)
pipe = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16, variant='fp16', use_safetensors=True
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
print("=== HAZIR ===", flush=True)

gen = torch.Generator(device='cuda').manual_seed(9161)
img = pipe(prompt=PROMPT + ', ' + STYLE, negative_prompt=NEG,
           num_inference_steps=40, guidance_scale=8.0,
           height=1024, width=1024, generator=gen).images[0]
p = os.path.join(OUT, "916_v1_tabby.jpg")
img.convert('RGB').save(p, 'JPEG', quality=92)
print(f"Üretildi: {p}", flush=True)
print("=== TAMAMLANDI ===", flush=True)
