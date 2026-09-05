import torch, json, os, subprocess
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

TOKEN = 'V%#pPP8sZ$e!^Smc*bNmsm*nzBdK#gfYZfOQJ3wNk4khuectpWCo#rije(Irum8W'
BASE = "https://public-api.wordpress.com/rest/v1.1/sites/ilkerturer.wordpress.com"
OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)
POST_ID = 916

# Daha sanatsal, daha güçlü varyasyonlar
VARIANTS = [
    # V1: Yağlı boya tablo tarzı, güçlü kompozisyon
    {
        "prompt": (
            "Expressive oil painting of a lone black cat sitting on cracked, parched earth "
            "as a thin stream of water carves through the soil like a deep wound. The water "
            "reflects a faint crimson glow. Dramatic chiaroscuro lighting, stormy sky, "
            "melancholic and poetic mood. Rich texture, visible brushstrokes, "
            "masterpiece composition, emotional depth"
        ),
        "seed": 9161,
    },
    # V2: Minimalist, sembolik, soyut
    {
        "prompt": (
            "Minimalist symbolic fine-art illustration: a single black cat silhouette on "
            "cracked earth, a thin red-tinted stream of water flowing through a wound-like "
            "fissure in the soil. Muted earthy palette, soft diffused light, vast empty sky. "
            "Quiet, profound, poetic loneliness. Clean composition, high detail, "
            "award-winning art"
        ),
        "seed": 9162,
    },
    # V3: Sinematik, atmosferik, dramatik
    {
        "prompt": (
            "Cinematic wide shot: a small black cat alone on vast cracked earth, a narrow "
            "stream of water glowing faintly red cutting through the ground like a scar. "
            "Overcast dramatic sky, volumetric light, dust in the air. Deep melancholic "
            "atmosphere, film still quality, anamorphic, rich color grading, "
            "emotional storytelling"
        ),
        "seed": 9163,
    },
]

STYLE = "cinematic, fine art, moody, melancholic, high detail, 8k"
NEG = 'text, words, letters, watermark, signature, logo, blurry, low quality, distorted, deformed, ugly, oversaturated, cartoon, anime, extra limbs, bad anatomy'

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

paths = []
for i, v in enumerate(VARIANTS, 1):
    gen = torch.Generator(device='cuda').manual_seed(v["seed"])
    img = pipe(prompt=v["prompt"] + ', ' + STYLE, negative_prompt=NEG,
               num_inference_steps=40, guidance_scale=8.0,
               height=1024, width=1024, generator=gen).images[0]
    p = os.path.join(OUT, f"916_v{i}.jpg")
    img.convert('RGB').save(p, 'JPEG', quality=92)
    paths.append(p)
    print(f"V{i} üretildi: {p}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
print("Varyasyonlar:", paths, flush=True)
