import torch, os
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)

# Şiirin TÜM öğeleri tek görselde: yaralı/kanayan toprak + karnını yara yara akan su/kan + üşüyen masum kedi + sahipsizlik
VARIANTS = [
    {
        "prompt": (
            "Expressive oil painting, all elements of a poem together: a vast expanse of "
            "cracked, parched earth personified as a wounded being. A deep gash runs across "
            "its belly like a wound, and a thin stream of blood-red water flows through it, "
            "cutting the earth open. Beside the wound, a small innocent stray tabby cat sits "
            "shivering, looking small and vulnerable. The whole scene feels abandoned and "
            "unowned, vast empty land under a stormy sky. Dramatic chiaroscuro, rich texture, "
            "visible brushstrokes, masterpiece composition, emotional depth"
        ),
        "seed": 9161,
    },
    {
        "prompt": (
            "Cinematic fine-art photograph, all elements of a poem together: a vast empty "
            "cracked earth field under a heavy overcast sky. A deep wound-like fissure runs "
            "across the foreground, filled with a thin stream of blood-red water cutting "
            "through the soil. A small lonely stray tabby cat sits right beside the fissure, "
            "shivering, clearly visible, small against the immense abandoned land. "
            "Moody, melancholic, profound sense of being unowned. Volumetric light, "
            "dust in air, film still quality"
        ),
        "seed": 9162,
    },
    {
        "prompt": (
            "Symbolic fine-art painting, all elements of a poem together: personified earth "
            "as a wounded, weeping landscape. The ground is cracked and scarred, a crimson "
            "stream of water cuts through it like a bleeding wound in its belly. A small "
            "innocent stray tabby cat sits at the edge of the wound, shivering, vulnerable. "
            "Muted earthy palette, soft diffused light, quiet profound loneliness, "
            "emotional depth, award-winning composition"
        ),
        "seed": 9163,
    },
]

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

paths = []
for i, v in enumerate(VARIANTS, 1):
    gen = torch.Generator(device='cuda').manual_seed(v["seed"])
    img = pipe(prompt=v["prompt"] + ', ' + STYLE, negative_prompt=NEG,
               num_inference_steps=40, guidance_scale=8.0,
               height=1024, width=1024, generator=gen).images[0]
    p = os.path.join(OUT, f"916_all{i}.jpg")
    img.convert('RGB').save(p, 'JPEG', quality=92)
    paths.append(p)
    print(f"V{i} üretildi: {p}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
print("Varyasyonlar:", paths, flush=True)
