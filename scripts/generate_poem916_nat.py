import torch, os
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)

# DOĞAL, gerçekçi fotoğraf — abartısız, sade, doğanın içinden bir kare
VARIANTS = [
    {
        "prompt": (
            "Natural realistic photograph: a small stream of water flowing through a "
            "cracked, dry earth field, cutting a narrow channel through the soil. "
            "A stray tabby cat sits quietly beside the stream on the dry ground. "
            "Soft natural daylight, overcast sky, muted earthy colors, no dramatic effects, "
            "honest and simple, like a real moment in nature"
        ),
        "seed": 9161,
    },
    {
        "prompt": (
            "Natural realistic photograph: water trickling through a wound-like crack in "
            "dry earth, the soil dark and damp where the water flows. A small stray cat "
            "sits nearby on the cracked ground, watching. Plain natural lighting, "
            "realistic colors, quiet and unadorned, a genuine scene from the countryside"
        ),
        "seed": 9162,
    },
    {
        "prompt": (
            "Natural realistic photograph: a narrow stream of water running through "
            "parched, cracked earth under soft daylight. A lonely stray cat sits at the "
            "edge of the stream, small and quiet. Simple, honest, natural composition, "
            "muted tones, no filters, like a real photograph taken in the field"
        ),
        "seed": 9163,
    },
]

STYLE = "natural, realistic, photograph, honest, simple, muted colors"
NEG = 'text, words, letters, watermark, signature, logo, blurry, low quality, distorted, deformed, ugly, oversaturated, cartoon, anime, oil painting, surreal, symbolic, dramatic, fantasy, extra limbs, bad anatomy, black cat'

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
               num_inference_steps=40, guidance_scale=7.0,
               height=1024, width=1024, generator=gen).images[0]
    p = os.path.join(OUT, f"916_nat{i}.jpg")
    img.convert('RGB').save(p, 'JPEG', quality=92)
    paths.append(p)
    print(f"V{i} üretildi: {p}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
print("Varyasyonlar:", paths, flush=True)
