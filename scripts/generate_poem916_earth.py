import torch, os
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)

# Şiirin anlattığı: yaralı, kanayan, sahipsiz toprak (ana özne) + onun yüzünden acı çeken masum kedi (küçük, uzakta)
VARIANTS = [
    # V1: Toprak önde, kişileşmiş yaralı toprak, kan kırmızısı su, kedi küçük uzakta
    {
        "prompt": (
            "Expressive oil painting, close-up of wounded personified earth: a vast expanse "
            "of cracked, parched soil with a deep gash across its surface like a wound, "
            "a thin stream of crimson water flowing through the gash like blood. "
            "In the far background, a tiny lonely stray cat sits shivering on the dry earth. "
            "Dramatic chiaroscuro, stormy sky, overwhelming sense of abandonment and "
            "loneliness. Rich texture, visible brushstrokes, masterpiece composition"
        ),
        "seed": 9161,
    },
    # V2: Toprak merkezde, kan kırmızısı akıntı, kedi küçük, geniş boş manzara
    {
        "prompt": (
            "Cinematic fine-art photograph: a vast, empty, cracked earth field under a "
            "heavy overcast sky. A deep wound-like fissure runs across the foreground, "
            "filled with a thin stream of blood-red water. A small lonely stray cat sits "
            "far away near the fissure, looking small against the immense empty land. "
            "Moody, melancholic, profound sense of being abandoned and unowned. "
            "Volumetric light, dust in air, film still quality"
        ),
        "seed": 9162,
    },
    # V3: Sembolik, toprak kişileşmiş, kedi masum detay
    {
        "prompt": (
            "Symbolic fine-art painting: personified earth as a wounded, weeping landscape. "
            "The ground is cracked and scarred, a crimson stream of water cuts through it "
            "like a bleeding wound. A tiny innocent stray cat sits at the edge, shivering, "
            "small and vulnerable against the vast wounded land. Muted earthy palette, "
            "soft diffused light, quiet profound loneliness, emotional depth, "
            "award-winning composition"
        ),
        "seed": 9163,
    },
]

STYLE = "cinematic, fine art, moody, melancholic, high detail, 8k"
NEG = 'text, words, letters, watermark, signature, logo, blurry, low quality, distorted, deformed, ugly, oversaturated, cartoon, anime, extra limbs, bad anatomy, black cat, cat in foreground, cat as main subject'

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
    p = os.path.join(OUT, f"916_earth{i}.jpg")
    img.convert('RGB').save(p, 'JPEG', quality=92)
    paths.append(p)
    print(f"V{i} üretildi: {p}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
print("Varyasyonlar:", paths, flush=True)
