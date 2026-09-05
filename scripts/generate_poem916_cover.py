import torch, json, os, time, subprocess
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

TOKEN = 'V%#pPP8sZ$e!^Smc*bNmsm*nzBdK#gfYZfOQJ3wNk4khuectpWCo#rije(Irum8W'
BASE = "https://public-api.wordpress.com/rest/v1.1/sites/ilkerturer.wordpress.com"
OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)
POST_ID = 916

# Şiir teması: su toprağın karnını yara yara akıyor, yara, kan, sahipsiz kedi, melankoli
PROMPT = (
    "A lone black cat sitting on cracked, dry earth as a thin stream of water "
    "cuts through the soil like a wound. Melancholic, poetic, moody atmosphere. "
    "Dramatic overcast sky, muted earthy tones, deep shadows. The water glints "
    "faintly red as if carrying blood. Surreal, emotional, fine-art photography style, "
    "cinematic lighting, high detail"
)
STYLE = "cinematic, fine art, moody, melancholic, high detail, 8k"
NEG = 'text, words, letters, watermark, signature, logo, blurry, low quality, distorted, deformed, ugly, oversaturated, cartoon, anime'

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

gen = torch.Generator(device='cuda').manual_seed(916)
img = pipe(prompt=PROMPT + ', ' + STYLE, negative_prompt=NEG,
           num_inference_steps=35, guidance_scale=7.5,
           height=1024, width=1024, generator=gen).images[0]
jpg_path = os.path.join(OUT, f"{POST_ID}.jpg")
img.convert('RGB').save(jpg_path, 'JPEG', quality=90)
print(f"Görsel üretildi: {jpg_path}", flush=True)

# Upload to WordPress
cmd = ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {TOKEN}",
       "-F", f"media[]=@{jpg_path}", BASE + "/media/new"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
try:
    d = json.loads(r.stdout)
    mid = d["media"][0]["ID"]
    print(f"Media ID: {mid}", flush=True)
except Exception as e:
    print(f"Upload HATA: {e} | {r.stdout[:200]}", flush=True)
    mid = None

if mid:
    cmd2 = ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {TOKEN}",
            "--data-urlencode", f"featured_image={mid}", f"{BASE}/posts/{POST_ID}"]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
    try:
        d2 = json.loads(r2.stdout)
        print(f"Featured set: {'OK' if 'ID' in d2 else d2}", flush=True)
    except Exception as e:
        print(f"Featured HATA: {e} | {r2.stdout[:200]}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
