import torch, json, os, time, subprocess
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

TOKEN = 'V%#pPP8sZ$e!^Smc*bNmsm*nzBdK#gfYZfOQJ3wNk4khuectpWCo#rije(Irum8W'
BASE = "https://public-api.wordpress.com/rest/v1.1/sites/ilkerturer.wordpress.com"
OUT = "/home/ilker/projects/wordpress-blog/covers/books"
os.makedirs(OUT, exist_ok=True)

with open("/home/ilker/projects/wordpress-blog/data/books_meta.json") as f:
    BOOKS = json.load(f)

STYLE = "dark romantic literary illustration, moody atmospheric lighting, cinematic composition, muted desaturated palette with deep shadows, painterly textured brushstrokes, evocative symbolic imagery, high detail, no text, no words, no letters"

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

NEG = 'text, words, letters, watermark, signature, logo, blurry, low quality, distorted, deformed, ugly, oversaturated'

def upload_media(path):
    cmd = ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {TOKEN}",
           "-F", "media[]=@" + path, BASE + "/media/new"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        return json.loads(r.stdout)["media"][0]["ID"]
    except Exception:
        return None

def update_post(pid, tags, seo):
    cmd = ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {TOKEN}",
           "--data-urlencode", f"tags={','.join(tags)}",
           "--data-urlencode", f"excerpt={seo}",
           f"{BASE}/posts/{pid}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        return "ID" in json.loads(r.stdout)
    except Exception:
        return False

def set_featured(pid, media_id):
    cmd = ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {TOKEN}",
           "--data-urlencode", f"featured_image={media_id}", f"{BASE}/posts/{pid}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        return "ID" in json.loads(r.stdout)
    except Exception:
        return False

results = []
for i, (pid, meta) in enumerate(BOOKS.items(), 1):
    title, author, genre, prompt, tags, seo = meta
    full_prompt = prompt + ', ' + STYLE
    gen = torch.Generator(device='cuda').manual_seed(2000 + int(pid))
    try:
        img = pipe(prompt=full_prompt, negative_prompt=NEG, num_inference_steps=30,
                   guidance_scale=7.5, height=1024, width=1024, generator=gen).images[0]
        jpg_path = os.path.join(OUT, f"{pid}.jpg")
        img.convert('RGB').save(jpg_path, 'JPEG', quality=90)
        mid = upload_media(jpg_path)
        if not mid:
            results.append((pid, title, "HATA: upload"))
            print(f"[{i}/33] {title}: HATA upload", flush=True)
            continue
        ok_feat = set_featured(pid, mid)
        ok_meta = update_post(pid, tags, seo)
        results.append((pid, title, f"OK media={mid} feat={ok_feat} meta={ok_meta}"))
        print(f"[{i}/33] {title}: OK media={mid} feat={ok_feat} meta={ok_meta}", flush=True)
    except Exception as e:
        results.append((pid, title, f"HATA: {str(e)[:80]}"))
        print(f"[{i}/33] {title}: HATA {str(e)[:80]}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
ok = sum(1 for _,_,s in results if s.startswith("OK"))
err = sum(1 for _,_,s in results if s.startswith("HATA"))
print(f"Başarılı: {ok}, Hata: {err}", flush=True)
with open("/home/ilker/projects/wordpress-blog/covers/books_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
