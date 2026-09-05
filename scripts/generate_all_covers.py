import torch, json, os, time, subprocess
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

TOKEN = 'V%#pPP8sZ$e!^Smc*bNmsm*nzBdK#gfYZfOQJ3wNk4khuectpWCo#rije(Irum8W'
BASE = "https://public-api.wordpress.com/rest/v1.1/sites/ilkerturer.wordpress.com"
OUT = "/home/ilker/projects/wordpress-blog/covers"
os.makedirs(OUT, exist_ok=True)

# Post ID mapping (title -> post_id)
POST_IDS = {
    "Mavi Karanlık": 714, "Bir Cinayettir Aşk": 715, "Son Savaş": 716,
    "Bıraktığım gibi": 717, "Kısırdöngü": 718, "Giderken": 719,
    "Yağmur ve İstanbul": 720, "Kozmos": 721, "Vertigo": 722,
    "Meczubun Türküsü": 723, "Seni İçmek": 724, "Unuttuğun": 725,
    "Sen": 726, "Bekleyen": 727, "Bir Ortadoğu Masalı": 728,
    "Yitik Hayal": 729, "Hiç gönderilmeyecek bir mektuba giriş": 730,
    "Arabeskleşmeler": 731, "Eskişehir Hatırası": 732, "Ruhların Dili": 733,
    "Şah,mat *": 734, "Komutan": 735, "Kelebek": 736,
    "Gece yolculuğu": 737, "Şeytan Gözyaşları": 738, "Tapınağın Yalnız Kralı": 739,
    "Meczup": 740, "Umutsuz": 741, "Söyleyememek": 742,
}

with open("/home/ilker/projects/wordpress-blog/data/poem_prompts.json") as f:
    P = json.load(f)
STYLE = P['style']
PROMPTS = P['prompts']

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
        d = json.loads(r.stdout)
        return d["media"][0]["ID"]
    except Exception:
        return None

def set_featured(post_id, media_id):
    cmd = ["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {TOKEN}",
           "--data-urlencode", f"featured_image={media_id}", f"{BASE}/posts/{post_id}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        d = json.loads(r.stdout)
        return "ID" in d
    except Exception:
        return False

results = []
for i, (title, pid) in enumerate(POST_IDS.items(), 1):
    if title == "Mavi Karanlık":
        results.append((title, "SKIP (testte)"))
        print(f"[{i}/29] {title}: SKIP", flush=True)
        continue
    prompt = PROMPTS[title] + ', ' + STYLE
    gen = torch.Generator(device='cuda').manual_seed(1000 + i)
    try:
        img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=30,
                   guidance_scale=7.5, height=1024, width=1024, generator=gen).images[0]
        jpg_path = os.path.join(OUT, f"{pid}.jpg")
        img.convert('RGB').save(jpg_path, 'JPEG', quality=90)
        mid = upload_media(jpg_path)
        if mid:
            ok = set_featured(pid, mid)
            results.append((title, f"OK media={mid} featured={ok}"))
            print(f"[{i}/29] {title}: OK media={mid} featured={ok}", flush=True)
        else:
            results.append((title, "HATA: upload başarısız"))
            print(f"[{i}/29] {title}: HATA upload", flush=True)
    except Exception as e:
        results.append((title, f"HATA: {str(e)[:80]}"))
        print(f"[{i}/29] {title}: HATA {str(e)[:80]}", flush=True)

print("=== TAMAMLANDI ===", flush=True)
ok = sum(1 for _,s in results if s.startswith("OK"))
err = sum(1 for _,s in results if s.startswith("HATA"))
print(f"Başarılı: {ok}, Hata: {err}, Skip: {sum(1 for _,s in results if s.startswith('SKIP'))}", flush=True)
with open("/home/ilker/projects/wordpress-blog/covers/results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
