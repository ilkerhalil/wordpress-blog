#!/usr/bin/env node
/**
 * blog_raw.json + poems_raw.json → Astro content koleksiyonu markdown dosyaları
 * Çıktı: src/content/books/*.md ve src/content/poems/*.md
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const DATA = join(ROOT, 'data');
const BOOKS_OUT = join(ROOT, 'src', 'content', 'books');
const POEMS_OUT = join(ROOT, 'src', 'content', 'poems');

mkdirSync(BOOKS_OUT, { recursive: true });
mkdirSync(POEMS_OUT, { recursive: true });

// ---------- HTML entity decode ----------
function decodeEntities(s) {
  const map = {
    '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#8220;': '“', '&#8221;': '”', '&#8216;': '‘', '&#8217;': '’',
    '&#8211;': '–', '&#8212;': '—', '&#8230;': '…', '&#8217;': '’',
    '&#39;': "'", '&rsquo;': '’', '&ldquo;': '“', '&rdquo;': '”',
    '&hellip;': '…', '&mdash;': '—', '&ndash;': '–',
  };
  return s.replace(/&(?:#\d+|#x[\da-fA-F]+|\w+);/g, (m) => {
    if (m.startsWith('&#x')) return String.fromCodePoint(parseInt(m.slice(3, -1), 16));
    if (m.startsWith('&#')) return String.fromCodePoint(parseInt(m.slice(2, -1), 10));
    return map[m] ?? m;
  });
}

// ---------- HTML kalıntılarını temizle ----------
function cleanHtml(s) {
  // WordPress reklam/script bloklarını at (script tag'lı ve tagsız __ATA blokları)
  s = s.replace(/<script[\s\S]*?<\/script>/gi, '');
  s = s.replace(/<style[\s\S]*?<\/style>/gi, '');
  s = s.replace(/<!--[\s\S]*?-->/g, '');
  s = s.replace(/__ATA[\s\S]*?initVideoSlot\([^)]*\)[\s\S]*?\);\s*$/gm, '');
  s = s.replace(/__ATA[\s\S]*?\);\s*$/gm, '');
  // kalan tag'ları kaldır
  s = s.replace(/<[^>]+>/g, '');
  return s;
}

// ---------- düz metin → markdown paragraflar ----------
function toMarkdown(s) {
  s = decodeEntities(s);
  s = cleanHtml(s);
  // satır sonlarını normalize et
  s = s.replace(/\r\n/g, '\n');
  // 3+ boş satırı 2'ye indir
  s = s.replace(/\n{3,}/g, '\n\n');
  // tek satır sonlarını boşluk yap (paragraf bölmek için çift satır gerek)
  s = s.replace(/([^\n])\n([^\n])/g, '$1 $2');
  return s.trim();
}

// ---------- slug ----------
function slugify(s) {
  return s
    .toLowerCase()
    .replace(/[ğ]/g, 'g').replace(/[ü]/g, 'u').replace(/[ş]/g, 's')
    .replace(/[ı]/g, 'i').replace(/[ö]/g, 'o').replace(/[ç]/g, 'c')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// ---------- KİTAPLAR ----------
const blog = JSON.parse(readFileSync(join(DATA, 'blog_raw.json'), 'utf8'));
let bookCount = 0;
for (const [url, v] of Object.entries(blog)) {
  const m = url.match(/https:\/\/ilkerturer\.wordpress\.com\/(\d{4})\/(\d{2})\/(\d{2})\/([^/]+)\//);
  if (!m) continue;
  const [, year, month, day, slug] = m;
  const title = decodeEntities(v.title).replace(/\s+/g, ' ').trim();
  const body = toMarkdown(v.body);
  const date = `${year}-${month}-${day}`;
  // İçerikten anlamlı description üret (ilk cümle, 150-160 karakter)
  const firstSentence = body
    .replace(/\s+/g, ' ')
    .match(/[^.!?]+[.!?]/);
  const excerpt = (firstSentence ? firstSentence[0] : body.slice(0, 160))
    .trim()
    .slice(0, 155);
  const description = `${excerpt} — İlker Halil Türer'in kitap incelemesi.`;
  const fm = `---
title: "${title.replace(/"/g, '\\"')}"
date: ${date}
description: "${description.replace(/"/g, '\\"')}"
tags: ["kitap", "inceleme"]
---

${body}
`;
  writeFileSync(join(BOOKS_OUT, `${slug}.md`), fm);
  bookCount++;
}

// ---------- ŞİİRLER ----------
const poems = JSON.parse(readFileSync(join(DATA, 'poems_raw.json'), 'utf8'));
const prompts = JSON.parse(readFileSync(join(DATA, 'poem_prompts.json'), 'utf8')).prompts;

// Şiir post ID'leri (kapak görseli eşleştirme)
const POST_IDS = {
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
};

let poemCount = 0;
for (const [title, text] of Object.entries(poems)) {
  // tarihi çıkar
  const tm = text.match(/Kayıt Tarihi\s*:\s*([\d.]+)\s+([\d:]+)/);
  let date = '2007-01-01';
  if (tm) {
    const [d, t] = tm[1].split('.').map(Number).reverse(); // dd.mm.yyyy → yyyy,mm,dd
    date = `${d}-${String(tm[1].split('.')[1]).padStart(2, '0')}-${String(tm[1].split('.')[0]).padStart(2, '0')}`;
  }
  // imza ve tarih satırlarını temizle
  let body = text
    .replace(/\n?\s*İlker Halil Türer\s*\n?/g, '\n')
    .replace(/\n?\s*Kayıt Tarihi\s*:\s*[\d.]+ [\d:]+\s*\n?/g, '')
    .trim();
  body = decodeEntities(body);
  // şiir satır sonlarını markdown'da koru (her satır sonuna 2 boşluk)
  body = body.replace(/\n/g, '  \n');
  const prompt = prompts[title] || '';
  const postId = POST_IDS[title];
  const cover = postId ? `/covers/${postId}.jpg` : '';
  // Şiirin ilk mısralarından description üret
  const poemLines = body
    .replace(/  \n/g, '\n')
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  const firstLines = poemLines.slice(0, 3).join(' ');
  const poemDesc = `${firstLines.slice(0, 120)} — ${title} şiiri, İlker Halil Türer.`;
  const fm = `---
title: "${title.replace(/"/g, '\\"')}"
date: ${date}
description: "${poemDesc.replace(/"/g, '\\"')}"
tags: ["şiir"]
coverPrompt: "${(prompt || '').replace(/"/g, '\\"')}"
cover: "${cover}"
---

${body}
`;
  writeFileSync(join(POEMS_OUT, `${slugify(title)}.md`), fm);
  poemCount++;
}

console.log(`✓ ${bookCount} kitap → src/content/books/`);
console.log(`✓ ${poemCount} şiir → src/content/poems/`);
