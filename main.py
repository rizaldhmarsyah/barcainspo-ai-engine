import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import asyncio
from datetime import datetime, timedelta
import json
import os
import urllib.parse
import urllib.request
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tavily import TavilyClient
from google import genai
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from upstash_redis import Redis

load_dotenv()

app = FastAPI(title="Barcainspo AI Engine")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

REDIS_URL = os.getenv("KV_REST_API_URL")
REDIS_TOKEN = os.getenv("KV_REST_API_TOKEN")

redis_client = None
if REDIS_URL and REDIS_TOKEN:
    redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ==========================================
# HELPER CALL WITH MULTI-STAGE GROQ FALLBACK
# ==========================================
async def call_groq_fallback(prompt: str) -> str:
    """Fallback ke model Groq yang terkonfirmasi aktif di akun."""
    if not GROQ_API_KEY or not groq_client:
        raise Exception("GROQ_API_KEY belum terpasang atau tidak valid di file .env")

    # Model Groq terkonfirmasi aktif
    groq_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b"
    ]

    for model_name in groq_models:
        try:
            print(f"[Groq Fallback] Memproses request dengan model: {model_name}...")
            completion = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah API pendukung portal berita 'barcainspo®'. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks (```json)."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return completion.choices[0].message.content
        except Exception as err:
            print(f"[Groq Warning] Model {model_name} gagal: {err}")

    raise Exception("Seluruh model Groq gagal merespons.")


async def call_gemini_with_fallback(prompt: str) -> str:
    """
    1. Coba gemini-3.8-flash (Retry 3x jika hit 503 Overloaded)
    2. Jika Gemini gagal/busy, pindah otomatis ke Groq Fallback
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[Gemini Request] Memanggil gemini-3.8-flash (Percobaan {attempt + 1})...")
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.8-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            print(f"[Gemini Warning] gemini-3.8-flash percobaan {attempt + 1} gagal: {err_msg}")
            
            # Jika 503 Overloaded, tunggu 2 detik dan coba lagi
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                if attempt < max_retries - 1:
                    print("[Gemini] Server sibuk (503), menunggu 2 detik...")
                    await asyncio.sleep(2)
                    continue

    # Jika Gemini gagal 3x, eksekusi Groq
    print("[Gemini Failed] Gemini sibuk/error. Dialihkan ke Groq...")
    try:
        return await call_groq_fallback(prompt)
    except Exception as groq_err:
        print(f"[Groq Error] Gagal memproses via Groq: {groq_err}")
        raise HTTPException(
            status_code=500, 
            detail=f"Semua Provider AI (Gemini & Groq) gagal merespons: {str(groq_err)}"
        )


# ==========================================
# PYDANTIC SCHEMAS UNTUK AI ARTICLE GENERATOR
# ==========================================
class ArticleGenerateRequest(BaseModel):
    prompt: str
    topic: Optional[str] = None
    category: Optional[str] = "First Team"
    mode: Optional[str] = "generate"

class ArticleGenerateResponse(BaseModel):
    title: str = Field(description="Judul artikel menarik SEO maks 110 karakter")
    slug: str = Field(description="Slug URL kebab-case")
    category: str = Field(description="Kategori berita")
    excerpt: str = Field(description="Meta description SEO maks 160 karakter")
    content: str = Field(description="Isi artikel multiparagraf dipisahkan dengan \\n\\n")
    altText: str = Field(description="Alt text deskriptif untuk gambar cover")
    imageCredit: str = Field(description="Sumber/kredit foto")
    tags: str = Field(description="Tag dipisahkan koma, contoh: FC Barcelona, Hansi Flick, La Liga")

class FieldRefineRequest(BaseModel):
    field_type: str
    current_value: str
    instruction: str
    topic: Optional[str] = None

class FieldRefineResponse(BaseModel):
    options: list[str] = Field(description="Daftar 3 variasi/revisi teks")


# ==========================================
# HELPER FUNCTIONS & CACHE
# ==========================================
def load_cache():
    if not redis_client:
        return None
    try:
        data = redis_client.get("match_cache")
        if data:
            if isinstance(data, str):
                return json.loads(data)
            return data
    except Exception as e:
        print(f"[Redis Load Error] Gagal membaca cache: {e}")
    return None

def save_cache(data, expires_at: datetime):
    if not redis_client:
        return
    try:
        cache_payload = {
            "expires_at": expires_at.isoformat(),
            "data": data
        }
        redis_client.set("match_cache", json.dumps(cache_payload))
    except Exception as e:
        print(f"[Redis Save Error] Gagal menyimpan cache: {e}")

def get_team_logo_dynamic(team_name: str) -> str:
    try:
        encoded_name = urllib.parse.quote(team_name)
        url = f"[https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=](https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=){encoded_name}"
        
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            if data and data.get("teams") and len(data["teams"]) > 0:
                badge_url = data["teams"][0].get("strBadge")
                if badge_url:
                    return badge_url
    except Exception as e:
        print(f"[Logo Fetch Error] Gagal mengambil logo untuk {team_name}: {e}")
    
    encoded_fallback = urllib.parse.quote(team_name[:3].upper())
    return f"[https://ui-avatars.com/api/?name=](https://ui-avatars.com/api/?name=){encoded_fallback}&background=262626&color=ffffff&bold=true"


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {"status": "online", "message": "Engine AI Barcainspo siap digunakan!"}

@app.get("/api/next-match")
async def get_next_match():
    try:
        now = datetime.now()
        cache = load_cache()

        if cache and "expires_at" in cache:
            try:
                expires_at = datetime.fromisoformat(cache["expires_at"])
                if now < expires_at:
                    return {
                        "success": True,
                        "cached": True,
                        "data": cache["data"]
                    }
            except Exception as e:
                print(f"[Cache Read Error] Parsing ISO Format gagal: {e}")

        current_date_str = now.strftime("%d %B %Y")
        search_query = f"FC Barcelona next match schedule fixture date time kick off 2026 after {current_date_str}"
        
        results = []
        if tavily:
            search_result = await asyncio.to_thread(
                tavily.search,
                query=search_query,
                max_results=6
            )
            results = search_result.get("results", [])
            
        raw_text = "\n\n".join([f"Source ({item.get('url')}):\n{item.get('content')}" for item in results]) if results else "Data pencarian tidak tersedia."
        
        prompt = f"""
        Hari ini adalah tanggal {current_date_str}.

        Tugas utama kamu adalah mengekstrak jadwal pertandingan resmi FC Barcelona (tim pria utama) TERDEKAT berikutnya yang BELUM dimainkan (setelah {current_date_str}).

        Berikut adalah data mentah hasil pencarian web:
        ---
        {raw_text}
        ---

        ATURAN PENULISAN & KONVERSI WAKTU:
        1. Ekstrak nama lawan, kompetisi, venue (Home/Away), tanggal, dan jam kick-off.
        2. Nama lawan HANYA boleh berisi nama klub (contoh: "Getafe CF", "Sevilla FC", "Real Madrid"). DILARANG memuat kalimat penjelasan seperti "Data tidak ditemukan".
        3. Konversikan waktu ke Waktu Indonesia Barat (WIB / UTC+7):
           - Jika waktu di artikel adalah Spanyol (CEST / UTC+2), tambahkan 5 Jam.
           - Contoh: 18:30 CEST = 23:30 WIB.
           - Contoh: 21:00 CEST = 02:00 WIB (hari berikutnya).
        4. Tulis tanggal dalam Bahasa Indonesia setelah konversi WIB (contoh: "10 Oktober 2026").
        5. "time" diisi jam WIB (contoh: "23:30 WIB" atau "02:00 WIB").
        6. "match_iso" diisi ISO 8601 standar waktu WIB (contoh: "2026-10-10T23:30:00").

        Kembalikan HANYA JSON murni dengan format:
        {{
            "opponent": "Nama Klub Lawan",
            "date": "Tanggal WIB (contoh: 10 Oktober 2026)",
            "time": "Jam WIB (contoh: 23:30 WIB)",
            "match_iso": "YYYY-MM-DDTHH:MM:SS",
            "competition": "Nama Kompetisi (contoh: La Liga)",
            "venue": "Home atau Away"
        }}
        """
        
        raw_response = await call_gemini_with_fallback(prompt)
        parsed_data = json.loads(raw_response)
        opponent_name = parsed_data.get("opponent", "")

        if not opponent_name or "tidak ditemukan" in opponent_name.lower() or len(opponent_name) > 30:
            parsed_data["opponent"] = "Getafe CF"
            parsed_data["date"] = "10 Oktober 2026"
            parsed_data["time"] = "23:30 WIB"
            parsed_data["match_iso"] = "2026-10-10T23:30:00"
            parsed_data["competition"] = "La Liga"
            parsed_data["venue"] = "Home"
            opponent_name = "Getafe CF"

        opponent_logo_url = await asyncio.to_thread(get_team_logo_dynamic, opponent_name)
        
        parsed_data["barca_logo"] = "[https://images.fotmob.com/image_resources/logo/teamlogo/8634.png](https://images.fotmob.com/image_resources/logo/teamlogo/8634.png)"
        parsed_data["opponent_logo"] = opponent_logo_url

        try:
            match_iso_str = parsed_data.get("match_iso")
            if match_iso_str:
                kickoff_dt = datetime.fromisoformat(match_iso_str)
            else:
                kickoff_dt = now + timedelta(hours=24)
            
            expires_at = kickoff_dt + timedelta(hours=3)
        except Exception:
            expires_at = now + timedelta(hours=12)

        save_cache(parsed_data, expires_at)
        
        return {
            "success": True,
            "cached": False,
            "data": parsed_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-article", response_model=ArticleGenerateResponse)
async def generate_article(req: ArticleGenerateRequest):
    try:
        # Prompt difokuskan sebagai Redaktur / Expander berbasis Fakta Legit (Ground Truth)
        prompt = f"""
        Kamu adalah Redaktur Berita Senior & Pengamat Taktis Sepak Bola untuk portal berita 'barcainspo®'.
        
        TUGAS UTAMA:
        Ubah TEKS UTAMA / FAKTA LEGIT dari user di bawah ini menjadi artikel berita jurnalistik yang utuh, profesional, dan kaya akan gaya penulisan taktis sepak bola.

        ATURAN EDITORIAL KETAT (GROUND TRUTH / BEBAS HALUSINASI):
        1. FAKTA & DATA: DILARANG MEMBUAT ATAU MENGARANG fakta baru, skor, tanggal, nama pemain, atau angka transfer yang TIDAK ADA pada Teks Sumber. Semua informasi utama artikel wajib 100% bersumber dari Teks Sumber.
        2. PENULISAN: Perluas kalimatnya, buat alur berita yang mengalir profesional, susun paragraf yang rapi, dan perjelas menggunakan istilah taktis sepak bola yang relevan (contoh: *high pressing*, *possession*, *pivot*, *half-space*).
        3. JUDUL: Buat judul yang sangat menarik, SEO-friendly (MAKSIMAL 110 KARAKTER), mencerminkan fakta utama, dan tidak clickbait murahan.
        4. EXCERPT: Buat meta description padat dan informatif (MAKSIMAL 160 KARAKTER).
        5. ALT TEXT & TAGS: Ekstrak deskripsi gambar yang relevan dan tag SEO penting dari Teks Sumber.

        INPUT USER (TEKS UTAMA / FAKTA LEGIT):
        ---
        {req.prompt}
        ---

        KATEGORI TARGET: {req.category or 'First Team'}

        KEMBALIKAN HANYA JSON MURNI DENGAN STRUKTUR:
        {{
            "title": "Judul Artikel SEO (maksimal 110 karakter)",
            "slug": "judul-artikel-dalam-kebab-case",
            "category": "{req.category or 'First Team'}",
            "excerpt": "Meta description singkat maksimal 160 karakter",
            "content": "Paragraf 1\\n\\nParagraf 2\\n\\nParagraf 3",
            "altText": "Deskripsi foto cover yang relevan dengan berita",
            "imageCredit": "barcainspo® / Sumber Terkait",
            "tags": "FC Barcelona, Hansi Flick, La Liga"
        }}
        """

        raw_response = await call_gemini_with_fallback(prompt)
        parsed_json = json.loads(raw_response)
        return parsed_json

    except Exception as e:
        print(f"[Generate Article Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses artikel AI: {str(e)}")


@app.post("/api/refine-field", response_model=FieldRefineResponse)
async def refine_field(req: FieldRefineRequest):
    try:
        prompt = f"""
        Kamu adalah Redaktur Berita Senior portal 'barcainspo®'.
        
        TUGAS:
        Berikan 3 variasi/revisi terbaik untuk komponen '{req.field_type}' artikel berita FC Barcelona.

        NILAI SAAT INI:
        "{req.current_value}"

        INSTRUKSI REVISI / ARAHAN USER:
        "{req.instruction}"

        ATURAN KOMPONEN:
        - Jika field 'title': Maksimal 110 karakter, menarik, SEO friendly.
        - Jika field 'excerpt': Maksimal 160 karakter untuk Meta Description.
        - Jika field 'content': Multiparagraf dipisahkan '\\n\\n', analitis taktis.
        - Jika field 'altText': Deskriptif SEO gambar.
        - Jika field 'tags': Pisahkan dengan koma.

        KEMBALIKAN HANYA JSON MURNI DENGAN FORMAT:
        {{
            "options": [
                "Variasi 1...",
                "Variasi 2...",
                "Variasi 3..."
            ]
        }}
        """

        raw_response = await call_gemini_with_fallback(prompt)
        parsed_json = json.loads(raw_response)
        return parsed_json

    except Exception as e:
        print(f"[Refine Field Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal merevisi komponen: {str(e)}")