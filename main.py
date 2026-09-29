import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import asyncio
from datetime import datetime, timedelta
import json
import os
import re
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

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def clean_json_string(text: str) -> str:
    """Membersihkan markdown code blocks agar json.loads() tidak crash."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


# ==========================================
# HELPER CALL WITH MULTI-STAGE GROQ FALLBACK
# ==========================================
async def call_groq_fallback(prompt: str) -> str:
    """Fallback ke model Groq aktif dan terkonfirmasi valid."""
    if not GROQ_API_KEY or not groq_client:
        raise Exception("GROQ_API_KEY tidak dikonfigurasi di .env")

    # Model Groq resmi dan stabil
    groq_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192"
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
                        "content": "Kamu adalah API pendukung portal berita 'barcainspo®'. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks."
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
    """Coba Gemini 2.5 Flash, jika gagal switch otomatis ke Groq."""
    if client and GEMINI_API_KEY:
        gemini_models = ["gemini-2.5-flash"]
        for model_name in gemini_models:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"[Gemini Request] Memanggil {model_name} (Percobaan {attempt + 1})...")
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    return response.text
                except Exception as e:
                    err_msg = str(e)
                    print(f"[Gemini Warning] {model_name} percobaan {attempt + 1} gagal: {err_msg}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)

    print("[Gemini Unavailable] Memindahkan proses ke Groq Fallback...")
    return await call_groq_fallback(prompt)


# ==========================================
# PYDANTIC SCHEMAS
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
    tags: str = Field(description="Tag dipisahkan koma")

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
    now = datetime.now()
    
    # 1. Cek Cache Redis terlebih dahulu
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
        try:
            search_result = await asyncio.to_thread(
                tavily.search,
                query=search_query,
                max_results=5
            )
            results = search_result.get("results", [])
        except Exception as t_err:
            print(f"[Tavily Warning] Search failed: {t_err}")

    raw_text = "\n\n".join([f"Source ({item.get('url')}):\n{item.get('content')}" for item in results]) if results else "Data pencarian tidak tersedia."
    
    prompt = f"""
    Hari ini adalah tanggal {current_date_str}.
    Tugas utama kamu adalah mengekstrak jadwal pertandingan resmi FC Barcelona TERDEKAT berikutnya setelah tanggal {current_date_str}.

    Data Mentah:
    ---
    {raw_text}
    ---

    Aturan output JSON:
    1. Konversi waktu ke WIB (UTC+7).
    2. Tanggal Bahasa Indonesia (contoh: "10 Oktober 2026").
    3. Output HANYA JSON murni tanpa backticks markdown:
    {{
        "opponent": "Nama Klub Lawan",
        "date": "Tanggal WIB",
        "time": "Jam WIB (contoh: 23:30 WIB)",
        "match_iso": "YYYY-MM-DDTHH:MM:SS",
        "competition": "Nama Kompetisi",
        "venue": "Home atau Away"
    }}
    """

    try:
        raw_response = await call_gemini_with_fallback(prompt)
        cleaned_response = clean_json_string(raw_response)
        parsed_data = json.loads(cleaned_response)
    except Exception as ai_err:
        print(f"[AI Process Warning] Gagal memproses AI, menggunakan data fallback aman: {ai_err}")
        # Default Fallback Data agar UI tidak crash jika semua provider AI down
        parsed_data = {
            "opponent": "Getafe CF",
            "date": "10 Oktober 2026",
            "time": "23:30 WIB",
            "match_iso": "2026-10-10T23:30:00",
            "competition": "La Liga",
            "venue": "Home"
        }

    opponent_name = parsed_data.get("opponent", "Getafe CF")
    if not opponent_name or "tidak ditemukan" in opponent_name.lower() or len(opponent_name) > 30:
        opponent_name = "Getafe CF"
        parsed_data["opponent"] = opponent_name

    opponent_logo_url = await asyncio.to_thread(get_team_logo_dynamic, opponent_name)
    parsed_data["barca_logo"] = "[https://images.fotmob.com/image_resources/logo/teamlogo/8634.png](https://images.fotmob.com/image_resources/logo/teamlogo/8634.png)"
    parsed_data["opponent_logo"] = opponent_logo_url

    expires_at = now + timedelta(hours=6)
    save_cache(parsed_data, expires_at)

    return {
        "success": True,
        "cached": False,
        "data": parsed_data
    }


@app.post("/api/generate-article", response_model=ArticleGenerateResponse)
async def generate_article(req: ArticleGenerateRequest):
    try:
        prompt = f"""
        Kamu adalah Redaktur Berita Senior & Pengamat Taktis Sepak Bola untuk portal berita 'barcainspo®'.
        
        TUGAS UTAMA:
        Ubah TEKS UTAMA dari user di bawah ini menjadi artikel berita jurnalistik yang utuh.

        INPUT USER:
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
        cleaned_response = clean_json_string(raw_response)
        parsed_json = json.loads(cleaned_response)
        return parsed_json

    except Exception as e:
        print(f"[Generate Article Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses artikel AI: {str(e)}")


@app.post("/api/refine-field", response_model=FieldRefineResponse)
async def refine_field(req: FieldRefineRequest):
    try:
        prompt = f"""
        Kamu adalah Redaktur Berita Senior portal 'barcainspo®'.
        
        Berikan 3 variasi/revisi terbaik untuk komponen '{req.field_type}' artikel berita FC Barcelona.
        NILAI SAAT INI: "{req.current_value}"
        INSTRUKSI REVISI: "{req.instruction}"

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
        cleaned_response = clean_json_string(raw_response)
        parsed_json = json.loads(cleaned_response)
        return parsed_json

    except Exception as e:
        print(f"[Refine Field Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal merevisi komponen: {str(e)}")