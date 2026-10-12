import os
import re
import json
import asyncio
import warnings
import urllib.parse
import urllib.request
from typing import Optional, List
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from groq import Groq
from tavily import TavilyClient
from upstash_redis import Redis

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

load_dotenv()

app = FastAPI(title="Barcainspo AI Engine")

# ==========================================
# ENV & CLIENT INITIALIZATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

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

mistral_client = None
if MISTRAL_API_KEY:
    try:
        from mistralai import Mistral
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except ImportError:
        print("[Warning] Package 'mistralai' belum terinstall. Install via: pip install mistralai")


def clean_json_string(text: str) -> str:
    """Membersihkan markdown code blocks agar json.loads() tidak crash."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


# ==========================================
# MULTI-TIER AI FAILOVER MECHANISM
# ==========================================
async def call_mistral_fallback(prompt: str) -> str:
    """Tier 3: Fallback paling akhir menggunakan Mistral AI."""
    if not MISTRAL_API_KEY or not mistral_client:
        raise Exception("MISTRAL_API_KEY belum terpasang atau client Mistral tidak aktif.")

    mistral_models = ["mistral-small-latest", "open-mistral-7b"]

    for model_name in mistral_models:
        try:
            print(f"[Mistral Fallback] Memproses request dengan model: {model_name}...")
            response = await asyncio.to_thread(
                mistral_client.chat.complete,
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah API pendukung portal berita 'barcainspo®'. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
        except Exception as err:
            print(f"[Mistral Warning] Model {model_name} gagal: {err}")

    raise Exception("Seluruh model Mistral gagal merespons.")


async def call_groq_fallback(prompt: str) -> str:
    """Tier 2: Fallback ke model Groq."""
    if not GROQ_API_KEY or not groq_client:
        print("[Groq Skipped] GROQ_API_KEY belum terpasang.")
    else:
        groq_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
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
                            "content": "Kamu adalah API pendukung portal berita 'barcainspo®'. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                if completion and completion.choices[0].message.content:
                    return completion.choices[0].message.content
            except Exception as err:
                print(f"[Groq Warning] Model {model_name} gagal: {err}")

    print("[Groq Failed] Semua model Groq gagal/di-skip. Dialihkan ke Mistral...")
    return await call_mistral_fallback(prompt)


async def call_gemini_with_fallback(prompt: str) -> str:
    """Tier 1 -> Tier 2 -> Tier 3 AI Failover Flow."""
    if client and GEMINI_API_KEY:
        gemini_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

        for model_name in gemini_models:
            try:
                print(f"[Gemini Request] Memanggil {model_name}...")
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                print(f"[Gemini Warning] {model_name} gagal: {e}")

    print("[Gemini Failed] Semua model Gemini sibuk/error. Dialihkan ke Groq...")
    try:
        return await call_groq_fallback(prompt)
    except Exception as fallback_err:
        print(f"[AI Chain Error] Seluruh Provider AI gagal: {fallback_err}")
        raise HTTPException(
            status_code=500,
            detail=f"Semua Provider AI (Gemini, Groq, & Mistral) gagal merespons: {str(fallback_err)}",
        )


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
    tags: str = Field(description="Tag dipisahkan koma, contoh: FC Barcelona, Hansi Flick, La Liga")


class FieldRefineRequest(BaseModel):
    field_type: str
    current_value: str
    instruction: str
    topic: Optional[str] = None


class FieldRefineResponse(BaseModel):
    options: list[str] = Field(description="Daftar 3 variasi/revisi teks")


class XNewsFetchRequest(BaseModel):
    handles: Optional[List[str]] = ["FabrizioRomano", "gerardromero", "ToniJuanmarti", "ReshadFCB"]
    max_results: Optional[int] = 5


class XNewsLeadItem(BaseModel):
    sourceHandle: str
    headline: str
    summary: str
    rawContent: str
    tweetUrl: Optional[str] = None
    createdAt: str


class XNewsFetchResponse(BaseModel):
    success: bool
    count: int
    leads: List[XNewsLeadItem]


# ==========================================
# HELPER FUNCTIONS & LOGO API
# ==========================================
def load_cache():
    if not redis_client:
        return None
    try:
        data = redis_client.get("match_cache_v2")
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
        cache_payload = {"expires_at": expires_at.isoformat(), "data": data}
        redis_client.set("match_cache_v2", json.dumps(cache_payload))
    except Exception as e:
        print(f"[Redis Save Error] Gagal menyimpan cache: {e}")


def get_team_logo_dynamic(team_name: str) -> str:
    """Mengambil logo tim dari Sports API / Wikimedia / Fallback UI Avatars."""
    if not team_name:
        return "[https://crests.football-data.org/724.png](https://crests.football-data.org/724.png)"

    try:
        encoded_name = urllib.parse.quote(team_name)
        url = f"[https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=](https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=){encoded_name}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            if data and data.get("teams") and len(data["teams"]) > 0:
                badge_url = data["teams"][0].get("strBadge")
                if badge_url and badge_url.startswith("http"):
                    return badge_url
    except Exception as e:
        print(f"[Logo Fetch Warning] TheSportsDB lookup failed for {team_name}: {e}")

    encoded_fallback = urllib.parse.quote(team_name)
    return f"[https://ui-avatars.com/api/?name=](https://ui-avatars.com/api/?name=){encoded_fallback}&background=1e293b&color=ffffff&bold=true&rounded=true"


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Engine AI Barcainspo siap digunakan!",
    }


@app.get("/api/next-match")
async def get_next_match():
    try:
        now = datetime.now()
        cache = load_cache()

        # 1. CEK CACHE REDIS
        if cache and "expires_at" in cache:
            try:
                expires_at = datetime.fromisoformat(cache["expires_at"])
                if now < expires_at:
                    return {
                        "success": True,
                        "cached": True,
                        "data": cache["data"],
                    }
            except Exception as e:
                print(f"[Cache Read Error] Parsing ISO Format gagal: {e}")

        current_date_str = now.strftime("%d %B %Y")
        
        # 2. QUERY TAVILY SEARCH (Pencarian Seluruh Kompetisi)
        search_query = (
            f"FC Barcelona upcoming fixture schedule Champions League La Liga "
            f"date kick off time 2026 after {current_date_str}"
        )

        results = []
        if tavily:
            try:
                search_result = await asyncio.to_thread(
                    tavily.search, query=search_query, max_results=8, search_depth="advanced"
                )
                results = search_result.get("results", [])
            except Exception as t_err:
                print(f"[Tavily Search Warning]: {t_err}")

        raw_text = (
            "\n\n".join(
                [
                    f"Source ({item.get('url')}):\n{item.get('content')}"
                    for item in results
                ]
            )
            if results
            else "Data pencarian web tidak tersedia."
        )

        # 3. PROMPT LLM GEMINI
        prompt = f"""
        Hari ini adalah tanggal {current_date_str}.

        TUGAS UTAMA:
        Cari dan ekstrak 1 PERTANDINGAN TERDEKAT BARCELONA (tim utama pria) yang BELUM dimainkan setelah tanggal {current_date_str}.

        PERIKSA SELURUH KOMPETISI:
        - UEFA Champions League (UCL)
        - La Liga
        - Copa del Rey / Supercopa

        DATA MENTAH PENCARIAN WEB:
        ---
        {raw_text}
        ---

        ATURAN PENULISAN & KONVERSI WAKTU:
        1. "opponent": Nama resmi klub lawan (Contoh: "Galatasaray", "Real Betis", "Real Madrid"). DILARANG memuat kata tambahan seperti "Data tidak ditemukan".
        2. "competition": Nama kompetisi (Contoh: "UEFA Champions League" atau "La Liga").
        3. "venue": Tulis "Home" atau "Away".
        4. KONVERSI JAM KE WIB (UTC+7):
           - Jika waktu di Europe/Spain (CEST / UTC+2), TAMBAHKAN 5 JAM ke WIB.
           - Pastikan jam kick-off diisi secara akurat (contoh: "02:00 WIB" atau "23:30 WIB").
        5. "date": Tulis tanggal lengkap WIB dalam Bahasa Indonesia (contoh: "14 Oktober 2026" atau "17 Oktober 2026").
        6. "match_iso": Format ISO 8601 standar waktu WIB (contoh: "2026-10-14T02:00:00").

        KEMBALIKAN HANYA JSON MURNI DENGAN STRUKTUR:
        {{
            "opponent": "Galatasaray",
            "date": "14 Oktober 2026",
            "time": "02:00 WIB",
            "match_iso": "2026-10-14T02:00:00",
            "competition": "UEFA Champions League",
            "venue": "Away"
        }}
        """

        raw_response = await call_gemini_with_fallback(prompt)
        cleaned_response = clean_json_string(raw_response)
        parsed_data = json.loads(cleaned_response)

        opponent_name = parsed_data.get("opponent", "")

        # Fallback jika parsing LLM gagal/kosong
        if not opponent_name or "tidak ditemukan" in opponent_name.lower() or len(opponent_name) > 35:
            parsed_data["opponent"] = "Galatasaray"
            parsed_data["date"] = "14 Oktober 2026"
            parsed_data["time"] = "02:00 WIB"
            parsed_data["match_iso"] = "2026-10-14T02:00:00"
            parsed_data["competition"] = "UEFA Champions League"
            parsed_data["venue"] = "Away"
            opponent_name = "Galatasaray"

        # Fetch Logo Dinamis
        opponent_logo_url = await asyncio.to_thread(
            get_team_logo_dynamic, opponent_name
        )

        parsed_data["barca_logo"] = "[https://crests.football-data.org/81.png](https://crests.football-data.org/81.png)"
        parsed_data["opponent_logo"] = opponent_logo_url

        # 4. LOGIKA SMART CACHE (DYNAMICAL & TOKEN SAVER)
        try:
            match_iso_str = parsed_data.get("match_iso")
            if match_iso_str:
                kickoff_dt = datetime.fromisoformat(match_iso_str)
            else:
                kickoff_dt = now + timedelta(hours=24)

            match_end_dt = kickoff_dt + timedelta(hours=3)
            hours_until_kickoff = (kickoff_dt - now).total_seconds() / 3600

            if now >= match_end_dt:
                # 1. Pertandingan sudah selesai (>3 jam setelah kick-off): langsung expire sekarang
                expires_at = now
            elif now >= kickoff_dt - timedelta(hours=3) and now < match_end_dt:
                # 2. Sedang bertanding: expire tepat setelah pertandingan usai
                expires_at = match_end_dt
            elif hours_until_kickoff <= 24:
                # 3. Mendekati pertandingan (<24 jam): update tiap 6 jam
                expires_at = now + timedelta(hours=6)
            else:
                # 4. Pertandingan masih lama (>24 jam): HEMAT TOKEN! Simpan cache selama 24 JAM (1 Hari)
                expires_at = now + timedelta(hours=24)

        except Exception:
            expires_at = now + timedelta(hours=24)

        # Simpan ke Redis
        save_cache(parsed_data, expires_at)

        return {"success": True, "cached": False, "data": parsed_data}

    except Exception as e:
        print(f"[Next Match API Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-article", response_model=ArticleGenerateResponse)
async def generate_article(req: ArticleGenerateRequest):
    try:
        prompt = f"""
        Kamu adalah Redaktur Berita Senior & Pengamat Taktis Sepak Bola untuk portal berita 'barcainspo®'.
        
        TUGAS UTAMA:
        Ubah TEKS UTAMA / FAKTA LEGIT dari user di bawah ini menjadi artikel berita jurnalistik yang utuh, profesional, dan kaya akan gaya penulisan taktis sepak bola.

        ATURAN EDITORIAL KETAT (GROUND TRUTH / BEBAS HALUSINASI):
        1. FAKTA & DATA: DILARANG MEMBUAT ATAU MENGARANG fakta baru, skor, tanggal, nama pemain, atau angka transfer yang TIDAK ADA pada Teks Sumber.
        2. PENULISAN: Perluas kalimatnya, buat alur berita yang mengalir profesional, susun paragraf yang rapi, dan perjelas menggunakan istilah taktis sepak bola.
        3. JUDUL: Buat judul yang sangat menarik, SEO-friendly (MAKSIMAL 110 KARAKTER).
        4. EXCERPT: Buat meta description padat dan informatif (MAKSIMAL 160 KARAKTER).

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
        cleaned_response = clean_json_string(raw_response)
        parsed_json = json.loads(cleaned_response)
        return parsed_json

    except Exception as e:
        print(f"[Generate Article Error]: {e}")
        raise HTTPException(
            status_code=500, detail=f"Gagal memproses artikel AI: {str(e)}"
        )


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
        raise HTTPException(
            status_code=500, detail=f"Gagal merevisi komponen: {str(e)}"
        )


@app.post("/api/x-news/fetch", response_model=XNewsFetchResponse)
async def fetch_x_news_leads(req: XNewsFetchRequest):
    try:
        today_dt = datetime.now()
        today_iso = today_dt.strftime("%Y-%m-%d")
        current_date_str = today_dt.strftime("%d %B %Y")

        clean_handles = [h.replace("@", "").strip() for h in (req.handles or []) if h.strip()]
        if not clean_handles:
            clean_handles = ["FabrizioRomano", "BarcaUniversal"]

        handles_query = " OR ".join([f'"{h}"' for h in clean_handles])
        search_query = f"({handles_query}) Barcelona news transfer"

        results = []
        if tavily:
            try:
                search_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        tavily.search,
                        query=search_query,
                        max_results=int(req.max_results or 5),
                        search_depth="basic",
                    ),
                    timeout=15.0,
                )
                results = search_result.get("results", [])
            except asyncio.TimeoutError:
                print("⚠️ [Tavily Warning]: Search timed out after 15 seconds.")
                results = []
            except Exception as t_err:
                print(f"⚠️ [Tavily Error]: {t_err}")
                results = []

        if not results:
            return {"success": True, "count": 0, "leads": []}

        raw_text = "\n\n".join([
            f"Source URL ({item.get('url')}):\n{item.get('content')}"
            for item in results
        ])

        prompt = f"""
        Hari ini adalah tanggal {current_date_str} (ISO: {today_iso}).
        Kamu adalah Head Editor Berita Utama untuk portal berita olahraga 'barcainspo®'.

        TUGAS UTAMA:
        Analisis teks mentah hasil pemantauan dari X (Twitter) berikut ini.
        Saring HANYA cuitan yang MEMILIKI VALUE BERITA UTAMA FC BARCELONA dan SANGAT LAYAK DIBAWA KE DRAFT ARTIKEL.

        DATA MENTAH DARI X:
        ---
        {raw_text}
        ---

        STRUKTUR JSON MUSTI PERSIS SEPERTI INI:
        {{
            "leads": [
                {{
                    "sourceHandle": "@{clean_handles[0]}",
                    "headline": "Judul berita yang padat, menarik, dan informatif dalam Bahasa Indonesia",
                    "summary": "Ringkasan poin-poin fakta utama berita dalam Bahasa Indonesia (2-3 kalimat)",
                    "rawContent": "Kutipan teks asli dari cuitan tersebut",
                    "tweetUrl": "URL sumber asli dari data di atas jika ada",
                    "createdAt": "{today_iso}"
                }}
            ]
        }}
        """

        try:
            raw_response = await asyncio.wait_for(
                call_gemini_with_fallback(prompt),
                timeout=30.0
            )
            cleaned_response = clean_json_string(raw_response)
            parsed_data = json.loads(cleaned_response)
            leads_data = parsed_data.get("leads", [])
            if not isinstance(leads_data, list):
                leads_data = []
        except Exception as llm_err:
            print(f"⚠️ [LLM Parsing Fallback]: {llm_err}")
            leads_data = []
            for item in results[:3]:
                leads_data.append({
                    "sourceHandle": f"@{clean_handles[0]}",
                    "headline": item.get("title", "Update Berita Terkini FC Barcelona"),
                    "summary": item.get("content", "")[:200] + "...",
                    "rawContent": item.get("content", ""),
                    "tweetUrl": item.get("url", ""),
                    "createdAt": today_iso,
                })

        return {
            "success": True,
            "count": len(leads_data),
            "leads": leads_data
        }

    except Exception as e:
        print(f"[X-News Agent Error Handled]: {e}")
        return {
            "success": False,
            "count": 0,
            "leads": []
        }