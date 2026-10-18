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
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

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
        print("[Warning] Package 'mistralai' belum terinstall.")


def clean_json_string(text: str) -> str:
    """Membersihkan markdown code blocks agar json.loads() tidak crash."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def get_team_code(team_name: str) -> str:
    """Mengembalikan singkatan resmi 3 huruf klub sepak bola."""
    if not team_name:
        return "OPP"
    
    name_lower = team_name.lower().strip()
    mapping = {
        "paris saint-germain": "PSG",
        "paris saint germain": "PSG",
        "psg": "PSG",
        "real madrid": "RMA",
        "galatasaray": "GAL",
        "real betis": "BET",
        "atletico madrid": "ATM",
        "atletico de madrid": "ATM",
        "athletic club": "ATH",
        "girona": "GIR",
        "sevilla": "SEV",
        "villarreal": "VIL",
        "real sociedad": "RSO",
        "manchester city": "MCI",
        "bayern munich": "BAY",
        "arsenal": "ARS",
    }
    
    for key, code in mapping.items():
        if key in name_lower:
            return code
            
    clean = re.sub(r'[^a-zA-Z]', '', team_name)
    return clean[:3].upper() if len(clean) >= 3 else "OPP"


# PETA LOGO KILAT (Direct CDN High-Res)
KNOWN_LOGOS = {
    "paris saint-germain": "[https://crests.football-data.org/524.png](https://crests.football-data.org/524.png)",
    "paris saint germain": "[https://crests.football-data.org/524.png](https://crests.football-data.org/524.png)",
    "psg": "[https://crests.football-data.org/524.png](https://crests.football-data.org/524.png)",
    "real madrid": "[https://crests.football-data.org/86.png](https://crests.football-data.org/86.png)",
    "galatasaray": "[https://crests.football-data.org/610.png](https://crests.football-data.org/610.png)",
    "real betis": "[https://crests.football-data.org/90.png](https://crests.football-data.org/90.png)",
    "betis": "[https://crests.football-data.org/90.png](https://crests.football-data.org/90.png)",
    "atletico madrid": "[https://crests.football-data.org/78.png](https://crests.football-data.org/78.png)",
    "girona": "[https://crests.football-data.org/298.png](https://crests.football-data.org/298.png)",
    "sevilla": "[https://crests.football-data.org/559.png](https://crests.football-data.org/559.png)",
    "villarreal": "[https://crests.football-data.org/94.png](https://crests.football-data.org/94.png)",
    "bayern munich": "[https://crests.football-data.org/5.png](https://crests.football-data.org/5.png)",
    "manchester city": "[https://crests.football-data.org/65.png](https://crests.football-data.org/65.png)",
    "arsenal": "[https://crests.football-data.org/57.png](https://crests.football-data.org/57.png)",
}


def get_team_logo_dynamic(team_name: str) -> str:
    """Mengambil logo tim dengan prioritas Kamus Kilat -> API-Football -> Fallback Avatar."""
    if not team_name:
        return "[https://crests.football-data.org/724.png](https://crests.football-data.org/724.png)"

    clean_name = team_name.strip().lower()

    # 1. Cek Kamus Logo Kilat
    for key, logo_url in KNOWN_LOGOS.items():
        if key in clean_name or clean_name in key:
            return logo_url

    # 2. Cari Dinamis via API-Football
    if API_FOOTBALL_KEY:
        try:
            encoded_name = urllib.parse.quote(team_name.strip())
            url = f"[https://v3.football.api-sports.io/teams?search=](https://v3.football.api-sports.io/teams?search=){encoded_name}"
            req = urllib.request.Request(
                url,
                headers={
                    "x-apisports-key": API_FOOTBALL_KEY.strip(),
                    "User-Agent": "Mozilla/5.0",
                },
            )

            with urllib.request.urlopen(req, timeout=4) as response:
                data = json.loads(response.read().decode())
                if data and data.get("response") and len(data["response"]) > 0:
                    team_info = data["response"][0].get("team", {})
                    logo_url = team_info.get("logo")
                    if logo_url and logo_url.startswith("http"):
                        return logo_url
        except Exception as e:
            print(f"[API-Football Warning] Gagal mengambil logo {team_name}: {e}")

    # 3. Fallback Terakhir
    encoded_avatar = urllib.parse.quote(team_name.strip())
    return f"[https://ui-avatars.com/api/?name=](https://ui-avatars.com/api/?name=){encoded_avatar}&background=000000&color=ffffff&bold=true"


# ==========================================
# MULTI-TIER AI FAILOVER MECHANISM
# ==========================================
async def call_mistral_fallback(prompt: str) -> str:
    if not MISTRAL_API_KEY or not mistral_client:
        raise Exception("MISTRAL_API_KEY belum terpasang atau client Mistral tidak aktif.")

    mistral_models = ["mistral-small-latest", "open-mistral-7b"]

    for model_name in mistral_models:
        try:
            response = await asyncio.to_thread(
                mistral_client.chat.complete,
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah API pendukung portal berita 'barcainspo®'. KELUARKAN HANYA JSON MURNI.",
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
                completion = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Kamu adalah API pendukung portal berita 'barcainspo®'. KELUARKAN HANYA JSON MURNI.",
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

    return await call_mistral_fallback(prompt)


async def call_gemini_with_fallback(prompt: str) -> str:
    if client and GEMINI_API_KEY:
        gemini_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

        for model_name in gemini_models:
            try:
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

    try:
        return await call_groq_fallback(prompt)
    except Exception as fallback_err:
        raise HTTPException(
            status_code=500,
            detail=f"Semua Provider AI gagal merespons: {str(fallback_err)}",
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
# HELPER FUNCTIONS & REDIS CACHE (CACHE V4)
# ==========================================
def load_cache():
    if not redis_client:
        return None
    try:
        data = redis_client.get("match_cache_v4")
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
        redis_client.set("match_cache_v4", json.dumps(cache_payload))
    except Exception as e:
        print(f"[Redis Save Error] Gagal menyimpan cache: {e}")


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

        # 1. CEK CACHE REDIS (V4)
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
        
        # 2. QUERY TAVILY SEARCH
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

        # 3. PROMPT GEMINI PERKETAT SELEKSI JADWAL TANGGAL TERDEKAT
        prompt = f"""
        Hari ini adalah tanggal {current_date_str}.

        TUGAS UTAMA:
        Cari dan ekstrak 1 PERTANDINGAN TERDEKAT BARCELONA (tim utama pria) yang BELUM dimainkan setelah tanggal {current_date_str}.

        ATURAN KETAT KRONOLOGIS:
        1. WAJIB AMBIL PERTANDINGAN YANG TANGGALNYA PALING AWAL / PALING DEKAT SECARA KRONOLOGIS SETELAH {current_date_str}.
        2. PERIKSA LAGA CHAMPIONS LEAGUE TERDEKAT: Paris Saint-Germain pada tanggal 21 Oktober 2026.
        3. DILARANG MELOMPATI KE REAL MADRID (25 Oktober 2026) KARENA LAGA PARIS SAINT-GERMAIN (21 OKTOBER 2026) DIMAINKAN LEBIH DULU!

        DATA MENTAH PENCARIAN WEB:
        ---
        {raw_text}
        ---

        STRUKTUR JSON MUSTI PERSIS SEPERTI INI:
        {{
            "opponent": "Paris Saint-Germain",
            "opponent_code": "PSG",
            "date": "21 Oktober 2026",
            "time": "02:00 WIB",
            "match_iso": "2026-10-21T02:00:00",
            "competition": "UEFA Champions League",
            "venue": "Away"
        }}
        """

        raw_response = await call_gemini_with_fallback(prompt)
        cleaned_response = clean_json_string(raw_response)
        parsed_data = json.loads(cleaned_response)

        opponent_name = parsed_data.get("opponent", "")

        # Fallback proteksi jika AI gagal parsing
        if not opponent_name or "tidak ditemukan" in opponent_name.lower() or len(opponent_name) > 35:
            parsed_data["opponent"] = "Paris Saint-Germain"
            parsed_data["opponent_code"] = "PSG"
            parsed_data["date"] = "21 Oktober 2026"
            parsed_data["time"] = "02:00 WIB"
            parsed_data["match_iso"] = "2026-10-21T02:00:00"
            parsed_data["competition"] = "UEFA Champions League"
            parsed_data["venue"] = "Away"
            opponent_name = "Paris Saint-Germain"

        # Pastikan kode singkatan tim terisi resmi
        parsed_data["opponent_code"] = get_team_code(opponent_name)

        # Fetch Logo Dinamis & Cepat
        opponent_logo_url = await asyncio.to_thread(
            get_team_logo_dynamic, opponent_name
        )

        parsed_data["barca_logo"] = "[https://crests.football-data.org/81.png](https://crests.football-data.org/81.png)"
        parsed_data["opponent_logo"] = opponent_logo_url

        # 4. LOGIKA SMART CACHE
        try:
            match_iso_str = parsed_data.get("match_iso")
            if match_iso_str:
                kickoff_dt = datetime.fromisoformat(match_iso_str)
            else:
                kickoff_dt = now + timedelta(hours=24)

            match_end_dt = kickoff_dt + timedelta(hours=3)
            hours_until_kickoff = (kickoff_dt - now).total_seconds() / 3600

            if now >= match_end_dt:
                expires_at = now
            elif now >= kickoff_dt - timedelta(hours=3) and now < match_end_dt:
                expires_at = match_end_dt
            elif hours_until_kickoff <= 24:
                expires_at = now + timedelta(hours=6)
            else:
                expires_at = now + timedelta(hours=24)

        except Exception:
            expires_at = now + timedelta(hours=24)

        # Simpan ke Redis Cache V4
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
            except Exception as t_err:
                print(f"⚠️ [Tavily Error]: {t_err}")
                results = []

        if not results:
            return {"success": True, "count": 0, "leads": []}

        raw_text = "\n\n".join([f"Source URL ({item.get('url')}):\n{item.get('content')}" for item in results])

        prompt = f"""
        Hari ini adalah tanggal {current_date_str} (ISO: {today_iso}).
        Kamu adalah Head Editor Berita Utama untuk portal 'barcainspo®'.

        DATA MENTAH DARI X:
        ---
        {raw_text}
        ---

        STRUKTUR JSON MUSTI PERSIS SEPERTI INI:
        {{
            "leads": [
                {{
                    "sourceHandle": "@{clean_handles[0]}",
                    "headline": "Judul berita yang padat",
                    "summary": "Ringkasan poin-poin fakta utama",
                    "rawContent": "Kutipan teks asli",
                    "tweetUrl": "URL sumber asli",
                    "createdAt": "{today_iso}"
                }}
            ]
        }}
        """

        try:
            raw_response = await asyncio.wait_for(call_gemini_with_fallback(prompt), timeout=30.0)
            cleaned_response = clean_json_string(raw_response)
            parsed_data = json.loads(cleaned_response)
            leads_data = parsed_data.get("leads", [])
            if not isinstance(leads_data, list):
                leads_data = []
        except Exception:
            leads_data = []

        return {"success": True, "count": len(leads_data), "leads": leads_data}

    except Exception as e:
        return {"success": False, "count": 0, "leads": []}


# ==========================================
# ENDPOINT KLASEMEN LA LIGA (FOOTBALL-DATA.ORG)
# ==========================================
@app.get("/api/standings")
async def get_la_liga_standings():
    try:
        now = datetime.now()

        # 1. CEK CACHE REDIS
        if redis_client:
            try:
                cached = redis_client.get("standings_fd_v1")
                if cached:
                    cached_data = json.loads(cached) if isinstance(cached, str) else cached
                    expires_at = datetime.fromisoformat(cached_data.get("expires_at"))
                    cached_list = cached_data.get("data", [])
                    
                    if now < expires_at and len(cached_list) > 0:
                        return {
                            "success": True,
                            "cached": True,
                            "data": cached_list,
                        }
            except Exception as e:
                print(f"[Standings Cache Load Error]: {e}")

        # 2. FETCH KE FOOTBALL-DATA.ORG
        if not FOOTBALL_DATA_API_KEY:
            raise HTTPException(
                status_code=500, detail="FOOTBALL_DATA_API_KEY belum terpasang di .env"
            )

        url = "[https://api.football-data.org/v4/competitions/PD/standings](https://api.football-data.org/v4/competitions/PD/standings)"
        req = urllib.request.Request(
            url,
            headers={
                "X-Auth-Token": FOOTBALL_DATA_API_KEY.strip(),
                "User-Agent": "Mozilla/5.0",
            },
        )

        with urllib.request.urlopen(req, timeout=6) as response:
            res_data = json.loads(response.read().decode())

        standings_list = []
        if res_data and res_data.get("standings") and len(res_data["standings"]) > 0:
            table = res_data["standings"][0].get("table", [])
            for item in table:
                team_info = item.get("team", {})
                standings_list.append({
                    "rank": item.get("position"),
                    "team_name": team_info.get("name"),
                    "team_logo": team_info.get("crest"),
                    "played": item.get("playedGames"),
                    "win": item.get("won"),
                    "draw": item.get("draw"),
                    "lose": item.get("lost"),
                    "goals_diff": item.get("goalDifference"),
                    "points": item.get("points"),
                })

        # 3. SIMPAN KE REDIS (30 MENIT)
        if standings_list:
            expires_at = now + timedelta(minutes=30)
            cache_payload = {
                "expires_at": expires_at.isoformat(),
                "data": standings_list,
            }
            if redis_client:
                try:
                    redis_client.set("standings_fd_v1", json.dumps(cache_payload))
                except Exception as e:
                    print(f"[Standings Cache Save Error]: {e}")

        return {"success": True, "cached": False, "data": standings_list}

    except Exception as e:
        print(f"[Football-Data Standings Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))