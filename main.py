import asyncio
from datetime import datetime, timedelta
import json
import os
import urllib.parse
import urllib.request
from fastapi import FastAPI, HTTPException
from tavily import TavilyClient
from google import genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from upstash_redis import Redis

load_dotenv()

app = FastAPI(title="Barcainspo AI Engine")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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
tavily = TavilyClient(api_key=TAVILY_API_KEY)

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
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={encoded_name}"
        
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            if data and data.get("teams") and len(data["teams"]) > 0:
                badge_url = data["teams"][0].get("strBadge")
                if badge_url:
                    return badge_url
    except Exception as e:
        print(f"[Logo Fetch Error] Gagal mengambil logo untuk {team_name}: {e}")
    
    encoded_fallback = urllib.parse.quote(team_name[:3].upper())
    return f"https://ui-avatars.com/api/?name={encoded_fallback}&background=262626&color=ffffff&bold=true"

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
        
        # Query dibuat dinamis & spesifik meminta jadwal WIB / Indonesia
        search_query = f"FC Barcelona next match fixture schedule kick off time WIB Indonesia after {current_date_str}"
        
        search_result = await asyncio.to_thread(
            tavily.search,
            query=search_query,
            max_results=5
        )
        
        results = search_result.get("results", [])
        raw_text = "\n\n".join([f"Source ({item.get('url')}):\n{item.get('content')}" for item in results])
        
        prompt = f"""
        Hari ini adalah tanggal {current_date_str} (WIB / UTC+7).
        
        Tugas utama kamu adalah mengekstrak jadwal pertandingan resmi FC Barcelona terdekat berikutnya yang BELUM dimainkan (setelah tanggal {current_date_str}).
        
        Berikut adalah data mentah hasil pencarian:
        ---
        {raw_text}
        ---

        ATURAN WAKTU & ZONA WAKTU (SANGAT PENTING):
        1. Cari jam kick-off dan konversikan secara akurat ke Waktu Indonesia Barat (WIB / UTC+7).
           - Catatan: Waktu lokal Spanyol (CEST) adalah UTC+2. Jadi Waktu WIB = Waktu Spanyol + 5 jam.
           - Contoh: Jika di Spanyol main jam 21:00 CEST tanggal 19 September, maka di WIB adalah jam 02:00 WIB tanggal 20 September.
        2. Format tanggal harus disesuaikan dengan hari di Indonesia (WIB) setelah konversi jam.
        3. Field "time" diisi format jam persis (contoh: "02:00 WIB" atau "21:15 WIB").
        4. Field "match_iso" diisi format ISO 8601 lengkap berdasarkan waktu WIB (contoh: "2026-09-20T02:00:00").
        5. Jika jam belum dikonfirmasi resmi oleh operator liga (TBD), perkirakan jadwal resmi malam Spanyol (biasanya 21:00 CEST / 02:00 WIB hari berikutnya).

        Kembalikan HANYA JSON murni dengan format:
        {{
            "opponent": "Nama Lawan",
            "date": "Tanggal dalam WIB (contoh: 20 September 2026)",
            "time": "Jam WIB (contoh: 02:00 WIB)",
            "match_iso": "YYYY-MM-DDTHH:MM:SS",
            "competition": "Nama Kompetisi (contoh: La Liga / Champions League)",
            "venue": "Home atau Away"
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
        except Exception:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.6-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
        
        parsed_data = json.loads(response.text)
        opponent_name = parsed_data.get("opponent", "Sevilla")
        
        opponent_logo_url = await asyncio.to_thread(get_team_logo_dynamic, opponent_name)
        
        parsed_data["barca_logo"] = "https://images.fotmob.com/image_resources/logo/teamlogo/8634.png"
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