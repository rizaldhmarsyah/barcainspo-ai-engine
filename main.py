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

load_dotenv()

app = FastAPI(title="Barcainspo AI Engine")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

CACHE_FILE = "match_cache.json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Helper untuk membaca cache lokal
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# Helper untuk menyimpan data ke cache lokal
def save_cache(data, expires_at: datetime):
    cache_payload = {
        "expires_at": expires_at.isoformat(),
        "data": data
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_payload, f, indent=2)

# Helper pengambilan logo dinamis dari TheSportsDB
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

        # 1. CEK CACHE: Jika cache ada dan waktu sekarang belum melewati batas expired
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

        # 2. CACHE EXPIRED / TIDAK ADA: AI Bekerja
        current_date_str = now.strftime("%d %B %Y")
        
        search_result = await asyncio.to_thread(
            tavily.search,
            query=f"Sevilla vs Barcelona September 2026 kick off time jam berapa WIB",
            max_results=3
        )
        
        results = search_result.get("results", [])
        raw_text = "\n".join([item.get("content", "") for item in results])
        
        prompt = f"""
        Hari ini adalah tanggal {current_date_str}.
        Berdasarkan data pencarian berikut:
        
        {raw_text}
        
        Tentukan jadwal pertandingan FC Barcelona terdekat berikutnya yang BELUM DIMAINKAN (setelah tanggal {current_date_str}).
        
        PENTING UNTUK WAKTU:
        - Ekstrak jam kick-off pertandingan dan format ke WIB (contoh: '02:00 WIB').
        - Jika hasil pencarian tidak secara eksplisit menyebutkan angka jam di teksnya, gunakan waktu standar kick-off La Liga yaitu 02:00 WIB.
        - Tambahkan ISO Timestamp standar untuk jam kickoff (YYYY-MM-DDTHH:MM:SS) pada field "match_iso".
        
        Kembalikan data dalam struktur JSON murni:
        {{
            "opponent": "Nama Lawan",
            "date": "Tanggal Pertandingan (contoh: 20 September 2026)",
            "time": "Jam Kick-off WIB (contoh: 02:00 WIB)",
            "match_iso": "YYYY-MM-DDTHH:MM:SS (contoh: 2026-09-20T02:00:00)",
            "competition": "Nama Kompetisi",
            "venue": "Home/Away"
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
        
        # Ambil logo tim lawan secara dinamis
        opponent_logo_url = await asyncio.to_thread(get_team_logo_dynamic, opponent_name)
        
        parsed_data["barca_logo"] = "https://images.fotmob.com/image_resources/logo/teamlogo/8634.png"
        parsed_data["opponent_logo"] = opponent_logo_url

        # 3. HITUNG EXPIRATION TIME (Kickoff + 3 Jam)
        try:
            match_iso_str = parsed_data.get("match_iso")
            if match_iso_str:
                kickoff_dt = datetime.fromisoformat(match_iso_str)
            else:
                # Fallback jika Gemini lupa format match_iso: set default 24 jam dari sekarang
                kickoff_dt = now + timedelta(hours=24)
            
            # Waktu expired diset 3 jam setelah kick-off
            expires_at = kickoff_dt + timedelta(hours=3)
        except Exception:
            # Fallback jika parsing tanggal gagal
            expires_at = now + timedelta(hours=12)

        # 4. SIMPAN HASIL KE FILE CACHE
        save_cache(parsed_data, expires_at)
        
        return {
            "success": True,
            "cached": False,
            "data": parsed_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))