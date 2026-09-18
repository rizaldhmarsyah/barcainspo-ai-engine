import os
import json
import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai

# 1. Muat API Key dari .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 2. Inisialisasi Client
client = genai.Client(api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

def get_next_match_data():
    # Search data jadwal Barcelona lewat Tavily
    search_result = tavily.search(query="FC Barcelona next match fixture schedule", max_results=3)
    results = search_result.get("results", [])
    
    # Rapikan dengan Pandas
    df = pd.DataFrame(results)
    raw_text = "\n".join(df["content"].tolist())
    
    prompt = f"""
    Kamu adalah asisten statistik FC Barcelona. 
    Berdasarkan data pencarian berikut:
    
    {raw_text}
    
    Ekstrak informasi pertandingan terdekat berikutnya dan kembalikan HANYA format JSON murni tanpa teks/markdown tambahan dengan struktur:
    {{
        "opponent": "Nama Lawan",
        "date": "Tanggal Pertandingan",
        "competition": "Nama Kompetisi (La Liga/UCL/dll)",
        "venue": "Home/Away"
    }}
    """
    
    # Memanggil Gemini 2.5 / 1.5 via SDK genai terbaru
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    
    # Bersihkan output JSON
    try:
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        return {"error": "Gagal merespon format JSON", "raw": response.text}

if __name__ == "__main__":
    print("Memproses data pertandingan lewat AI Agent...")
    match_info = get_next_match_data()
    print("\n--- HASIL EKSTRAKSI GEMINI (JSON) ---")
    print(json.dumps(match_info, indent=2))