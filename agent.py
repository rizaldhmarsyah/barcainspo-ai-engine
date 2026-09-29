#agent.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from groq import Groq

# 1. Muat API Key dari .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Inisialisasi Client
client = genai.Client(api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def call_groq_fallback(prompt: str) -> str:
    """Fallback ke model Groq aktif."""
    if not GROQ_API_KEY or not groq_client:
        raise Exception("GROQ_API_KEY belum terpasang atau tidak valid di file .env")

    groq_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b"
    ]

    for model_name in groq_models:
        try:
            print(f"[Groq Fallback] Memproses request dengan model: {model_name}...")
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah asisten statistik FC Barcelona. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks (```json)."
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


def get_next_match_data():
    if not tavily:
        print("[Tavily Warning] API key Tavily tidak ditemukan.")
        results = []
    else:
        print("[Tavily] Mengambil jadwal pertandingan...")
        search_result = tavily.search(query="FC Barcelona next match fixture schedule 2026", max_results=3)
        results = search_result.get("results", [])
    
    df = pd.DataFrame(results)
    raw_text = "\n".join(df["content"].tolist()) if not df.empty else "Tidak ada data pencarian."
    
    prompt = f"""
    Kamu adalah asisten statistik FC Barcelona. 
    Berdasarkan data pencarian berikut:
    
    {raw_text}
    
    Ekstrak informasi pertandingan terdekat berikutnya dan kembalikan HANYA format JSON murni dengan struktur:
    {{
        "opponent": "Nama Lawan",
        "date": "Tanggal Pertandingan",
        "competition": "Nama Kompetisi (La Liga/UCL/dll)",
        "venue": "Home/Away"
    }}
    """
    
    gemini_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    for model_name in gemini_models:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[Gemini] Memanggil {model_name} (Percobaan {attempt + 1})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                err_msg = str(e)
                print(f"[Gemini Warning] {model_name} percobaan {attempt + 1} gagal: {err_msg}")
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue

    print("[Gemini Failed] Semua model Gemini sibuk/error. Dialihkan ke Groq...")
    try:
        groq_response = call_groq_fallback(prompt)
        return json.loads(groq_response)
    except Exception as groq_err:
        print(f"[Groq Error] Gagal memproses via Groq: {groq_err}")
        return {"error": f"Semua Provider AI (Gemini & Groq) gagal merespons: {str(groq_err)}"}


if __name__ == "__main__":
    print("Memproses data pertandingan lewat AI Agent...")
    match_info = get_next_match_data()
    print("\n--- HASIL EKSTRAKSI (JSON) ---")
    print(json.dumps(match_info, indent=2))