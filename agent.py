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
from mistralai import Mistral

# 1. Muat API Key dari .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# 2. Inisialisasi Client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None


def call_groq_primary(prompt: str) -> str:
    """Primary Generator: Groq dengan Llama 3."""
    if not GROQ_API_KEY or not groq_client:
        raise Exception("GROQ_API_KEY belum terpasang di .env")

    groq_models = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
    for model_name in groq_models:
        try:
            print(f"[Groq Primary] Memproses request dengan model: {model_name}...")
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah asisten statistik FC Barcelona. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks (```json)."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return completion.choices[0].message.content
        except Exception as err:
            print(f"[Groq Warning] Model {model_name} gagal: {err}")

    raise Exception("Seluruh model Groq gagal.")


def call_mistral_fallback(prompt: str) -> str:
    """Fallback 1: Mistral AI."""
    if not MISTRAL_API_KEY or not mistral_client:
        raise Exception("MISTRAL_API_KEY belum terpasang di .env")

    print("[Mistral Fallback] Memproses request dengan mistral-small-latest...")
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "system",
                "content": "Kamu adalah asisten statistik FC Barcelona. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    return response.choices[0].message.content


def call_gemini_fallback(prompt: str) -> str:
    """Fallback 2: Google Gemini."""
    if not client:
        raise Exception("GEMINI_API_KEY belum terpasang di .env")

    gemini_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
    for model_name in gemini_models:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"[Gemini Fallback] Memanggil {model_name} (Percobaan {attempt + 1})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                return response.text
            except Exception as e:
                print(f"[Gemini Warning] {model_name} gagal: {e}")
                time.sleep(1)

    raise Exception("Seluruh model Gemini gagal.")


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
    
    # Tiered execution: Groq -> Mistral -> Gemini
    try:
        raw_res = call_groq_primary(prompt)
        return json.loads(raw_res)
    except Exception as groq_err:
        print(f"[Groq Failed]: {groq_err}. Dialihkan ke Mistral AI...")

    try:
        raw_res = call_mistral_fallback(prompt)
        return json.loads(raw_res)
    except Exception as mistral_err:
        print(f"[Mistral Failed]: {mistral_err}. Dialihkan ke Gemini...")

    try:
        raw_res = call_gemini_fallback(prompt)
        return json.loads(raw_res)
    except Exception as gemini_err:
        print(f"[Gemini Failed]: {gemini_err}")
        return {"error": f"Semua Provider AI (Groq, Mistral, Gemini) gagal merespons."}


if __name__ == "__main__":
    print("Memproses data pertandingan lewat AI Agent...")
    match_info = get_next_match_data()
    print("\n--- HASIL EKSTRAKSI (JSON) ---")
    print(json.dumps(match_info, indent=2))