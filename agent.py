import os
import re
import time
import json
import warnings
import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai
from groq import Groq

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

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


def call_mistral_fallback(prompt: str) -> str:
    """Tier 3: Fallback paling akhir ke Mistral AI."""
    if not MISTRAL_API_KEY or not mistral_client:
        raise Exception("MISTRAL_API_KEY belum terpasang atau client Mistral tidak aktif di .env")

    # Menggunakan model Mistral yang sukses dites
    mistral_models = ["mistral-small-latest", "open-mistral-7b"]

    for model_name in mistral_models:
        try:
            print(f"[Mistral Fallback] Memproses request dengan model: {model_name}...")
            response = mistral_client.chat.complete(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah asisten statistik FC Barcelona. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta.",
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


def call_groq_fallback(prompt: str) -> str:
    """Tier 2: Fallback ke model Groq aktif."""
    if not GROQ_API_KEY or not groq_client:
        print("[Groq Skipped] GROQ_API_KEY belum terpasang.")
    else:
        # Menggunakan daftar model Groq resmi yang sukses dites
        groq_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ]

        for model_name in groq_models:
            try:
                print(f"[Groq Fallback] Memproses request dengan model: {model_name}...")
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Kamu adalah asisten statistik FC Barcelona. KELUARKAN HANYA JSON MURNI yang valid sesuai format yang diminta, tanpa markdown triple backticks."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                if completion and completion.choices[0].message.content:
                    return completion.choices[0].message.content
            except Exception as err:
                print(f"[Groq Warning] Model {model_name} gagal: {err}")

    print("[Groq Failed] Semua model Groq gagal/di-skip. Dialihkan ke Mistral...")
    return call_mistral_fallback(prompt)


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
    
    # Tier 1: Gemini (Model sesuai hasil pengujian)
    if client and GEMINI_API_KEY:
        gemini_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
        for model_name in gemini_models:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"[Gemini] Memanggil {model_name} (Percobaan {attempt + 1})...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    if response and response.text:
                        cleaned = clean_json_string(response.text)
                        return json.loads(cleaned)
                except Exception as e:
                    err_msg = str(e)
                    print(f"[Gemini Warning] {model_name} percobaan {attempt + 1} gagal: {err_msg}")
                    if any(code in err_msg for code in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429"]):
                        if attempt < max_retries - 1:
                            time.sleep(1.5)
                            continue

    # Tier 2 & Tier 3 Fallback
    print("[Gemini Failed] Semua model Gemini sibuk/error. Dialihkan ke Groq...")
    try:
        groq_response = call_groq_fallback(prompt)
        cleaned = clean_json_string(groq_response)
        return json.loads(cleaned)
    except Exception as fallback_err:
        print(f"[AI Chain Error] Seluruh Provider AI gagal: {fallback_err}")
        return {"error": f"Semua Provider AI (Gemini, Groq, & Mistral) gagal merespons: {str(fallback_err)}"}


if __name__ == "__main__":
    print("Memproses data pertandingan lewat AI Agent...")
    match_info = get_next_match_data()
    print("\n--- HASIL EKSTRAKSI (JSON) ---")
    print(json.dumps(match_info, indent=2))