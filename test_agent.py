import os
import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient

# 1. Muat API Key dari file .env
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("=== 1. CEK KONFIGURASI API KEY ===")
print("Gemini API Key :", "TERSEDIA" if gemini_key else "TIDAK ADA")
print("Tavily API Key :", "TERSEDIA" if tavily_key else "TIDAK ADA")

# 2. Uji Coba Web Search Agent dengan Tavily
print("\n=== 2. MENJALANKAN SEARCH AGENT (TAVILY) ===")
try:
    tavily = TavilyClient(api_key=tavily_key)
    # Melakukan pencarian data jadwal Barcelona
    response = tavily.search(query="FC Barcelona next match schedule", max_results=3)
    
    # Extract data hasil pencarian
    search_results = response.get("results", [])
    
    # 3. Olah Hasil Search Menggunakan Pandas Dataframe
    print("\n=== 3. MENGOLAH DATA HASIL SEARCH DENGAN PANDAS ===")
    data_list = []
    for item in search_results:
        data_list.append({
            "Judul Sumber": item.get("title"),
            "URL": item.get("url"),
            "Ringkasan Teks": item.get("content")[:100] + "..." # Ambil 100 karakter awal
        })
    
    # Konversi list dictionary ke Pandas DataFrame (Tabel)
    df = pd.DataFrame(data_list)
    print(df.to_string())

except Exception as e:
    print("Terjadi error saat menjalankan agent:", e)