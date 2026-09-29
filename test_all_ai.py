import os
import time
from dotenv import load_dotenv

# Load API Key dari file .env
load_dotenv()

PROMPT_TEST = "Halo! Balas singkat pesan ini untuk tes koneksi."

def test_gemini():
    print("🤖 Menguji GEMINI (Google GenAI)...", end=" ", flush=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ FAILED\n   Error: GEMINI_API_KEY tidak ditemukan di .env")
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        start_time = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEST,
        )
        elapsed = round(time.time() - start_time, 2)

        print(f"✅ SUCCESS ({elapsed}s)")
        print(f"   Respon: {response.text.strip()}\n")
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}\n")

def test_groq():
    print("⚡ Menguji GROQ...", end=" ", flush=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ FAILED\n   Error: GROQ_API_KEY tidak ditemukan di .env")
        return

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        start_time = time.time()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": PROMPT_TEST}],
            model="llama-3.3-70b-versatile",
        )
        elapsed = round(time.time() - start_time, 2)
        reply = chat_completion.choices[0].message.content.strip()

        print(f"✅ SUCCESS ({elapsed}s)")
        print(f"   Respon: {reply}\n")
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}\n")

def test_mistral():
    print("🌊 Menguji MISTRAL...", end=" ", flush=True)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ FAILED\n   Error: MISTRAL_API_KEY tidak ditemukan di .env")
        return

    try:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)

        start_time = time.time()
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": PROMPT_TEST}],
        )
        elapsed = round(time.time() - start_time, 2)
        reply = chat_response.choices[0].message.content.strip()

        print(f"✅ SUCCESS ({elapsed}s)")
        print(f"   Respon: {reply}\n")
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("    TES RESPONS TERMINAL: GEMINI, GROQ & MISTRAL    ")
    print("=" * 60 + "\n")
    
    test_gemini()
    test_groq()
    test_mistral()

    print("=" * 60)
    print("Selesai pengujian.")