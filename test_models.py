import asyncio
import os
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

mistral_client = None
if MISTRAL_API_KEY:
    try:
        from mistralai import Mistral

        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except ImportError:
        print(
            "⚠️ Package 'mistralai' belum diinstall. Install dulu via: pip install mistralai"
        )

TEST_PROMPT = 'Keluarkan JSON murni: {"status": "ok", "message": "Model aktif!"}'
TIMEOUT_SECONDS = 10.0  # Batas waktu maksimal per pengujian (10 detik)


async def test_gemini():
    print("\n--- 1. TESTING GEMINI MODELS ---")
    if not GEMINI_API_KEY or not gemini_client:
        print("❌ GEMINI_API_KEY tidak ditemukan di .env")
        return

    gemini_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    for model in gemini_models:
        try:
            print(f"Menguji Gemini ({model})...")
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model,
                    contents=TEST_PROMPT,
                    config={"response_mime_type": "application/json"},
                ),
                timeout=TIMEOUT_SECONDS,
            )
            print(f"✅ Gemini ({model}) BERHASIL!")
            print(f"   Response: {response.text.strip()}\n")
        except asyncio.TimeoutError:
            print(
                f"❌ Gemini ({model}) GAGAL: Timeout (tidak merespons dalam {TIMEOUT_SECONDS} detik)\n"
            )
        except Exception as e:
            print(f"❌ Gemini ({model}) GAGAL: {e}\n")


async def test_groq():
    print("\n--- 2. TESTING GROQ ACTIVE MODELS ---")
    if not GROQ_API_KEY or not groq_client:
        print("❌ GROQ_API_KEY tidak ditemukan di .env")
        return

    groq_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "allam-2-7b",
    ]

    for model in groq_models:
        try:
            print(f"Menguji Groq ({model})...")
            completion = await asyncio.wait_for(
                asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": "Keluarkan JSON murni."},
                        {"role": "user", "content": TEST_PROMPT},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                ),
                timeout=TIMEOUT_SECONDS,
            )
            print(f"✅ Groq ({model}) BERHASIL!")
            print(f"   Response: {completion.choices[0].message.content.strip()}\n")
        except asyncio.TimeoutError:
            print(
                f"❌ Groq ({model}) GAGAL: Timeout (tidak merespons dalam {TIMEOUT_SECONDS} detik)\n"
            )
        except Exception as e:
            print(f"❌ Groq ({model}) GAGAL: {e}\n")


async def test_mistral():
    print("\n--- 3. TESTING MISTRAL MODELS (FALLBACK TERAKHIR) ---")
    if not MISTRAL_API_KEY:
        print(
            "❌ MISTRAL_API_KEY tidak ditemukan di .env! Melewati tes Mistral."
        )
        return

    if not mistral_client:
        print(
            "❌ Mistral Client tidak terinisialisasi. Silakan install dengan: pip install mistralai"
        )
        return

    mistral_models = ["mistral-small-latest", "open-mistral-7b"]

    for model in mistral_models:
        try:
            print(f"Menguji Mistral ({model})...")
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    mistral_client.chat.complete,
                    model=model,
                    messages=[
                        {"role": "system", "content": "Keluarkan JSON murni."},
                        {"role": "user", "content": TEST_PROMPT},
                    ],
                    response_format={"type": "json_object"},
                ),
                timeout=TIMEOUT_SECONDS,
            )
            print(f"✅ Mistral ({model}) BERHASIL!")
            print(f"   Response: {response.choices[0].message.content.strip()}\n")
        except asyncio.TimeoutError:
            print(
                f"❌ Mistral ({model}) GAGAL: Timeout (tidak merespons dalam {TIMEOUT_SECONDS} detik)\n"
            )
        except Exception as e:
            print(f"❌ Mistral ({model}) GAGAL: {e}\n")


async def main():
    print("🚀 Memulai Pengujian Berantai Model AI (Gemini -> Groq -> Mistral)...")
    await test_gemini()
    await test_groq()
    await test_mistral()


if __name__ == "__main__":
    asyncio.run(main())