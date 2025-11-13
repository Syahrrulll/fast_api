import os
import httpx
import uvicorn
import uuid
import re  # <-- IMPORT INI
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# ==============================================================================
# KONFIGURASI DAN KUNCI API
# ==============================================================================
app = FastAPI(title="Literise AI Service", version="1.0")

# --- MASUKKAN API KEY ANDA DI SINI ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBztaGYUxg3Mk7oVZWYYoH8euvz8Enh1ZI") # <-- KUNCI BARU ANDA
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"

# Klien HTTP Async untuk performa FastAPI yang lebih baik
client = httpx.AsyncClient(timeout=60.0)

# Penyimpanan Cache Sederhana (Simulasi Database)
GAME_CACHE = {}


# ==============================================================================
# FUNGSI HELPER UTAMA (PEMANGGIL API GEMINI)
# ==============================================================================
async def call_gemini_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fungsi helper untuk memanggil Gemini API dengan penanganan error.
    """
    try:
        response = await client.post(GEMINI_API_URL, json=payload)
        response.raise_for_status()  # Menangani error HTTP (spt 400, 500)

        data = response.json()

        if "candidates" not in data or not data["candidates"]:
            prompt_feedback = data.get('promptFeedback', {})
            block_reason = prompt_feedback.get('blockReason', 'REASON_UNSPECIFIED')
            safety_ratings = prompt_feedback.get('safetyRatings', [])
            raise HTTPException(
                status_code=500,
                detail=f"Error: Konten diblokir oleh Gemini. Alasan: {block_reason}. Detail: {safety_ratings}"
            )

        return data

    except httpx.HTTPStatusError as e:
        error_body = e.response.json()
        error_detail = error_body.get("error", {}).get("message", e.response.text)

        if e.response.status_code == 503: # Overloaded
            raise HTTPException(status_code=503, detail=f"Server AI sedang sibuk (Overloaded). Coba lagi dalam beberapa saat. Detail: {error_detail}")

        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error dari Gemini: {error_detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error internal Python: {str(e)}")

# ==============================================================================
# MODEL DATA (PYDANTIC) UNTUK VALIDASI REQUEST
# ==============================================================================

# --- Fitur 1: Reading Mission ---
class SearchTopicRequest(BaseModel):
    topic: str = Field(..., example="Efek Pemanasan Global")

class QuizSubmitRequest(BaseModel):
    answers: List[Dict[str, str]] = Field(..., example=[{"question": "Q1", "answer": "A1"}])

# --- Fitur 2: Hoax or Not? ---
class HoaxCheckRequest(BaseModel):
    mission_id: str
    user_choice: str = Field(..., example="Hoax")

# --- Fitur 3: Library Hub (Melengkapi Kata) ---
class LibraryGenerateRequest(BaseModel):
    format: str = Field(..., example="Cerpen")
    genre: str = Field(..., example="Fantasy")

class LibraryQuizSubmitRequest(BaseModel):
    user_answers: List[str]

# --- Fitur 4: Zona Tata Bahasa ---
class GrammarGenerateRequest(BaseModel):
    genre: str = Field(..., example="Slice of Life")

class GrammarSubmitRequest(BaseModel):
    user_corrections: List[str]


@app.get("/")
async def root():
    return {"message": "Hello World"}

# ==============================================================================
# API ENDPOINT - FITUR 1: READING MISSION (AI Search)
# ==============================================================================
# ... (Kode tidak berubah) ...
@app.post("/api/game/generate-mission")
async def generate_reading_mission(request: SearchTopicRequest):
    topic = request.topic
    mission_id = str(uuid.uuid4())

    system_prompt = (
        "Anda adalah asisten edukasi untuk platform literasi bernama Literise. "
        "Tugas Anda adalah membuat misi membaca berdasarkan topik yang diminta pengguna. "
        "Anda HARUS menghasilkan dua hal: "
        "1. 'reading_text': Artikel singkat (sekitar 150-200 kata) tentang topik tersebut. Gunakan paragraf (\\n\\n). "
        "2. 'quiz_questions': TEPAT 3 pertanyaan pemahaman (tipe esai singkat) HANYA berdasarkan teks yang Anda tulis. "
        "3. 'correct_answers': Jawaban singkat dan ideal untuk setiap pertanyaan."
        "JANGAN gunakan Markdown (seperti #, *, atau **)."
    )
    user_prompt = f"Topik: {topic}"

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "reading_text": {"type": "STRING"},
                    "quiz_questions": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Tepat 3 pertanyaan"
                    },
                    "correct_answers": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Tepat 3 jawaban"
                    }
                },
                "required": ["reading_text", "quiz_questions", "correct_answers"]
            }
        }
    }

    try:
        response_data = await call_gemini_api(payload)
        generated_data = response_data["candidates"][0]["content"]["parts"][0]["text"]
        import json
        data = json.loads(generated_data)

        GAME_CACHE[mission_id] = {
            "title": topic,
            "questions": data["quiz_questions"],
            "answers": data["correct_answers"]
        }

        return {
            "mission_id": mission_id,
            "title": topic,
            "reading_text": data["reading_text"],
            "quiz_questions": [{"question": q} for q in data["quiz_questions"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses permintaan AI: {str(e)}")

# ... (Kode tidak berubah) ...
@app.post("/api/game/validate-quiz/{mission_id}")
async def validate_reading_mission_quiz(mission_id: str, request: QuizSubmitRequest):
    if mission_id not in GAME_CACHE:
        raise HTTPException(status_code=404, detail="Misi tidak ditemukan atau sudah kedaluwarsa.")

    cached_data = GAME_CACHE[mission_id]
    correct_answers = cached_data["answers"]
    user_answers = [ans["answer"] for ans in request.answers]
    questions = cached_data["questions"]

    if len(user_answers) != len(correct_answers):
        raise HTTPException(status_code=400, detail="Jumlah jawaban tidak sesuai.")

    system_prompt = (
        "Anda adalah seorang guru yang menilai kuis pemahaman. "
        "Bandingkan setiap 'jawaban_pengguna' dengan 'jawaban_ideal'. "
        "Berikan 'skor' (0 hingga 100) dan 'umpan_balik' singkat untuk SETIAP jawaban. "
        "JANGAN tambahkan penjelasan umum, hanya fokus pada daftar hasil."
    )

    prompt_sections = [f"Konteks Misi: {cached_data['title']}"]
    for i in range(len(questions)):
        prompt_sections.append(f"\nPertanyaan {i+1}: {questions[i]}")
        prompt_sections.append(f"Jawaban Ideal {i+1}: {correct_answers[i]}")
        prompt_sections.append(f"Jawaban Pengguna {i+1}: {user_answers[i]}")

    user_prompt = "\n".join(prompt_sections)

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "results": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "question": {"type": "STRING"},
                                "user_answer": {"type": "STRING"},
                                "score": {"type": "NUMBER"},
                                "feedback": {"type": "STRING"}
                            },
                            "required": ["question", "user_answer", "score", "feedback"]
                        }
                    },
                    "total_score": {"type": "NUMBER"}
                },
                "required": ["results", "total_score"]
            }
        }
    }

    try:
        response_data = await call_gemini_api(payload)
        generated_data = response_data["candidates"][0]["content"]["parts"][0]["text"]
        import json
        data = json.loads(generated_data)
        del GAME_CACHE[mission_id]

        return {
            "title": cached_data["title"],
            "total_score": data["total_score"],
            "results": data["results"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menilai kuis: {str(e)}")


# ==============================================================================
# API ENDPOINT - FITUR 2: HOAX OR NOT?
# ==============================================================================
# ... (Kode tidak berubah) ...
@app.get("/api/hoax-quiz/generate")
async def generate_hoax_quiz():
    mission_id = str(uuid.uuid4())

    system_prompt = (
        "Anda adalah asisten pembuat kuis literasi media. "
        "Tugas Anda adalah membuat SATU skenario berita (nyata atau hoaks) yang viral. "
        "Anda HARUS menghasilkan 4 hal: "
        "1. 'news_snippet': Teks berita (sekitar 2-3 kalimat) seolah-olah viral di media sosial. "
        "2. 'is_hoax': Boolean (true jika hoaks, false jika fakta). "
        "3. 'explanation': Penjelasan logis mengapa ini hoaks atau fakta. "
        "4. 'source_url': URL sumber (jika fakta) atau URL halaman debunk (jika hoaks). Gunakan 'N/A' jika tidak relevan. "
        "Topiknya harus beragam (kesehatan, politik, sains, hiburan)."
        "JANGAN gunakan Markdown."
    )
    user_prompt = "Buatkan saya satu skenario kuis 'Hoax or Not?' baru."

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "news_snippet": {"type": "STRING"},
                    "is_hoax": {"type": "BOOLEAN"},
                    "explanation": {"type": "STRING"},
                    "source_url": {"type": "STRING"}
                },
                "required": ["news_snippet", "is_hoax", "explanation"]
            }
        }
    }

    try:
        response_data = await call_gemini_api(payload)
        generated_data = response_data["candidates"][0]["content"]["parts"][0]["text"]

        import json
        data = json.loads(generated_data)

        GAME_CACHE[mission_id] = {
            "is_hoax": data["is_hoax"],
            "explanation": data["explanation"],
            "source_url": data.get("source_url", "N/A")
        }

        return {
            "mission_id": mission_id,
            "news_snippet": data["news_snippet"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat kuis Hoax: {str(e)}")

# ... (Kode tidak berubah) ...
@app.post("/api/hoax-quiz/check")
async def check_hoax_answer(request: HoaxCheckRequest):
    mission_id = request.mission_id
    user_choice_str = request.user_choice.lower()

    if mission_id not in GAME_CACHE:
        raise HTTPException(status_code=404, detail="Kuis tidak ditemukan atau sudah kedaluwarsa.")

    cached_data = GAME_CACHE[mission_id]
    correct_answer_bool = cached_data["is_hoax"]
    correct_answer_str = "hoax" if correct_answer_bool else "fakta"
    is_correct = (user_choice_str == correct_answer_str)
    del GAME_CACHE[mission_id]

    return {
        "is_correct": is_correct,
        "correct_answer": correct_answer_str.capitalize(),
        "explanation": cached_data["explanation"],
        "source_url": cached_data["source_url"]
    }


# ==============================================================================
# API ENDPOINT - FITUR 3: LIBRARY HUB (Melengkapi Kata)
# ==============================================================================

@app.post("/api/library/generate-full-text")
async def generate_library_full_text(request: LibraryGenerateRequest):
    """
    Membuat Teks Lengkap + Kuis Kata Hilang (tapi disembunyikan).
    """
    game_id = str(uuid.uuid4())

    system_prompt = (
        "Anda adalah seorang pendongeng / penulis artikel. "
        "Tugas Anda adalah membuat DUA hal berdasarkan permintaan pengguna: "
        "1. 'full_text': Teks lengkap (sekitar 150-200 kata) sesuai Format dan Genre. "
        "2. 'blanks': Daftar TEPAT 5 kata penting dari teks tersebut untuk dijadikan kuis melengkapi kata. "
        "JANGAN gunakan Markdown (seperti #, *, atau **)."
    )
    user_prompt = f"Format: {request.format}, Genre: {request.genre}"

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "full_text": {"type": "STRING"},
                    "blanks": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Tepat 5 kata"
                    }
                },
                "required": ["full_text", "blanks"]
            }
        }
    }

    try:
        response_data = await call_gemini_api(payload)
        generated_data = response_data["candidates"][0]["content"]["parts"][0]["text"]

        import json
        data = json.loads(generated_data)

        # --- PERBAIKAN BUG (VERIFIKASI KATA HILANG) ---
        full_text_lower = data["full_text"].lower()
        verified_blanks = []

        # Memastikan jumlah kata kunci tidak lebih dari 5 (jika AI memberi lebih)
        words_to_check = data["blanks"][:5]

        for blank in words_to_check:
            # Cek jika kata (tanpa spasi) benar-benar ada di teks
            # Kita juga cek tanpa tanda baca (sederhana)
            clean_blank_for_check = blank.strip().lower().rstrip(".,?!")

            # Verifikasi bahwa kata yang bersih ada di teks yang bersih
            if clean_blank_for_check and clean_blank_for_check in full_text_lower:
                # Simpan versi asli (dengan huruf besar/tanda baca)
                verified_blanks.append(blank.strip())
            else:
                # Jika kata kunci dari AI tidak ada di teks, log ini
                print(f"WARNING: Kata kunci '{blank}' dari AI tidak ditemukan di teks.")


        # Jika AI gagal total (atau tidak ada kata kunci valid), kita tidak bisa melanjutkan
        if not verified_blanks:
             raise HTTPException(status_code=500, detail="AI gagal membuat kata kunci yang valid untuk teks ini.")
        # --- AKHIR PERBAIKAN ---

        # Simpan data lengkap (termasuk jawaban YANG SUDAH DIVERIFIKASI) di cache
        GAME_CACHE[game_id] = {
            "full_text": data["full_text"],
            "correct_answers": verified_blanks # <-- Gunakan list yang sudah bersih
        }

        # Kirimkan HANYA teks lengkap ke Laravel
        return {
            "game_id": game_id,
            "full_text": data["full_text"],
            "title": f"{request.format} ({request.genre})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat teks Library: {str(e)}")


@app.get("/api/library/get-quiz-text/{game_id}")
async def get_library_quiz_text(game_id: str):
    """
    Mengambil data dari cache dan membuat teks dengan kata yang hilang.
    """
    if game_id not in GAME_CACHE or "correct_answers" not in GAME_CACHE[game_id]:
        raise HTTPException(status_code=404, detail="Game tidak ditemukan atau data tidak valid.")

    cached_data = GAME_CACHE[game_id]
    full_text = cached_data["full_text"]
    answers = cached_data["correct_answers"] # Ini adalah list yang sudah diverifikasi
    expected_blanks = len(answers)

    text_with_blanks = full_text
    placeholder = "[.....]"

    # =================================================================
    # --- AWAL PERBAIKAN: GANTI LOOP DENGAN re.sub (CASE-INSENSITIVE) ---
    # =================================================================

    blanks_created = 0
    for word in answers:
        # Gunakan re.sub untuk replace case-insensitive, HANYA 1x (count=1)
        # re.escape() penting untuk menangani tanda baca dalam kata (jika ada)
        # (misal: "kata." akan di-escape menjadi "kata\.")
        new_text, count = re.subn(
            re.escape(word),
            placeholder,
            text_with_blanks,
            count=1,
            flags=re.IGNORECASE
        )

        if count > 0:
            text_with_blanks = new_text
            blanks_created += 1
        else:
             # Ini bisa terjadi jika kata kunci tumpang tindih
             # (misal: "sangat" dan "sangat baik")
             # Kita log saja, tapi jangan hentikan game
             print(f"WARNING: Kata '{word}' tidak dapat diganti di game {game_id}")

    # =================================================================
    # --- AKHIR PERBAIKAN ---
    # =================================================================

    # Verifikasi Final: Cek apakah jumlah placeholder yang dibuat sama
    actual_blanks_created = text_with_blanks.count(placeholder)

    if actual_blanks_created != expected_blanks:
        # Jika jumlahnya tidak cocok, ini masalah serius.
        # Hapus game yang rusak ini agar tidak bisa disubmit.
        del GAME_CACHE[game_id]
        print(f"FATAL ERROR: Library Game ID {game_id} mismatch.")
        print(f"Expected {expected_blanks} blanks, created {actual_blanks_created}.")
        print(f"Answers: {answers}")
        print(f"Original Text: {full_text}")
        raise HTTPException(status_code=500, detail=f"Gagal membuat kuis: Terjadi ketidakcocokan kata kunci. Silakan coba buat game baru.")

    return {
        "game_id": game_id,
        "text_with_blanks": text_with_blanks,
        "total_questions": len(answers) # Ini adalah jumlah yang DIHARAPKAN
    }


@app.post("/api/library/validate-blanks/{game_id}")
async def validate_library_blanks(game_id: str, request: LibraryQuizSubmitRequest):
    """
    Memeriksa jawaban kuis melengkapi kata.
    """
    if game_id not in GAME_CACHE or "correct_answers" not in GAME_CACHE[game_id]:
        raise HTTPException(status_code=404, detail="Game tidak ditemukan atau jawaban tidak valid.")

    cached_data = GAME_CACHE[game_id]
    correct_answers = cached_data["correct_answers"]
    user_answers = request.user_answers

    # Sekarang cek ini seharusnya LOLOS berkat perbaikan di atas
    if len(user_answers) != len(correct_answers):
        # Jika ini MASIH terjadi, itu berarti logic di Blade (PHP) salah
        # Tapi berdasarkan file Anda, logic Blade sudah benar.
        raise HTTPException(status_code=400, detail=f"Jumlah jawaban tidak sesuai. Diharapkan: {len(correct_answers)}, Diterima: {len(user_answers)}")

    results = []
    total_score = 0
    score_per_item = 100 / len(correct_answers)

    for i in range(len(correct_answers)):
        is_correct = user_answers[i].strip().lower() == correct_answers[i].strip().lower()
        score = score_per_item if is_correct else 0
        total_score += score

        results.append({
            "blank_index": i + 1,
            "user_answer": user_answers[i],
            "correct_answer": correct_answers[i],
            "is_correct": is_correct
        })

    # Hapus game dari cache setelah selesai
    del GAME_CACHE[game_id]

    return {
        "total_score": round(total_score),
        "results": results,
        "full_text": cached_data["full_text"]
    }


# ==============================================================================
# API ENDPOINT - FITUR 4: ZONA TATA BAHASA (Perbaiki Kalimat)
# ==============================================================================
# ... (Kode tidak berubah) ...
@app.post("/api/grammar-zone/generate-game")
async def generate_grammar_game(request: GrammarGenerateRequest):
    game_id = str(uuid.uuid4())

    system_prompt = (
        "Anda adalah asisten pembuat kuis tata bahasa. "
        "Buat TEPAT 5 kalimat berdasarkan Genre yang diminta. "
        "BEBERAPA kalimat harus benar secara tata bahasa, BEBERAPA harus salah (misal: typo, ejaan salah, struktur aneh). "
        "Anda HARUS menghasilkan dua hal: "
        "1. 'sentences_to_fix': Daftar 5 kalimat (campuran benar/salah). "
        "2. 'correct_sentences': Daftar 5 kalimat versi benar/ideal (jika kalimat asli sudah benar, ulangi saja)."
        "JANGAN gunakan Markdown."
    )
    user_prompt = f"Genre: {request.genre}"

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "sentences_to_fix": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "correct_sentences": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                },
                "required": ["sentences_to_fix", "correct_sentences"]
            }
        }
    }

    try:
        response_data = await call_gemini_api(payload)
        generated_data = response_data["candidates"][0]["content"]["parts"][0]["text"]

        import json
        data = json.loads(generated_data)

        GAME_CACHE[game_id] = {
            "correct_sentences": data["correct_sentences"],
            "original_sentences": data["sentences_to_fix"]
        }

        return {
            "game_id": game_id,
            "genre": request.genre,
            "sentences_to_fix": data["sentences_to_fix"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat game Tata Bahasa: {str(e)}")

# ... (Kode tidak berubah) ...
@app.post("/api/grammar-zone/submit-game/{game_id}")
async def submit_grammar_game(game_id: str, request: GrammarSubmitRequest):
    if game_id not in GAME_CACHE or "correct_sentences" not in GAME_CACHE[game_id]:
        raise HTTPException(status_code=404, detail="Game tidak ditemukan atau data tidak valid.")

    cached_data = GAME_CACHE[game_id]
    correct_sentences = cached_data["correct_sentences"]
    original_sentences = cached_data["original_sentences"]
    user_corrections = request.user_corrections

    if len(user_corrections) != len(correct_sentences):
        raise HTTPException(status_code=400, detail="Jumlah jawaban tidak sesuai.")

    results = []
    total_score = 0
    score_per_item = 100 / len(correct_sentences)

    for i in range(len(correct_sentences)):
        is_correct = user_corrections[i].strip().lower() == correct_sentences[i].strip().lower()
        score = score_per_item if is_correct else 0
        total_score += score

        results.append({
            "original": original_sentences[i],
            "user_correction": user_corrections[i],
            "correct_sentence": correct_sentences[i],
            "is_correct": is_correct
        })

    del GAME_CACHE[game_id]

    return {
        "total_score": round(total_score),
        "results": results,
    }


# ==============================================================================
# ENTRY POINT UNTUK MENJALANKAN SERVER (jika dijalankan sebagai script)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
