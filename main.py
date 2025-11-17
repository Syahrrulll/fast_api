import os
import re
import json
import uuid
import httpx
import uvicorn
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ======================================================================
# CONFIG
# ======================================================================
app = FastAPI(title="Literise AI Service", version="1.0")

# Ambil API key dari env (ubah di OS / Vercel); ada default untuk testing lokal
CHUTES_API_KEY = os.getenv(
    "CHUTES_API_KEY",
    "cpk_49d03a0e918f44c5b753d8aefa411eb0.0140b8ee2e8c5bfbae7e6bc921a677ba.VYnSymDVRjdpY53MK4NduBfyff9RKdoD"
)

# Endpoint Chutes (tanpa ?key=)
CHUTES_API_URL = "https://llm.chutes.ai/v1/chat/completions"

# Model yang dipakai (pastikan valid di dashboard Chutes)
MODEL_NAME = "moonshotai/Kimi-K2-Instruct-0905"

# In-memory cache (ephemeral; serverless tidak persist)
GAME_CACHE: Dict[str, Dict[str, Any]] = {}

# ======================================================================
# HELPERS: call chat-style API (per-request HTTP client)
# ======================================================================
async def call_ai_chat(messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
    Kirim request chat-style (OpenAI-like) ke Chutes.
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    """
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json"
    }

    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(CHUTES_API_URL, json=payload, headers=headers)
        # For debugging you can uncomment:
        # print("AI STATUS:", resp.status_code)
        # print("AI RESPONSE:", resp.text)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # try to return human-friendly error
            try:
                err_json = resp.json()
                detail = err_json.get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=f"AI provider error: {detail}")
        return resp.json()

async def call_ai_json(system_prompt: str, user_prompt: str, expect_json: bool = True, max_tokens: Optional[int] = None) -> Any:
    """
    Kirim system + user via chat, lalu ambil content (text) dari AI.
    Jika expect_json True -> coba parse return content ke JSON.
    Kembalikan parsed object (dict/list) jika berhasil, atau raise HTTPException.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    resp = await call_ai_chat(messages, max_tokens=max_tokens)

    # Extract content robustly
    content_text = None
    try:
        first_choice = resp.get("choices", [])[0]
        if not first_choice:
            raise Exception("No choices in AI response.")
        # chat format
        if "message" in first_choice and isinstance(first_choice["message"], dict):
            content_text = first_choice["message"].get("content")
        # older or text field
        elif "text" in first_choice:
            content_text = first_choice.get("text")
        else:
            # try to stringify whole response
            content_text = json.dumps(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI response format unexpected: {e}")

    if content_text is None:
        raise HTTPException(status_code=500, detail="AI returned empty content.")

    # Try to parse JSON if expected
    if expect_json:
        try:
            parsed = json.loads(content_text)
            return parsed
        except json.JSONDecodeError:
            # If AI returned JSON-like with trailing text, try to extract JSON substring
            m = re.search(r'(\{.*\}|\[.*\])', content_text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"AI did not return valid JSON. Raw: {content_text[:500]}")
    else:
        return content_text

# ======================================================================
# Simple chat page (GET form + POST)
# ======================================================================
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Chat Kimmi</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 24px; background:#f6f7fb; }
            .card { max-width:800px; margin:auto; background:white; padding:20px; border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.08); }
            textarea { width:100%; height:140px; padding:10px; border-radius:8px; border:1px solid #ddd; resize:vertical; }
            button { padding:10px 16px; border-radius:8px; background:#4f46e5; color:white; border:none; cursor:pointer; }
            pre { background:#f3f4ff; padding:12px; border-radius:8px; white-space:pre-wrap; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Chat dengan Kimmi (test)</h2>
          <form action="/chat" method="post">
            <textarea name="message" placeholder="Ketik pesan..."></textarea><br/><br/>
            <button type="submit">Kirim</button>
          </form>
        </div>
      </body>
    </html>
    """

from fastapi import Form

@app.post("/chat", response_class=HTMLResponse)
async def chat_page_post(message: str = Form(...)):
    # send as simple user message; you can include system prompt to set persona
    system_prompt = "Kamu adalah Kimmi, asisten ramah yang membantu dengan singkat dan jelas."
    try:
        # we expect plain text back
        reply = await call_ai_json(system_prompt=system_prompt, user_prompt=message, expect_json=False, max_tokens=400)
    except HTTPException as e:
        # show error on page
        return HTMLResponse(f"<h3>Error memanggil AI:</h3><pre>{e.detail}</pre><a href='/chat'>Kembali</a>")
    return HTMLResponse(f"""
    <html>
      <body style="font-family: Arial; padding:20px;">
        <h2>Jawaban AI:</h2>
        <pre>{reply}</pre>
        <br/><a href="/chat">Kembali</a>
      </body>
    </html>
    """)

# ======================================================================
# Pydantic models (request validation)
# ======================================================================
class SearchTopicRequest(BaseModel):
    topic: str = Field(..., example="Efek Pemanasan Global")

class QuizSubmitRequest(BaseModel):
    answers: List[Dict[str, str]] = Field(..., example=[{"question": "Q1", "answer": "A1"}])

class HoaxCheckRequest(BaseModel):
    mission_id: str
    user_choice: str = Field(..., example="Hoax")

class LibraryGenerateRequest(BaseModel):
    format: str = Field(..., example="Cerpen")
    genre: str = Field(..., example="Fantasy")

class LibraryQuizSubmitRequest(BaseModel):
    user_answers: List[str]

class GrammarGenerateRequest(BaseModel):
    genre: str = Field(..., example="Slice of Life")

class GrammarSubmitRequest(BaseModel):
    user_corrections: List[str]

# ======================================================================
# Endpoint: generate reading mission (refactored -> use call_ai_json)
# ======================================================================
@app.post("/api/game/generate-mission")
async def generate_reading_mission(request: SearchTopicRequest):
    topic = request.topic
    mission_id = str(uuid.uuid4())

    system_prompt = (
        "Anda adalah asisten edukasi untuk platform literasi bernama Literise. "
        "Buat artikel singkat sekitar 150-200 kata, lalu buat tepat 3 pertanyaan pemahaman dan jawaban ideal. "
        "Kembalikan hasil sebagai JSON object dengan keys: reading_text, quiz_questions (array of strings), correct_answers (array of strings). "
        "JANGAN gunakan Markdown."
    )
    user_prompt = f"Topik: {topic}"

    try:
        data = await call_ai_json(system_prompt=system_prompt, user_prompt=user_prompt, expect_json=True, max_tokens=800)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat misi dari AI. Coba lagi. Detail: {e.detail}")

    # validate fields
    if not all(k in data for k in ("reading_text", "quiz_questions", "correct_answers")):
        raise HTTPException(status_code=500, detail="AI tidak mengembalikan field yang diperlukan.")

    # store in cache
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

# ======================================================================
# Endpoint: validate reading mission quiz
# ======================================================================
@app.post("/api/game/validate-quiz/{mission_id}")
async def validate_reading_mission_quiz(mission_id: str, request: QuizSubmitRequest):
    if mission_id not in GAME_CACHE:
        raise HTTPException(status_code=404, detail="Misi tidak ditemukan atau sudah kedaluwarsa.")

    cached = GAME_CACHE[mission_id]
    correct_answers = cached["answers"]
    user_answers = [a["answer"] for a in request.answers]
    questions = cached["questions"]

    if len(user_answers) != len(correct_answers):
        raise HTTPException(status_code=400, detail="Jumlah jawaban tidak sesuai.")

    system_prompt = (
        "Anda adalah seorang guru yang menilai kuis pemahaman. "
        "Bandingkan setiap jawaban pengguna dengan jawaban ideal. "
        "Kembalikan JSON: { results: [ {question, user_answer, score, feedback} ], total_score }"
    )

    user_prompt_parts = [f"Konteks Misi: {cached['title']}"]
    for i in range(len(questions)):
        user_prompt_parts.append(f"Pertanyaan {i+1}: {questions[i]}")
        user_prompt_parts.append(f"Jawaban Ideal {i+1}: {correct_answers[i]}")
        user_prompt_parts.append(f"Jawaban Pengguna {i+1}: {user_answers[i]}")
    user_prompt = "\n".join(user_prompt_parts)

    try:
        data = await call_ai_json(system_prompt=system_prompt, user_prompt=user_prompt, expect_json=True, max_tokens=800)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Gagal menilai kuis: {e.detail}")

    # cleanup cache
    del GAME_CACHE[mission_id]
    return {
        "title": cached["title"],
        "total_score": data.get("total_score", 0),
        "results": data.get("results", [])
    }

# ======================================================================
# Endpoint: Hoax quiz generate + check
# ======================================================================
@app.get("/api/hoax-quiz/generate")
async def generate_hoax_quiz():
    mission_id = str(uuid.uuid4())
    system_prompt = (
        "Anda adalah pembuat kuis literasi media. Buat satu skenario berita viral (2-3 kalimat), "
        "tunjukkan apakah itu hoax (true/false), berikan penjelasan singkat, dan source_url atau 'N/A'. "
        "Return JSON with keys: news_snippet, is_hoax, explanation, source_url."
    )
    user_prompt = "Buat satu skenario kuis 'Hoax or Not?'"

    try:
        data = await call_ai_json(system_prompt=system_prompt, user_prompt=user_prompt, expect_json=True, max_tokens=400)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat kuis Hoax: {e.detail}")

    GAME_CACHE[mission_id] = {
        "is_hoax": data.get("is_hoax", False),
        "explanation": data.get("explanation", ""),
        "source_url": data.get("source_url", "N/A")
    }

    return {"mission_id": mission_id, "news_snippet": data.get("news_snippet", "")}

@app.post("/api/hoax-quiz/check")
async def check_hoax_answer(request: HoaxCheckRequest):
    mission_id = request.mission_id
    user_choice_str = request.user_choice.strip().lower()

    if mission_id not in GAME_CACHE:
        raise HTTPException(status_code=404, detail="Kuis tidak ditemukan atau sudah kedaluwarsa.")

    cached = GAME_CACHE[mission_id]
    correct_bool = cached["is_hoax"]
    correct_str = "hoax" if correct_bool else "fakta"
    is_correct = (user_choice_str == correct_str)
    del GAME_CACHE[mission_id]

    return {
        "is_correct": is_correct,
        "correct_answer": correct_str.capitalize(),
        "explanation": cached["explanation"],
        "source_url": cached["source_url"]
    }

# ======================================================================
# Endpoint: Library Hub (generate full text)
# ======================================================================
@app.post("/api/library/generate-full-text")
async def generate_library_full_text(request: LibraryGenerateRequest):
    game_id = str(uuid.uuid4())
    system_prompt = (
        "Anda adalah penulis. Buat full_text sekitar 150-200 kata sesuai format dan genre, "
        "dan berikan array 'blanks' tepat 5 kata penting dari teks. Return JSON with keys: full_text, blanks."
    )
    user_prompt = f"Format: {request.format}, Genre: {request.genre}"

    try:
        data = await call_ai_json(system_prompt=system_prompt, user_prompt=user_prompt, expect_json=True, max_tokens=800)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat teks Library: {e.detail}")

    # verify blanks exist
    full_text = data.get("full_text", "")
    blanks = data.get("blanks", [])
    if not full_text or not isinstance(blanks, list) or len(blanks) == 0:
        raise HTTPException(status_code=500, detail="AI tidak mengembalikan teks atau kata kunci yang valid.")

    full_lower = full_text.lower()
    verified = []
    for b in blanks:
        cb = str(b).strip()
        if cb and cb.lower().rstrip(".,?!") in full_lower:
            verified.append(cb)

    if not verified:
        raise HTTPException(status_code=500, detail="AI gagal membuat kata kunci valid untuk teks ini.")

    GAME_CACHE[game_id] = {"full_text": full_text, "correct_answers": verified}
    return {"game_id": game_id, "full_text": full_text, "title": f"{request.format} ({request.genre})"}

@app.get("/api/library/get-quiz-text/{game_id}")
async def get_library_quiz_text(game_id: str):
    if game_id not in GAME_CACHE or "correct_answers" not in GAME_CACHE[game_id]:
        raise HTTPException(status_code=404, detail="Game tidak ditemukan atau data tidak valid.")

    cached = GAME_CACHE[game_id]
    text = cached["full_text"]
    answers = cached["correct_answers"]
    placeholder = "[.....]"

    # replace case-insensitive first occurrence per answer
    for w in answers:
        pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
        text, n = pattern.subn(placeholder, text, count=1)
    return {"game_id": game_id, "text_with_blanks": text, "total_questions": len(answers)}

@app.post("/api/library/validate-blanks/{game_id}")
async def validate_library_blanks(game_id: str, request: LibraryQuizSubmitRequest):
    if game_id not in GAME_CACHE or "correct_answers" not in GAME_CACHE[game_id]:
        raise HTTPException(status_code=404, detail="Game tidak ditemukan atau jawaban tidak valid.")
    cached = GAME_CACHE[game_id]
    correct = cached["correct_answers"]
    user_answers = request.user_answers

    if len(user_answers) != len(correct):
        raise HTTPException(status_code=400, detail="Jumlah jawaban tidak sesuai.")

    results = []
    total = 0
    per = 100 / len(correct)
    for i in range(len(correct)):
        ok = user_answers[i].strip().lower() == correct[i].strip().lower()
        score = per if ok else 0
        total += score
        results.append({"blank_index": i+1, "user_answer": user_answers[i], "correct_answer": correct[i], "is_correct": ok})

    del GAME_CACHE[game_id]
    return {"total_score": round(total), "results": results, "full_text": cached["full_text"]}

# ======================================================================
# Endpoint: Grammar Zone
# ======================================================================
@app.post("/api/grammar-zone/generate-game")
async def generate_grammar_game(request: GrammarGenerateRequest):
    game_id = str(uuid.uuid4())
    system_prompt = (
        "Anda pembuat kuis tata bahasa. Buat tepat 5 kalimat (campuran benar/salah). "
        "Return JSON: { sentences_to_fix: [...], correct_sentences: [...] }"
    )
    user_prompt = f"Genre: {request.genre}"
    try:
        data = await call_ai_json(system_prompt=system_prompt, user_prompt=user_prompt, expect_json=True, max_tokens=600)
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat game Tata Bahasa: {e.detail}")

    GAME_CACHE[game_id] = {"correct_sentences": data["correct_sentences"], "original_sentences": data["sentences_to_fix"]}
    return {"game_id": game_id, "genre": request.genre, "sentences_to_fix": data["sentences_to_fix"]}

@app.post("/api/grammar-zone/submit-game/{game_id}")
async def submit_grammar_game(game_id: str, request: GrammarSubmitRequest):
    if game_id not in GAME_CACHE or "correct_sentences" not in GAME_CACHE[game_id]:
        raise HTTPException(status_code=404, detail="Game tidak ditemukan atau data tidak valid.")

    cached = GAME_CACHE[game_id]
    correct_sentences = cached["correct_sentences"]
    original_sentences = cached["original_sentences"]
    user_corrections = request.user_corrections

    if len(user_corrections) != len(correct_sentences):
        raise HTTPException(status_code=400, detail="Jumlah jawaban tidak sesuai.")

    results = []
    total = 0
    per = 100 / len(correct_sentences)
    for i in range(len(correct_sentences)):
        ok = user_corrections[i].strip().lower() == correct_sentences[i].strip().lower()
        score = per if ok else 0
        total += score
        results.append({"original": original_sentences[i], "user_correction": user_corrections[i], "correct_sentence": correct_sentences[i], "is_correct": ok})

    del GAME_CACHE[game_id]
    return {"total_score": round(total), "results": results}

# ======================================================================
# Run (for local dev)
# ======================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
