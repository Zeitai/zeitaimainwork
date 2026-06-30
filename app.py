import os
import json
import logging
from flask import Flask, render_template, request
from flask_sock import Sock
from groq import Groq
from deepgram import DeepgramClient
from dotenv import load_dotenv

# -------------------------------------------------
# 1️⃣ Load environment variables
# -------------------------------------------------
load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not DG_KEY:
    raise RuntimeError("❌ DEEPGRAM_API_KEY not found in .env")
if not GROQ_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found in .env")

# -------------------------------------------------
# 2️⃣ Initialise services
# -------------------------------------------------
app = Flask(__name__, template_folder="templates")
sock = Sock(app)

dg_client = DeepgramClient(api_key=DG_KEY)
groq_client = Groq(api_key=GROQ_KEY)

# -------------------------------------------------
# 3️⃣ Flask route – serves the HTML page
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------------------------
# 4️⃣ WebSocket endpoint
# -------------------------------------------------
@sock.route("/api/voice-stream")
def voice_stream(ws):
    clinic = request.args.get("clinic", "our clinic").strip()
    context = request.args.get("context", "General Medical Practice").strip()
    app.logger.info(f"🟢 New WS connection – clinic='{clinic}', context='{context}'")

    is_new_session = True

    # High-intelligence behavior prompt for structured voice appointment booking
    SYSTEM_PROMPT = (
        f"You are a crisp, warm, and highly efficient female AI receptionist for '{clinic}' ({context}).\n\n"
        f"YOUR JOB RULES:\n"
        f"1. Keep responses ultra-short and clear (maximum 1 sentence, around 10-15 words). Voice calls must stay fast!\n"
        f"2. Your main goal is to figure out if they want to book an appointment or need general information.\n"
        f"3. IF THEY ARE BOOKING AN APPOINTMENT, you must collect these 3 items ONE BY ONE. Never ask for multiple items together:\n"
        f"   - Patient Name\n"
        f"   - Desired Date\n"
        f"   - Preferred Time\n"
        f"4. Once you collect all three, summarize them briefly to confirm the booking.\n"
        f"5. Avoid conversational fluff. Speak concisely, guiding the conversation with clear, single questions."
    )

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            raw = ws.receive()
            if raw is None:
                app.logger.info("🔴 WS client disconnected")
                break

            payload = json.loads(raw)
            user_text = payload.get("text", "").strip()
            if not user_text:
                continue

            app.logger.info(f"🗣️ Received: {user_text}")

            # -------------------------------------------------
            # 5️⃣ Ask Groq for a dynamic AI reply
            # -------------------------------------------------
            try:
                if is_new_session:
                    # Clear, immediate, guiding prompt for the user
                    reply = f"Hello! Thanks for calling {clinic}. Are you looking to book an appointment, or can I help you with some info?"
                    is_new_session = False
                    conversation_history.append({"role": "user", "content": user_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                else:
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=conversation_history
                    )
                    reply = response.choices[0].message.content.strip()
                    conversation_history.append({"role": "assistant", "content": reply})
                
                app.logger.info(f"🤖 AI Reply: {reply}")
            except Exception as groq_err:
                app.logger.error(f"Groq error: {groq_err}")
                reply = "I'm sorry, let's try that again. What was your request?"

            # -------------------------------------------------
            # 6️⃣ Call Deepgram TTS (Clean Female Voice)
            # -------------------------------------------------
            try:
                dg_response = dg_client.speak.v1.audio.generate(
                    text=reply,
                    model="aura-asteria-en",
                    encoding="linear16",
                    sample_rate=24000
                )
            except Exception as dg_err:
                app.logger.error(f"Deepgram error: {dg_err}")
                ws.send(json.dumps({"error": "TTS failed"}))
                continue

            audio_bytes = b"".join(dg_response)
            if not audio_bytes:
                app.logger.warning("Deepgram returned empty audio")
                continue

            # -------------------------------------------------
            # 7️⃣ Send the raw PCM back as a binary frame
            # -------------------------------------------------
            ws.send(audio_bytes)

        except (json.JSONDecodeError, KeyError) as parse_err:
            app.logger.error(f"Bad payload from client: {parse_err}")
            ws.send(json.dumps({"error": "invalid payload"}))
        except Exception as e:
            app.logger.exception(f"Unexpected WS error: {e}")
            ws.send(json.dumps({"error": str(e)}))
            break

    return ""

# -------------------------------------------------
# 8️⃣ Run the dev server
# -------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app.run(host="0.0.0.0", port=5000, debug=True)