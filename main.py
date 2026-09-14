import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

APP_NAME = "RpsGBT AI Backend"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
API_KEY = os.getenv("OPENAI_API_KEY")
APP_TOKEN = os.getenv("RPSGBT_APP_TOKEN", "")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=API_KEY)

app = FastAPI(title=APP_NAME, version="1.0.0")

# The Android app does not use browser CORS, but enabling it makes testing easier.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """أنت RpsGBT، مساعد متخصص في Minecraft Java Edition 1.21.
أجب بالعربية الواضحة، واستخدم أسماء Minecraft الإنجليزية بين قوسين عند الحاجة.

عند تحليل latest.log أو crash-report:
- حدد السبب الرئيسي للكراش أولاً.
- فرّق بين السبب الحقيقي والأخطاء الثانوية.
- اذكر الدليل من اللوق، مع اقتباس قصير فقط عند الحاجة.
- حدّد المود أو الـdependency أو إصدار Java/Minecraft إذا كان واضحاً.
- أعط خطوات إصلاح مرتبة.
- إذا لم يكن الدليل كافياً، قل إن التشخيص غير مؤكد ولا تخمّن.

عند سؤال عنصر أو بلوك:
- أعط الاسم العربي والإنجليزي وID إذا كنت واثقاً.
- طريقة الحصول.
- وصفة الـCrafting إن وجدت.
- العوالم/الأبعاد التي يوجد فيها طبيعياً.
- الـStructures ذات الصلة إن وجدت.

عند سؤال Enchantments:
- اختر فقط التطويرات المتوافقة مع العنصر.
- أعط 5 خيارات متنوعة عندما يكون ذلك ممكناً.
- اشرح فائدة كل تطوير بالعربية بشكل بسيط.
"""

class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=250_000)

class AnalyzeResponse(BaseModel):
    answer: str
    model: str

def check_token(x_rpsgbt_token: Optional[str]) -> None:
    # Optional protection. Set RPSGBT_APP_TOKEN on the server to require it.
    if APP_TOKEN and x_rpsgbt_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid app token.")

@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "model": MODEL}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, x_rpsgbt_token: Optional[str] = Header(default=None)):
    check_token(x_rpsgbt_token)

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=req.prompt,
        )
        return AnalyzeResponse(
            answer=response.output_text,
            model=MODEL,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI provider error: {type(exc).__name__}",
        )

@app.get("/")
def root():
    return {"service": APP_NAME, "status": "running"}
