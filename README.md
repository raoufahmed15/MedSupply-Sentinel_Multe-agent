# MedSupply Sentinel — Streamlit + Groq

نفس الـ multi-agent workflow اللي في النوتبوك، بس بواجهة Streamlit، والـ LLM بقى **Groq API** بدل Mistral-Nemo المحلي.
(الحسابات، الـ validation، والـ human approval لسه deterministic — الـ LLM بيكتب التحليل الشرحي بس.)

## الملفات
- `app.py` — واجهة Streamlit
- `sentinel_core.py` — الـ agents / schemas / LangGraph (متسحبة من النوتبوك) + `GroqLLM`
- `data/` — الداتا (تحمّلها من Kaggle، تحت)

## تشغيل محلي
1. اعمل API key من https://console.groq.com/keys
2. نزّل الداتاست `raouf158/medsupply-sentinel-demo-data` من Kaggle وفكّها جوه فولدر `data/`
   (لازم تلاقي `synthetic/` و `guidelines/` و `config/` — لو جوه فولدر فرعي عادي، بيلاقيها لوحده).
3. ```bash
   pip install -r requirements.txt
   export GROQ_API_KEY=gsk_...        # أو حطه في .streamlit/secrets.toml
   streamlit run app.py
   ```
   من غير key التطبيق بيشتغل في Offline mode (من غير LLM).

## Deploy على Streamlit Community Cloud
1. ارفع الفولدر على GitHub **ومعاه `data/`** (بيانات synthetic بس). ماترفعش `secrets.toml`.
2. share.streamlit.io → New app → اختار `app.py`.
3. Settings → Secrets → حط: `GROQ_API_KEY = "gsk_..."`

## الموديلات
الافتراضي `llama-3.3-70b-versatile` ولو فشل بيجرّب `llama-3.1-8b-instant`. ممكن تكتب أي model id تاني من الـ sidebar
(شوف https://console.groq.com/docs/models — الأسماء بتتغير).
لو Groq مردّش خالص، الـ workflow بيكمّل والنص بيبقى `[LLM unavailable …]`.
