# AI Virtual Career Counsellor

A Streamlit AI assistant with two modes: an offline, explainable career recommendation engine and an optional general-purpose LLM chat service. The career profile, recommendations, skill gap, and five-stage roadmap remain available without an API key. General questions use a genuine model when configured; the app never substitutes canned responses when a provider fails.

## Features

- General questions, coding help, explanations, and multi-turn follow-ups through an OpenAI-compatible chat-completions endpoint.
- Streamed assistant responses, bounded context, selectable response styles, conversation export, and answer regeneration.
- Transparent career matches, optional student profile, skill gap analysis, personalized learning roadmap, and career library.
- Clear provider status and actionable errors; API credentials stay on the server.

## Architecture

```text
app.py                         Streamlit UI, session conversation, controls
services/llm_service.py        OpenAI-compatible provider, SSE streaming, retries
services/chatbot.py            Offline career-focused conversation flow
services/recommendation_engine.py  Ranking and skill gap analysis
services/roadmap_generator.py  Career learning plans
utils/data_loader.py           Validated data/careers.json loader
tests/                         Offline unit tests (provider calls mocked)
```

LLM context includes up to 12 recent messages, with each message bounded to 3,000 characters. Conversations live only in the current Streamlit session and are not stored in a database. `LLM_BASE_URL` must identify an OpenAI-compatible API that implements `/chat/completions` and server-sent event streaming.

## Requirements and setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with settings from your chosen provider:

```dotenv
LLM_API_KEY=your-provider-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT=45
LLM_MAX_TOKENS=1200
```

Restart the app after changing `.env`:

```powershell
streamlit run app.py
```

Without `LLM_API_KEY`, offline career guidance still works; general questions explain how to enable the model. API keys and `.env` are excluded from Git. Do not put secrets in frontend code or commit them.

## Provider choices

The default shown above is an example OpenAI model/API configuration. Any provider or local server can be used if it exposes the compatible endpoint and accepts a bearer token. Check that provider's current model names, pricing, and rate limits before use. Local OpenAI-compatible servers can be configured with a local base URL and a placeholder key if the server requires one. This project does not require a paid service.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The unit suite tests career recommendations, conversation follow-ups, skill gaps, roadmap generation, configuration, context bounds, SSE parsing, and provider errors. Provider requests are mocked; these tests do not measure model factual quality or call a paid API.

## Career data

Edit `data/careers.json` to add or update roles and their skills and roadmap. Recommendations are ranked from matching skills, interests, topics, titles, and saved profile details. The offline engine covers a curated set of ten technical careers and is intended as guidance, not a scientific assessment or job-market forecast.

## Current limitations

- Conversation state lasts only for the current Streamlit session; there is no account system or durable database.
- The LLM requires a configured compatible provider. No live web search, citations, document upload/RAG, database persistence, or stop-generation control is included.
- Provider rate limits, model availability, privacy, and cost depend on the selected service. Do not send sensitive personal data unless the provider's terms are appropriate.
- Career coverage and the offline recommendation vocabulary are intentionally limited and hand maintained.
