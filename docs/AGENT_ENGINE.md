# Agent Engine — Build, Deploy & Run Agents

Complete reference for building and deploying Agentic Cinema agents to
Vertex AI Agent Engine. Covers LangChain agents, ADK agents, bidirectional
streaming (Live API), and the three deployment methods.

---

## Install

```bash
pip install "google-cloud-aiplatform[agent_engines,langchain,adk]>=1.101.0" \
    cloudpickle==3.0.0 "pydantic>=2.10" requests

# For Live API / audio agents add:
pip install google-adk numpy websockets google-auth
```

---

## Setup

```python
import os, vertexai
from vertexai import agent_engines

PROJECT_ID     = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION       = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = f"gs://{os.environ.get('STAGING_BUCKET', PROJECT_ID)}"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
```

---

## 1. LangChain Agent (quick start)

The fastest path: define Python functions as tools, wrap in `LangchainAgent`,
deploy with `agent_engines.create()`.

```python
from vertexai.preview.reasoning_engines import LangchainAgent

# --- Tools ---
def get_exchange_rate(currency_from: str = "USD",
                      currency_to: str = "EUR",
                      currency_date: str = "latest"):
    """Retrieves the exchange rate between two currencies on a specified date."""
    import requests
    response = requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    )
    return response.json()

# --- Agent ---
agent = LangchainAgent(
    model="gemini-3.6-flash",
    tools=[get_exchange_rate],
    agent_executor_kwargs={"return_intermediate_steps": True},
)

# --- Test locally ---
print(agent.query(input="Exchange rate from USD to SEK today?"))

# Streaming local test
for chunk in agent.stream_query(input="What is 100 USD in JPY?"):
    print(chunk)

# --- Deploy ---
remote_agent = agent_engines.create(
    agent,
    requirements=[
        "google-cloud-aiplatform[agent_engines,langchain]",
        "cloudpickle==3.0.0",
        "pydantic>=2.10",
        "requests",
    ],
)
print(remote_agent.resource_name)

# --- Query deployed agent ---
remote_agent.query(input="What is 100 USD in JPY?")

# --- Reconnect from another environment ---
# remote_agent = agent_engines.get("projects/.../locations/.../reasoningEngines/...")
```

### Model & agent configuration

```python
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:       HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:      HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

model_kwargs = {
    "temperature": 0.3,
    "max_output_tokens": 1000,
    "top_p": 0.95,
    "safety_settings": safety_settings,
}

agent_executor_kwargs = {
    "return_intermediate_steps": True,
    "max_iterations": 5,
    "handle_parsing_errors": True,
    "trim_intermediate_steps": -1,
}
```

---

## 2. ADK Agent — Three Interaction Modes

Every Agent Engine agent has three interaction modes:

| Mode | Method | Pattern |
|---|---|---|
| Request-response | `query()` | One input → one complete output |
| Server streaming | `stream_query()` | One input → stream of chunks |
| Bidirectional streaming | `bidi_stream_query()` | Queue in → async stream of chunks |

```python
import asyncio
from typing import Any, AsyncIterator, Iterator

class CinemaStudioAgent:
    """
    Agent skeleton showing all three interaction modes.
    Copy and extend this for each studio sub-agent.
    """

    def __init__(self, project: str, location: str):
        self.project = project
        self.location = location

    def set_up(self):
        """Heavy initialization — called once when deployed."""
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("CinemaStudioAgent ready")

    # --- Mode 1: request-response ---
    def query(self, input: str) -> dict[str, Any]:
        return {"output": f"Processed: {input}"}

    # --- Mode 2: server-side streaming ---
    def stream_query(self, input: str) -> Iterator[dict[str, Any]]:
        for word in input.split():
            yield {"chunk": word + " "}

    # --- Mode 3: bidirectional streaming ---
    async def bidi_stream_query(
        self, queue: asyncio.Queue
    ) -> AsyncIterator[dict[str, Any]]:
        while True:
            message = await queue.get()
            user_input = message.get("input", "")
            if user_input.lower() in ("exit", "quit"):
                yield {"output": "Session ended."}
                break
            yield {"output": f"Echo: {user_input}"}

    def register_operations(self) -> dict:
        return {
            "":            ["query"],
            "stream":      ["stream_query"],
            "bidi_stream": ["bidi_stream_query"],
        }
```

---

## 3. ADK Agent with Google Search (Express Mode)

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

search_agent = LlmAgent(
    name="cinema_research_agent",
    model="gemini-2.5-flash",
    description="Research agent for film production decisions",
    instruction="Use Google Search for up-to-date information. Always cite sources.",
    tools=[google_search],
)
```

### Deploy via ADK CLI (free 90-day Express Mode)

```bash
# 1. Create project
adk create my_agent --api_key=YOUR_API_KEY

# 2. Deploy
adk deploy agent_engine my_agent

# 3. Query
```

```python
import vertexai

client = vertexai.Client(api_key="YOUR_API_KEY")
agent = client.agent_engines.get(name="projects/.../reasoningEngines/...")

async for item in agent.async_stream_query(
    message="What are the top cinematographers working in sci-fi right now?",
    user_id="director_001",
):
    if "content" in item and item["content"] and "parts" in item["content"]:
        for part in item["content"]["parts"]:
            if "text" in part:
                print(part["text"], end="", flush=True)
```

---

## 4. ADK App Deployment (production, with sessions)

Use `AdkApp` to get managed session state, auto-scaling, and tracing.

```python
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

# 1. Create agent
agent = LlmAgent(
    name="cinema_agent",
    model="gemini-2.5-flash",
    instruction="You are the Studio Head of Agentic Cinema...",
    tools=[google_search],
)

# 2. Wrap in AdkApp with session + memory management
app = AdkApp(
    agent=agent,
    enable_tracing=True,
    session_service_builder=lambda: InMemorySessionService(),
    memory_service_builder=lambda: InMemoryMemoryService(),
)

# 3. Deploy
remote_app = client.agent_engines.create(
    agent=app,
    config=dict(
        display_name="Cinema Studio Agent",
        description="Full film production pipeline agent",
        requirements=["google-cloud-aiplatform[adk,agent_engines]"],
        staging_bucket=STAGING_BUCKET,
    ),
)
print(remote_app.api_resource.name)

# 4. Create session
session = await remote_app.async_create_session(user_id="producer_001")

# 5. Query with session
async for event in remote_app.async_stream_query(
    user_id="producer_001",
    session_id=session["id"],
    message="Analyse the script for locations that require Google Maps grounding.",
):
    if event.get("content", {}).get("parts"):
        for part in event["content"]["parts"]:
            if "text" in part:
                print(part["text"], end="", flush=True)
```

---

## 5. Live API — Bidirectional Audio Agent

For interactive script rehearsal and live voice interactions. Uses WebSockets
directly to the Gemini Live API.

```python
import json, base64, asyncio
import numpy as np
import websockets
import google.auth, google.auth.transport.requests
from typing import AsyncIterator, Any
from IPython.display import Audio, display

class LiveAudioAgent:
    """
    Agent with real-time bidirectional audio via Gemini Live API WebSockets.
    Used for: interactive script rehearsal, live director feedback.
    """

    def __init__(self, project: str, location: str,
                 model_id: str = "gemini-2.0-flash-live-preview-04-09"):
        self.project   = project
        self.location  = location
        self.model_id  = model_id

    def set_up(self):
        host = f"{self.location}-aiplatform.googleapis.com"
        self.service_url = (
            f"wss://{host}/ws/google.cloud.aiplatform.v1."
            "LlmBidiService/BidiGenerateContent"
        )
        self.model = (
            f"projects/{self.project}/locations/{self.location}/"
            f"publishers/google/models/{self.model_id}"
        )
        self.config = {"response_modalities": ["AUDIO"]}

    async def _authenticate(self) -> str:
        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token

    async def _setup_session(self, ws) -> None:
        await ws.send(json.dumps({
            "setup": {"model": self.model, "generation_config": self.config}
        }))
        await ws.recv(decode=False)

    async def _send_text(self, ws, text: str) -> bool:
        if text.lower() in ("exit", "quit"):
            return False
        await ws.send(json.dumps({
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turn_complete": True,
            }
        }))
        return True

    async def _receive_audio(self, ws) -> AsyncIterator[dict]:
        async for raw in ws:
            resp = json.loads(raw.decode())
            sc = resp.get("serverContent", {})
            mt = sc.get("modelTurn", {})
            for part in mt.get("parts", []):
                if "inlineData" in part:
                    pcm = base64.b64decode(part["inlineData"]["data"])
                    yield {"output": np.frombuffer(pcm, dtype=np.int16).tolist()}
            if sc.get("turnComplete"):
                break
        yield {"output": "end_of_turn"}

    async def bidi_stream_query(
        self, input_queue: asyncio.Queue
    ) -> AsyncIterator[dict[str, Any]]:
        token = await self._authenticate()
        async with websockets.asyncio.client.connect(
            self.service_url,
            additional_headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
        ) as ws:
            await self._setup_session(ws)
            while True:
                req = await input_queue.get()
                if not await self._send_text(ws, req["input"]):
                    break
                async for chunk in self._receive_audio(ws):
                    yield chunk

    def register_operations(self) -> dict:
        return {"bidi_stream": ["bidi_stream_query"]}


# --- Deploy ---
live_agent = LiveAudioAgent(project=PROJECT_ID, location=LOCATION)

remote_live = client.agent_engines.create(
    agent=live_agent,
    config={
        "display_name": "Cinema Live Audio Agent",
        "requirements": ["numpy", "google-auth", "websockets"],
        "staging_bucket": STAGING_BUCKET,
    },
)

# --- Interactive audio chat ---
async def audio_chat(agent_name: str):
    async with client.aio.live.agent_engines.connect(
        agent_engine=agent_name,
        config={"class_method": "bidi_stream_query"}
    ) as session:
        print("🎤 Live audio ready — type 'exit' to quit")
        while True:
            text = input("You: ")
            if text.lower() == "exit":
                await session.send({"input": text})
                break
            await session.send({"input": text})
            chunks = []
            while True:
                resp = await session.receive()
                out = resp["bidiStreamOutput"]["output"]
                if out == "end_of_turn":
                    break
                chunks.append(np.array(out))
            if chunks:
                display(Audio(np.concatenate(chunks), rate=24000, autoplay=True))

await audio_chat(remote_live.api_resource.name)
```

---

## 6. Inline Source Deployment (CI/CD)

Deploy directly from source files — no in-memory objects, works in pipelines.

```python
class_methods = [
    {
        "name":        "async_stream_query",
        "api_mode":    "async_stream",
        "description": "Stream responses from the agent",
        "parameters":  {},
    },
    {
        "name":        "async_create_session",
        "api_mode":    "async",
        "description": "Create a new session",
        "parameters":  {},
    },
]

inline_agent = client.agent_engines.create(
    config={
        "display_name":     "Cinema Studio Agent (Source)",
        "source_packages":  ["cinema_agent", "deployment", "requirements.txt"],
        "entrypoint_module": "deployment.deploy",
        "entrypoint_object": "adk_app",
        "class_methods":    class_methods,
    },
)
```

---

## Deployment Method Decision Tree

```
Starting a new agent?
│
├── Just experimenting / no GCP billing yet?
│   └── ► Express Mode (adk create + adk deploy agent_engine)
│          Free 90 days, API key only
│
├── Interactive notebook development?
│   └── ► ADK App deployment (client.agent_engines.create(agent=AdkApp(...)))
│          Needs project + staging bucket
│
├── CI/CD pipeline / automated deploys?
│   └── ► Inline Source deployment
│          No staging bucket needed, pure source files
│
└── Need real-time audio (script rehearsal / live director voice)?
    └── ► LiveAudioAgent with bidi_stream_query
           WebSocket → Gemini Live API
```

---

## Studio Agent Deployment Plan

| Agent | Class | Mode | Model |
|---|---|---|---|
| Studio Orchestrator | `AdkApp(LlmAgent)` | streaming | `gemini-2.5-pro` |
| Script Development | `LangchainAgent` | streaming | `gemini-3.6-flash` |
| Location Scout | `LangchainAgent` + Maps tool | query | `gemini-2.5-flash` |
| Research & Continuity | `LangchainAgent` + Parallel tool | streaming | `gemini-2.5-flash` |
| Production Design | `LangchainAgent` + Imagen tool | query | `gemini-3-pro-image` |
| Audio Production | `LangchainAgent` + Lyria/TTS tools | query | `gemini-3.6-flash` |
| Live Rehearsal | `LiveAudioAgent` | bidi_stream | `gemini-2.0-flash-live` |

---

## Cleaning Up

```python
# Delete a deployed agent to avoid charges
remote_agent.delete(force=True)
print("Agent deleted.")
```
