# Agentic Cinema

> **Partner Track: Parallel** — Full-length book-to-movie production studio powered by Gemini, grounded in Parallel Web Search, Google Maps, and BigQuery.

## Project Docs

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full system blueprint — agent network, data flow, tech stack |
| [`docs/PARALLEL_INTEGRATION.md`](docs/PARALLEL_INTEGRATION.md) | Parallel Web Search integration — setup, code, all 4 use sites |
| [`docs/GROUNDING.md`](docs/GROUNDING.md) | All grounding sources — Parallel, Maps, Agent Search, BigQuery |
| [`docs/MOVIE_EXPERT_AGENT.md`](docs/MOVIE_EXPERT_AGENT.md) | Movie Expert prebuilt agent — setup, tools, SQL patterns, extensions |
| [`docs/MULTIMODAL.md`](docs/MULTIMODAL.md) | Multimodal reference — text, PDF, image, video, audio, image-gen, caching |
| [`docs/AUDIO_PRODUCTION.md`](docs/AUDIO_PRODUCTION.md) | Audio production — Lyria 3 music gen, Gemini TTS, sentiment analysis |
| [`docs/AGENT_ENGINE.md`](docs/AGENT_ENGINE.md) | Agent Engine — LangChain, ADK, Live API, 3 deployment methods |

---

# Hackathon Resources Guide

Welcome to the backlot! Use this curated guide to find the documentation, SDKs, and code samples needed to build your agentic workflows and movie studio tools. All links have been verified for accuracy.

## The Challenge

Your mission is to code a functional, production-ready AI agent or multi-agent network—powered by Gemini and Google Cloud Agent Builder—that integrates a Partner Entity's product or MCP server to solve critical bottlenecks across the entertainment and media value chain, specifically targeting the workflows of filmmakers, screenwriters, studio crews, or fans.

---

## 🛠️ Phase 1: Core Frameworks & Environment

Choose your building stage. Whether you want a managed, low-code interface or a fully custom SDK setup, begin here:

- **Managed Setup:** [Gemini Enterprise Agent Platform API Setup](https://cloud.google.com/vertex-ai/docs) — The central control panel for all Google Cloud agent configurations.
- **The Low-Code Path:** [Agent Builder Guide](https://cloud.google.com/dialogflow/cx/docs/concept/agent) — Best for rapid development using playbooks, managed grounding, and out-of-the-box data stores.
- **Developer SDK:** [Gemini Enterprise Agent Platform SDK for Python](https://github.com/googleapis/python-genai) — The client library required for writing custom agent logic, tool calls, and API integrations.
- **Agent Starter Pack:** [Agent Engine Getting Started](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/intro_agent_engine.ipynb) — Initialize your first custom agent backend.
- **Cloud Access & Credits:**
  - Sign up for a no-cost trial at [cloud.google.com/free](https://cloud.google.com/free).
  - Use an existing account and request $100 in credits via [Hackathon Credit Form](https://forms.gle/XPe837tzogh8L5sX6) (approved within 1–5 business days, while supplies last).

---

## 🎬 Phase 2: Action Mechanisms & Data Connectivity (GenMedia Focus)

Your agents need to "do" things (tools) and "know" things (grounding). Use these media-focused resources to process scripts, audio, and visual assets:

### 📄 Script Processing & Document Grounding

- **Document & PDF Analysis:** [Document Processing Guide](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/document_processing.ipynb) — Learn how to parse complex PDF scripts, schedules, or box office reports.
- **RAG with BigQuery:** [RAG Q&A with BigQuery & PDFs](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/rag_qna_with_bq_and_featurestore.ipynb) — Query script databases using LangChain and BigQuery Vector Search.
- **Zero-Config Grounding:** [Agent Builder Data Stores Overview](https://cloud.google.com/vertex-ai/docs/generative-ai/grounding/overview) — Ground your agent apps using Vertex AI Search data stores.

### 🎥 Video Analysis & VFX (Multimodal Storyboarding)

- **Multimodal Introduction:** [Gemini Multimodal Use Cases](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/intro_multimodal_use_cases.ipynb) — Learn to feed text, video, and audio simultaneously to Gemini.
- **Comprehensive Video Transcription:** [Video Transcription Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/video-analysis/multimodal_video_transcription.ipynb) — Generate timestamped transcriptions and speaker annotations from raw video clips.
- **Video Asset Captioning:** [Video Captioning Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/multimodal-data-curation/captioning.ipynb) — Generate detailed metadata and descriptions to make video libraries searchable.
- **Visual VFX Generation:** [Imagen 3 Image Generation Guide](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_3_image_gen.ipynb) — Generate VFX mood boards, concept art, and storyboard panels from simple text prompts.

### 🎵 Audio & Speech Generation (Lyria & Gemini TTS)

- **Audio Generation (Lyria 3):** [Lyria 3 Music Generation Guide](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/music/getting-started/lyria3_music_generation.ipynb) — Generate high-fidelity music clips, soundtracks, or sound effects across various genres.
- **Speech Generation (Gemini TTS):** [Gemini 3.1 Flash TTS Tutorial](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/gemini_3_1_flash_tts.ipynb) — Convert text to expressive speech using prebuilt voices, and configure multi-speaker settings.
- **Multi-Speaker Podcast Generator:** [Multi-Speaker Podcast Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/use-cases/podcast/multi-speaker-podcast.ipynb) — Generate multi-speaker dialogues, script reads, or discussions using Gemini and Speech APIs.
- **Dialogue Sentiment Analysis:** [Multimodal Sentiment Analysis](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/multimodal-sentiment-analysis/intro_to_multimodal_sentiment_analysis.ipynb) — Analyze actor reads or voiceovers by comparing audio tone against script text.

---

## 🤝 Phase 3: Partner Integration & Infrastructure

Connect your agents to external systems, developer tools, and operational metrics.

- [IBM](https://agentic-cinema.devpost.com/details/ibm-resources)
- [Grafana](https://agentic-cinema.devpost.com/details/grafana-resources)
- [Parallel](https://agentic-cinema.devpost.com/details/parallel-resources)
- [Clickhouse](https://agentic-cinema.devpost.com/details/clickhouse-resources)
- [Replit](https://agentic-cinema.devpost.com/details/replit-resources)
  - [Replit credit codes request form](https://forms.gle/pwwvgDvbkgiRpADm6)

---

## 🧠 Phase 4: Reasoning, State, & Logic Hosting

Complex cinematic workflows require state tracking, tool invocation, and managed hosting. We recommend building your agents natively using the Agent Development Kit (ADK) instead of external wrapper libraries:

### 📦 ADK Installation

Install the ADK and Agent Engine client libraries:

```bash
pip install "google-cloud-aiplatform[agent_engines,adk]>=1.101.0"
```

### 🤖 Native ADK & Agent Engine Tutorials

- **ADK Introduction:** [Introduction to Agent Engine](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/intro_agent_engine.ipynb) — Learn how to define Python functions as tools, wrap them in your agent, and serialize configurations natively.
- **Deploying ADK Agents:** [Deploying ADK Agents to Agent Engine](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/tutorial_deploy_your_first_adk_agent_on_agent_engine.ipynb) — Step-by-step instructions to deploy your ADK agent to a serverless Vertex AI environment from a local notebook or from source code.
- **Live API Streaming (Bidirectional Audio):** [Live API on Agent Engine Tutorial](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/tutorial_get_started_with_live_api_on_agent_engine.ipynb) — Connect ADK agents directly to the Gemini Live API WebSockets for low-latency voice interactions (e.g., interactive script rehearsal).
- **API & Tool Integration (VFX Tooling):** [Google Maps Agent Tutorial](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/tutorial_google_maps_agent.ipynb) — A blueprint for building an agent integrated with external tools, APIs, and image/satellite tools.
- **Agentic Database Interaction:** [Model Context Protocol (MCP) Database Toolbox](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/tutorial_mcp_toolbox_for_databases.ipynb) — Ground your agent and query SQL databases securely using the Model Context Protocol.

### 🔌 Agent Tools & Function Calling

- **Introduction to Function Calling:** [Introduction to Function Calling](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb) — Allow your agent to query scheduling APIs, write to databases, or trigger cloud tasks.
- **Forced Agent Actions:** [Forced Function Calling Guide](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/forced_function_calling.ipynb) — Ensure your agent always queries safety checks or permissions before executing a command.
- **Multimodal Tool Triggering:** [Multimodal Function Calling](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/multimodal_function_calling.ipynb) — Make your agent trigger database updates or email alerts directly from visual video cues.

---

## 🚀 Phase 5: Deployment & Safety

Get your production ready for the red carpet and protect the studio assets:

- **Agent Deployment:** [Agent Builder Deployment Guide](https://cloud.google.com/dialogflow/cx/docs/concept/version) — Publish your agent as a web chat interface or a REST endpoint using environments.
- **Logic Hosting:** [Cloud Run Quickstart](https://cloud.google.com/run/docs/quickstarts) — Fast, serverless deployment for custom agent backends, APIs, and tool servers.
- **Studio Secrets:** [Secret Manager Guide](https://cloud.google.com/secret-manager/docs) — Securely store and retrieve API keys for your partner tools.
- **Safety & Guardrails:** [Gemini Safety Settings](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_3_image_gen.ipynb#Safety-Settings) — Configure moderation filters for hate speech, harassment, and unsafe creative outputs.

---

## 🌟 About the Hackathon

Calling all ecosystem architects and autonomous builders! Step onto the studio lot and transform real-world enterprise chaos into a seamless cinematic production with Gemini. Join Google Cloud and our elite, star-studded partner ecosystem for a first-of-its-kind Summer Blockbuster hackathon!

In the era of Agentic AI, writing raw lines of code is just the background noise. The real magic happens when you orchestrate the system. Whether you want to play the visionary **Director** building production-ready autonomous agent networks, the **Technical Producer** connecting secure data pipelines via managed MCP servers, or the **Studio Head** enforcing Cloud IAM security and governance across multi-agent workflows—the backlot is yours.

Vibe code with Gemini Enterprise and our leading partner technologies to cast the perfect enterprise tech stack and build the next generation of AI applications fit for the big screen.

### Why Join?

- **Direct Your AI Crew:** Get hands-on with Gemini Enterprise Agent Platform and cutting-edge partner tools to transition your local ideas into production-scale architectures.
- **Walk the Red Carpet:** Pitch your automated production workflows directly to executive judges, enterprise experts, and elite engineering leaders.

---

## 🏆 The Box Office Prizes

There are three identical prize buckets—one for each of our featured partner tracks. Instead of a single, crowded pool, you are competing directly within the specific "studio track" of your chosen partner technology.

### How to Secure the Greenlight

1. **Cast Your Co-Star:** Choose the partner platform that holds the data or application workflows your agent needs to interact with.
2. **Build with Gemini Enterprise:** Build your agentic workflow on Google Cloud using the Gemini Enterprise Agent Platform. Connect to your partner tool using robust enterprise pipelines, API frameworks, or managed protocol adapters.
3. **Dominate Your Track:** Show off a deterministic, multi-step agent that solves enterprise friction. You will be judged exclusively against other builders inside your chosen partner track.

---

## 📋 Requirements

### What to Build

🎬 Build a functional agent—powered by Gemini and Google Cloud Agent Builder—that integrates a Partner Entity's product or MCP to power a real media & entertainment workflow.

**Partners:** IBM · Grafana · Parallel · Clickhouse · Replit

### Submission Requirements

- Include a URL to the hosted project
- **The 3-Minute Trailer (Demo Video):** A demo video showing your project/agent functioning as built — not a cinematic trailer. Upload to YouTube or Vimeo, make it publicly visible, and include English subtitles if not in English.
- Include a URL to your open-source code repository (GitHub, GitLab, or Bitbucket) containing all source code, assets, and instructions needed to run. Must demonstrate actual runtime use of Google Cloud and your chosen Partner's service (imported and called in code, not just named in the README).
- The repository must be public and include a complete open source license file, detectable and visible at the top of the repository page.
- Select which partner track you'll be submitting to.
- Complete your Devpost submission form.
- Read the full Official Rules for detailed partner track requirements and eligibility terms.
