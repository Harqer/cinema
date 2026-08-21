# Agentic Cinema: System Architecture Blueprint

## Mission

Convert a full-length book into a production-ready movie — automatically. The system ingests a book (PDF/text), extracts story structure, generates a complete screenplay, grounds every creative decision in real-world data (locations, film history, current web), generates audio assets (score, dialogue), visual assets (storyboards, concept art), and assembles a full production package ready for human review.

---

## High-Level Agent Network

```
┌─────────────────────────────────────────────────────────────┐
│                    STUDIO ORCHESTRATOR                       │
│              (ADK Agent Engine — Master Agent)               │
└────────────────┬──────────────────────────────┬─────────────┘
                 │                              │
    ┌────────────▼────────────┐   ┌─────────────▼────────────┐
    │   DEVELOPMENT DEPT      │   │   PRODUCTION DEPT         │
    │  (Pre-production tools) │   │  (Asset generation tools) │
    └────────────┬────────────┘   └─────────────┬────────────┘
                 │                              │
   ┌─────────────┼─────────────┐   ┌────────────┼────────────┐
   ▼             ▼             ▼   ▼            ▼            ▼
Script      Location       Research  Audio     Visual    Distribution
Agent       Agent          Agent     Agent     Agent     Agent
```

---

## Core Data Flow

```
[Book PDF / Source Text]
        │
        ▼
[Document Processing Agent]
  - Entity extraction (characters, locations, themes)
  - Chapter segmentation
  - Story arc identification
        │
        ▼
[Script Development Agent]
  - Three-act structure generation
  - Scene-by-scene breakdown
  - Dialogue drafting (Gemini TTS-ready)
  - Grounded by: Parallel Web Search (film writing conventions, comparable films)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[Location Scouting Agent]           [Research & Continuity Agent]
  - Grounded by: Google Maps           - Grounded by: Parallel Web Search
  - Real location suggestions          - Historical accuracy checks
  - Routing between shoot days         - Cultural/technical fact verification
  - Place properties (hours, access)   - BigQuery movie metadata lookups
        │                                      │
        └──────────────────┬───────────────────┘
                           ▼
               [Production Design Agent]
                 - Storyboard panels (Imagen 3)
                 - VFX mood boards (Imagen 3)
                 - Scene-by-scene visual brief
                           │
                           ▼
               [Audio Production Agent]
                 - Score generation (Lyria 3)
                 - Dialogue synthesis (Gemini TTS)
                 - Multi-speaker read-throughs
                 - Sentiment analysis on dialogue
                           │
                           ▼
               [Final Package Assembly]
                 - Structured production bible (JSON + PDF)
                 - Shot list with location data
                 - Asset manifest
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent Framework | Google ADK + Agent Engine | Orchestration, state, tool hosting |
| Foundation Model | Gemini 3.6 Flash / 2.5 Pro | Reasoning, generation |
| Document Ingestion | `google-genai` SDK + `pypdf` | Book parsing, entity extraction |
| Vector Store (proto) | BigQuery Vector Search + LangChain | Scene/chapter semantic search |
| Vector Store (prod) | Vertex AI Feature Store | Low-latency retrieval at serve time |
| Grounding: Web | **Parallel Web Search** | Current film data, fact verification |
| Grounding: Geo | Google Maps Tool | Location scouting, routing |
| Grounding: Docs | Agent Search Data Store | Wikipedia plot summaries, film metadata |
| Structured Data | BigQuery (`movies_data`, `credits`) | Revenue, budget, cast queries via `bq-search` |
| Image Generation | Imagen 3 | Storyboards, concept art, VFX references |
| Audio Generation | Lyria 3 | Scores, soundtracks |
| TTS | Gemini 3.1 Flash TTS | Dialogue, multi-speaker read-throughs |
| Deployment | Cloud Run + Agent Engine | Serverless hosting |
| Secrets | Secret Manager | API keys (Parallel, Maps, etc.) |

---

## Agent Definitions (ADK)

Each agent is a Python function-as-tool wrapped in ADK. Install:

```bash
pip install "google-cloud-aiplatform[agent_engines,adk]>=1.101.0"
```

### Master Orchestrator

```python
from google.adk import Agent

studio_orchestrator = Agent(
    name="studio_orchestrator",
    model="gemini-2.5-pro",
    tools=[
        script_development_tool,
        location_scout_tool,
        research_tool,
        production_design_tool,
        audio_production_tool,
        bq_search_tool,         # OpenAPI → BigQuery
        cymbal_plots_tool,      # Data Store → Wikipedia plot summaries
    ],
    instruction="""
    You are the Studio Head of Agentic Cinema.
    Given a book PDF or source text, coordinate all departments
    to produce a complete production-ready movie package.
    Always ground creative decisions in real-world data.
    """,
)
```

---

## Data Stores

### BigQuery Tables (auto-created by Agent Builder)

| Table | Contents |
|---|---|
| `movies_data` | Title, revenue, budget, release date, language, genre |
| `credits` | Cast, crew, director, writer per film |

### Agent Search Data Store

- **Name:** `Movie Expert-data-store`
- **Source:** Cloud Storage bucket — Wikipedia movie plot summaries
- **Tool:** `cymbal-movie-plots` (Data Store tool in Agent Builder)

### Vector Store (for book/script chunks)

- **Prototype:** `BigQueryVectorStore` — zero setup, batch retrieval
- **Production:** `VertexFSVectorStore` — low-latency online serving

```python
from langchain_google_community import BigQueryVectorStore

bq_store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    location=LOCATION,
    dataset_name="cinema_production",
    table_name="script_chunks",
    embedding=VertexAIEmbeddings(model_name="text-embedding-005"),
)
```

---

## Grounding Strategy

| Decision Type | Grounding Source |
|---|---|
| Film industry conventions, comparable films | Parallel Web Search |
| Real filming locations, travel between them | Google Maps |
| Plot summaries, Wikipedia film data | Agent Search Data Store |
| Box office, budget, cast metadata | BigQuery `bq-search` tool |
| Historical/cultural accuracy | Parallel Web Search (`advanced` mode) |

See [`docs/GROUNDING.md`](GROUNDING.md) for full grounding implementation details.
See [`docs/PARALLEL_INTEGRATION.md`](PARALLEL_INTEGRATION.md) for the Parallel track integration.

---

## Deployment

```bash
# Deploy to Agent Engine (serverless Vertex AI)
gcloud ai agent-engines deploy \
  --project=$PROJECT_ID \
  --location=us-central1 \
  --agent-config=agent_config.yaml

# Or via Cloud Run for custom backends
gcloud run deploy cinema-studio \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Hackathon Track

This project submits to the **Parallel track**. The Parallel web search grounding
is a first-class integration, not decorative — it is used in:

1. Script development (live film market research)
2. Research & continuity checks (fact verification against current web)
3. Distribution analysis (comparable films, box office trends)

See [`docs/PARALLEL_INTEGRATION.md`](PARALLEL_INTEGRATION.md) for the full integration plan.
