# LLM Memory - Sistema di Memoria Condivisa per Agenti AI

Sistema di memoria a lungo termine condiviso, interrogabile semanticamente, persistente e riutilizzabile da agenti AI diversi.

## Caratteristiche

- 🔌 **MCP Server**: Accesso tramite Model Context Protocol
- 📝 **Persistenza Markdown**: Salvataggio append-only con YAML frontmatter
- 🔍 **Ricerca Semantica**: Indicizzazione vettoriale con LanceDB
- 🌍 **Multilingua**: Embedding locale con sentence-transformers
- 🤖 **Multi-Agent**: Supporto scopes private/shared/global
- ⚡ **Indicizzazione Ibrida**: Sync (default) o async configurabile

## Installazione

```bash
cd llm-memory
pip install -e .
```

## Configurazione

Crea un file `.env`:

```env
# Directory per i file Markdown
MEMORY_STORAGE_DIR=./memories

# Directory per LanceDB
LANCEDB_DIR=./data/lancedb

# Modello embedding (coerente con llm_context)
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
MCP_MODELS_DIR=.local/models
HF_HOME=.local/models/huggingface
TRANSFORMERS_CACHE=.local/models/huggingface/transformers
SENTENCE_TRANSFORMERS_HOME=.local/models/huggingface/sentence_transformers

# Modalità indicizzazione: sync | async | hybrid
INDEXING_MODE=sync
```

## Avvio Server MCP

```bash
python -m src.mcp_server.server
```

## API MCP

### `memory_write`
Salva una memoria nel sistema.

### `memory_search`
Ricerca semantica nelle memorie.

### `memory_read`
Legge una memoria specifica per ID.

### `memory_list`
Lista memorie per scope/agente.

## Project Structure

```
llm-memory/
├── src/                    # Source code
│   ├── storage/           # Markdown persistence
│   ├── embedding/         # Local embedding service
│   ├── vectordb/          # LanceDB integration
│   ├── indexing/          # Hybrid indexer
│   ├── coordination/      # Multi-agent coordination
│   └── mcp_server/        # MCP server implementation
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── memories/              # Stored memories (created at runtime)
├── data/                  # LanceDB data (created at runtime)
└── logs/                  # Server logs (created at runtime)
```

## Docker Deployment

### Quick Start

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete Docker documentation.

## Startup Scripts

### Windows

- `start_server.bat` - Interactive startup with console output
- `start_server_silent.bat` - Minimized window (ideal for autostart)
- `stop_server.bat` - Stop running server

See [STARTUP_GUIDE.md](STARTUP_GUIDE.md) for autostart configuration.

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Windows startup scripts and autostart
- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Docker deployment guide

## Licenza

MIT
