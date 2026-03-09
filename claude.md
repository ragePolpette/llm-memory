# claude.md — llm-memory

## Project summary

Server MCP (Model Context Protocol) in Python per la **memoria operativa persistente e tiered** degli agenti AI. Architettura local-first: nessun servizio cloud, tutti i dati restano in locale (SQLite + embedding locale).

Funzionalità principali:
- Tiering memorie: `tier-1` (sessione), `tier-2` (progetto), `tier-3` (long-term curato)
- Storage SQLite per metadata + embeddings (backend swappabile)
- Embedding provider swappabile: `hash-local` (default, offline) o `sentence-transformers` (locale)
- Governance: dedup hash+semantico, promotion, invalidation, audit trail
- Privacy: blocco rete outbound, cifratura opzionale payload sensibili, redazione metadati sensibili
- API MCP v2 (`memory.*`) + wrapper legacy v1 (`memory_write/search/read/list`)
- Import/export deterministico `memory.md` + JSONL + dump SQLite
- Autovalutazione memorie (scoring deterministico: novelty, surprise, inference)

Trasporto HTTP su `127.0.0.1:8767`. Runtime in `Binah`.

## Quickstart

```bash
cd llm-memory

# Installa dipendenze
pip install -e .

# Scarica modello embedding (opzionale, solo per sentence-transformers)
python scripts/download_model.py

# Configura env
cp config.local.example.env .env
# Edita .env se necessario (i default sono già funzionali)

# Avvia server MCP (HTTP)
python -m src.mcp_server.http_server
# → http://127.0.0.1:8767

# Oppure in modalità stdio
python -m src.mcp_server.server
```

## Architecture overview

```
MCP Tools v2 (memory.add/search/get/invalidate/promote/reembed/export/import)
    + wrapper v1 (memory_write/search/read/list)
    |
MemoryService (governance, orchestrazione)
    |-- PrivacyPolicy / Cipher / NetworkGuard
    |-- Dedup (hash + semantico)
    |-- Promotion + Invalidation + Audit trail
    |-- Retrieval Ranking (similarity + recency + tier + status)
    |
    +--> MetadataStore → SQLiteMemoryStore (default)
    +--> VectorStore   → SQLiteVectorStore (default) | LanceVectorStore (legacy)
    +--> EmbeddingProvider → hash-local (default) | sentence-transformers (locale)
```

## Key modules / folders

| Percorso | Ruolo |
|---|---|
| `src/mcp_server/server.py` | Entry point stdio MCP |
| `src/mcp_server/http_server.py` | Entry point HTTP MCP |
| `src/service/` | `MemoryService`: orchestrazione governance |
| `src/storage/` | SQLiteMemoryStore (metadata) |
| `src/vectordb/` | SQLiteVectorStore, LanceVectorStore |
| `src/embedding/` | EmbeddingProvider: hash-local, sentence-transformers |
| `src/security/` | NetworkGuard, Cipher, PrivacyPolicy |
| `src/interop/memory_markdown.py` | Parser/renderer `memory.md` canonico |
| `src/coordination/` | Dedup, promotion, invalidation, audit |
| `src/models.py` | Modelli dati Pydantic |
| `src/config.py` | Configurazione da env |
| `data/memory.db` | SQLite principale (non tracciato) |
| `memories/` | File `memory.md` (import/export) |
| `scripts/` | `download_model.py`, `build_finetune_dataset.py`, `migrate_v1_to_v2.py` |
| `tests/` | Test pytest (async) |
| `docker-compose.yml` | Avvio containerizzato |
| `Dockerfile` | Build immagine Docker |

## Dependencies & tooling

- **Python**: ≥3.11
- **MCP**: `fastmcp>=2.0.0`
- **HTTP server**: `uvicorn>=0.30.0`, `starlette>=0.37.0`
- **Vector DB**: `lancedb>=0.5.0`, `pyarrow>=14.0.0` (legacy)
- **Embedding**: `sentence-transformers>=2.2.0` (opzionale, local_files_only)
- **Crittografia**: `cryptography` (Fernet, opzionale)
- **Strutturato logging**: `structlog>=24.0`
- **Dev**: `pytest>=8.0`, `pytest-asyncio>=0.23`, `pytest-cov>=4.0`, `ruff>=0.1.0`
- **Build**: `hatchling`

## Configuration

File: `.env` (copiare da `config.local.example.env`)

| Variabile | Default | Note |
|---|---|---|
| `MEMORY_STORAGE_BACKEND` | `sqlite` | |
| `MEMORY_VECTOR_BACKEND` | `sqlite` | |
| `MEMORY_SQLITE_PATH` | `./data/memory.db` | |
| `EMBEDDING_PROVIDER` | `hash-local` | `hash-local` / `sentence-transformers` |
| `EMBEDDING_MODEL` | `local-hash-v1` | Con ST: `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIM` | `384` | |
| `MEMORY_ALLOW_OUTBOUND_NETWORK` | `false` | Blocco rete outbound |
| `MEMORY_ENCRYPTION_ENABLED` | `false` | Fernet se `true` |
| `MEMORY_ENCRYPTION_KEY_ENV` | `MEMORY_ENCRYPTION_KEY` | Nome env var della chiave |
| `MEMORY_PRIVACY_SENSITIVE_TAGS` | `pii,secret,credential` | |
| `MEMORY_PRIVACY_DROP_METADATA_KEYS` | `password,token,secret,api_key` | Redazione automatica |
| `MCP_MEMORY_HOST` | `127.0.0.1` | |
| `MCP_MEMORY_PORT` | `8767` | |
| `MCP_MEMORY_SSE_ENABLED` | `false` | |
| `DEDUP_HASH_ENABLED` | `true` | |
| `DEDUP_SEMANTIC_ENABLED` | `true` | |
| `DEDUP_SEMANTIC_THRESHOLD` | `0.97` | |
| `MEMORY_SELF_EVAL_ENFORCED` | `false` | Enforcement autovalutazione (sperimentale) |

## Common commands

```bash
# Avvio server HTTP (DEV)
python -m src.mcp_server.http_server

# Avvio server stdio
python -m src.mcp_server.server

# Test
pytest tests/ -v
pytest tests/test_storage.py -v

# Migrazione v1 → v2
python scripts/migrate_v1_to_v2.py --source-dir ./memories --workspace default --project default

# Build dataset fine-tuning
python scripts/build_finetune_dataset.py --db ./data/memory.db --output ./data/ft_dataset.jsonl

# Reembed (via tool MCP memory.reembed o da CLI se disponibile)

# Docker
docker-compose up
docker-compose down
```

## Operational notes

- **Porta**: 8767. Non cambiare senza aggiornare `tools/`.
- Deploy: `tools/deploy-mcp-dev-to-deploy.ps1` → copia in `Binah\llm-memory`.
- Rollback: fermare server → ripristinare backup `memory.db` → riavviare.
- Il file `memory.db` è l'unica fonte di verità persistente (markdown è import/export secondario).
- Con `EMBEDDING_PROVIDER=hash-local`: nessuna dipendenza da modelli, avvio istantaneo, ricerca semantica limitata.
- Con `EMBEDDING_PROVIDER=sentence-transformers`: richiede modello scaricato, `local_files_only=True` (no rete in runtime).
- Reembed è incrementale e ripristinabile: se interrotto, riparte dai chunk mancanti.
- Il docker-compose usa `stdin_open: true` + `tty: true` per il trasporto MCP stdio.

## Known issues / risks

- `LanceVectorStore` è legacy/opzionale: il backend default attivo è SQLite.
- `MEMORY_SELF_EVAL_ENFORCED=true` è sperimentale: non usare in produzione senza validazione.
- Se si cambia `EMBEDDING_MODEL` o `EMBEDDING_DIM`, il vecchio indice è incompatibile: eseguire `memory.reembed`.
- `pyproject.toml` lista `lancedb` e `sentence-transformers` come dipendenze obbligatorie anche se i default non le usano: il primo avvio scarica pesi (~500MB) se non presenti. Verificare uso di `local_files_only`.
- Assenza di backup automatico di `memory.db`: fare backup manuale prima di migrazioni.

## Roadmap / next actions

1. **(S)** Rendere `lancedb` e `sentence-transformers` dipendenze opzionali per ridurre footprint installazione base
2. **(M)** Aggiungere backup automatico periodico di `memory.db`
3. **(M)** Validare `MEMORY_SELF_EVAL_ENFORCED=true` in ambiente controllato e documentarne i requisiti
4. **(S)** Aggiungere health check endpoint HTTP (`/health`) per monitoring
5. **(L)** Valutare sostituzione LanceDB con DuckDB+vss per semplificare il dependency tree
