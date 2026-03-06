# llm-memory (Tiered, Local-Only MCP Memory)

Memoria RAG locale per MCP con architettura a tier, governance e audit trail. Nessun servizio cloud richiesto.
Questo MCP e' dedicato a memorie operative persistenti (non al retrieval di contesto codice repository).

## Obiettivi implementati

- Tiering: `tier-1` (sessione), `tier-2` (progetto), `tier-3` (long-term curato)
- Storage locale modulare: default `SQLite` (metadata + embeddings), backend vettoriale swappabile
- Embedding provider swappabile con versioning e `reembed` incrementale/resumable
- Sicurezza/privacy local-first: blocco rete outbound, privacy policy, cifratura opzionale payload sensibili
- Governance: dedup hash+semantico, promotion, invalidation con trail audit
- API MCP versionata: nuovi tool `memory.*` + wrapper legacy (`memory_write/search/read/list`)
- Import/export deterministico `memory.md` + JSONL + dump SQLite locale

## Architettura

```text
MCP Tools (v2 + v1 compat)
    |
    v
MemoryService (governance/orchestrazione)
    |-- PrivacyPolicy / Cipher / NetworkGuard
    |-- Dedup + Promotion + Invalidation + Audit
    |-- Retrieval Ranking (similarity+recency+tier+status)
    |
    +--> Metadata Store interface
    |        +--> SQLiteMemoryStore (default)
    |
    +--> Vector Store interface
    |        +--> SQLiteVectorStore (default)
    |        +--> LanceVectorStore (legacy/optional, equality-only filters hardened)
    |
    +--> EmbeddingProvider interface
             +--> hash-local (default offline hard-safe)
             +--> sentence-transformers (local_files_only)
```

## Schema memoria canonico (`memory.md`)

Parser/renderer in `src/interop/memory_markdown.py`.

Sezioni canoniche:

- `PURPOSE`
- `STABLE_FACTS` -> `fact`
- `ASSUMPTIONS` -> `assumption`
- `OPEN_UNKNOWNs` -> `unknown`
- `DECISIONS` -> `decision`
- `INVALIDATED` -> `invalidated`

La persistenza primaria non dipende dal markdown (SQLite), ma import/export `memory.md` e deterministico.

## Configurazione locale

Copia `.env.example` in `.env` e modifica i parametri:

```env
# Backend
MEMORY_STORAGE_BACKEND=sqlite
MEMORY_VECTOR_BACKEND=sqlite
MEMORY_SQLITE_PATH=./data/memory.db

# Embedding
EMBEDDING_PROVIDER=hash-local
EMBEDDING_MODEL=local-hash-v1
EMBEDDING_DIM=384
# oppure:
# EMBEDDING_PROVIDER=sentence-transformers
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Security
MEMORY_ALLOW_OUTBOUND_NETWORK=false
MEMORY_ENCRYPTION_ENABLED=false
MEMORY_ENCRYPTION_KEY_ENV=MEMORY_ENCRYPTION_KEY

# Privacy policy
MEMORY_PRIVACY_SENSITIVE_TAGS=pii,secret,credential
MEMORY_PRIVACY_DROP_METADATA_KEYS=password,token,secret,api_key
MEMORY_PRIVACY_ENCRYPT_SENSITIVE=false

# Scope defaults
MEMORY_WORKSPACE_ID=default
MEMORY_PROJECT_ID=default

# MCP transport
MCP_MEMORY_HOST=127.0.0.1
MCP_MEMORY_PORT=8767
MCP_MEMORY_SSE_ENABLED=false
MCP_MEMORY_ALLOWED_HOSTS=localhost:*,127.0.0.1:*,[::1]:*
MCP_MEMORY_ALLOWED_ORIGINS=http://localhost:*,http://127.0.0.1:*,https://localhost:*,https://127.0.0.1:*

# Governance
DEDUP_HASH_ENABLED=true
DEDUP_SEMANTIC_ENABLED=true
DEDUP_SEMANTIC_THRESHOLD=0.97
PROMOTION_TARGET_TIER=tier-3
MEMORY_SELF_EVAL_ENFORCED=false
```

`MEMORY_SELF_EVAL_ENFORCED` abilita/disabilita l'enforcement della regola di autovalutazione
in fase di avvio server. Default `false` (modalita sperimentale non-hardened).

## Tool MCP (v2)

- `memory.add`
- `memory.search`
- `memory.get`
- `memory.invalidate`
- `memory.promote`
- `memory.reembed`
- `memory.export`
- `memory.import`

## Autovalutazione memorie (experimental)

`memory.add` arricchisce `metadata` con scoring deterministico:

- gerarchia surprise: `confidence` -> `disagreement` -> `self`
- novelty: `1 - max_similarity(top_k=5)` con fallback sicuro
- inference: normalizzazione su `tool_steps + correction_count + inference_level`
- negative impact boost: `+0.25 * negative_impact`
- `context_hash` deterministico (SHA-256 troncato a 16 char)

Con `MEMORY_SELF_EVAL_ENFORCED=true`, il server richiede in input:

- `writer_model`
- `context_fingerprint` (object)
- `importance` (object con segnali surprise + inference)

Dataset helper (mitigazione feedback loop):

```bash
python scripts/build_finetune_dataset.py --db ./data/memory.db --output ./data/ft_dataset.jsonl
```

Lo script applica:

- filtro novelty (`novelty_score >= 0.2`)
- bucket sampling `60/25/15` (top/mid/low)
- quota minima esterna (`is_external`) configurabile

Compatibilità legacy:

- `memory_write` -> wrapper su `memory.add`
- `memory_search` -> wrapper su `memory.search`
- `memory_read` -> wrapper su `memory.get`
- `memory_list` -> wrapper su list v2

## Reindex / Reembed

Eseguire via tool MCP `memory.reembed` (incrementale):

- crea una nuova `embedding_version` (provider/model/fingerprint/dim)
- ricalcola solo entry senza embedding per quella versione
- ripetibile/ripristinabile (se interrotto, riparte dai mancanti)

## Crittografia chiavi

Quando `MEMORY_ENCRYPTION_ENABLED=true`:

1. definisci chiave locale in env indicata da `MEMORY_ENCRYPTION_KEY_ENV`
2. usa una chiave Fernet valida oppure una passphrase lunga almeno 32 byte
3. non hardcodare chiavi nel codice
4. fare backup separato e protetto della chiave

Nota: la cifratura usa `cryptography` (`Fernet`) se disponibile localmente. Le passphrase vengono derivate con PBKDF2-HMAC-SHA256; le chiavi troppo corte vengono rifiutate.

## Privacy locale best practice

- Taggare dati sensibili in `sensitivity_tags` (`pii`, `secret`, ...)
- Usare `MEMORY_PRIVACY_ENCRYPT_SENSITIVE=true` quando la chiave e gestita
- In alternativa, lasciare `encrypt=false` e redazione automatica (`[REDACTED:SENSITIVE]`)
- Evitare campi sensibili in `metadata` (drop automatico configurabile)

## Migrazione v1 -> v2

Script:

```bash
python scripts/migrate_v1_to_v2.py --source-dir ./memories --workspace default --project default
```

### Rollback

1. fermare server
2. ripristinare backup `memory.db` (o file export sqlite)
3. riavviare server

## Perche SQLite default (vector+metadata locale)

Scelta pragmatica per local-only:

- zero dipendenze infrastrutturali esterne
- ACID + WAL robusto
- backup semplice (singolo file)
- facile migrazione futura (interfacce storage/vector separate)

Per workload maggiori si puo sostituire il backend vettoriale senza cambiare il service layer.
