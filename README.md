# llm-memory (Tiered, Local-Only MCP Memory)

Memoria RAG locale per MCP con architettura a tier, governance e audit trail. Nessun servizio cloud richiesto.

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
    |        +--> LanceVectorStore (legacy/optional)
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

# Governance
DEDUP_HASH_ENABLED=true
DEDUP_SEMANTIC_ENABLED=true
DEDUP_SEMANTIC_THRESHOLD=0.97
PROMOTION_TARGET_TIER=tier-3
```

## Tool MCP (v2)

- `memory.add`
- `memory.search`
- `memory.get`
- `memory.invalidate`
- `memory.promote`
- `memory.reembed`
- `memory.export`
- `memory.import`

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
2. non hardcodare chiavi nel codice
3. fare backup separato e protetto della chiave

Nota: la cifratura usa `cryptography` (`Fernet`) se disponibile localmente.

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
