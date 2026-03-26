# llm-memory (Tiered, Local-Only MCP Memory)

Questo progetto e' destinato a vivere come **repository standalone** dentro il workspace DEV `Yetzirah`.
Il runtime locale, quando deployato, resta in `Binah\llm-memory`.

Memoria RAG locale per MCP con architettura a tier, governance e audit trail. Nessun servizio cloud richiesto.
Questo MCP e' dedicato a memorie operative persistenti (non al retrieval di contesto codice repository).

Documenti operativi:

- `QUICKSTART.md`
- `DOCKER_GUIDE.md`
- `CONTRIBUTING.md`
- `docs/QUALITY_GATE.md`
- `docs/RELEASE_CHECKLIST.md`
- `CHANGELOG.md`

## Obiettivi implementati

- Tiering: `tier-1` (sessione), `tier-2` (progetto), `tier-3` (long-term curato)
- Storage locale `SQLite` per metadata e embeddings
- Embedding provider swappabile con versioning e `reembed` incrementale/resumable
- Sicurezza/privacy local-first: blocco rete outbound, privacy policy, cifratura opzionale payload sensibili
- Governance: dedup hash+semantico, promotion, invalidation con trail audit
- API MCP v2 unificata: `memory.*` con governance centralizzata della persistenza
- Discovery progetti e multi-project mode esplicito
- Gerarchia scope: `project` -> `workspace` -> `global`
- Import/export deterministico `memory.md` + JSONL + dump SQLite locale

## Architettura

```text
MCP Tool
   |
   v
MemoryService
   |
   +--> PersistencePolicy (deny-by-default)
   +--> ImportanceScoring / PrivacyPolicy / Cipher / NetworkGuard
   +--> Audit trail write_attempt
   |
   +--> SQLiteMemoryStore
   +--> SQLiteVectorStore
   +--> EmbeddingProvider
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
MEMORY_IMPORT_EXPORT_BASE_DIR=./data/exchange

# Privacy policy
MEMORY_PRIVACY_SENSITIVE_TAGS=pii,secret,credential
MEMORY_PRIVACY_DROP_METADATA_KEYS=password,token,secret,api_key
MEMORY_PRIVACY_ENCRYPT_SENSITIVE=false

# Scope defaults
MEMORY_MULTI_PROJECT_ENABLED=false
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

`MEMORY_MULTI_PROJECT_ENABLED=true` abilita il comportamento esplicito multi-project:

- i progetti diventano entita registrate e interrogabili
- i write/read project-scoped devono dichiarare esplicitamente `scope.project_id`
- gli scope `workspace` e `global` restano disponibili senza fallback impliciti pericolosi

`MEMORY_SELF_EVAL_ENFORCED` abilita/disabilita l'enforcement della regola di autovalutazione
in fase di avvio server. Default `false` (modalita sperimentale non-hardened).

`MEMORY_IMPORT_EXPORT_BASE_DIR` definisce la directory radice per `memory.import` e
`memory.export`. I path richiesti dai tool devono risolversi all'interno di questa base
directory; path esterni vengono rifiutati.

## Runtime Architecture

Il runtime attivo e' solo v2:

- `src/mcp_server/tools.py`
- `src/service/memory_service.py`
- `src/service/persistence_policy.py`
- `src/service/importance_scoring.py`
- `src/security/privacy.py`
- `src/storage/sqlite_store.py`
- `src/vectordb/sqlite_vector_store.py`
- `src/interop/memory_markdown.py` solo per import/export

Non esiste piu un path runtime alternativo basato su Markdown/LanceDB.

## Tool MCP (v2)

Discovery e amministrazione:

- `memory.about`
- `memory.list_projects`
- `memory.get_project_info`
- `memory.create_project`
- `memory.scope_overview`

Memoria operativa:

- `memory.add`
- `memory.search`
- `memory.get`
- `memory.invalidate`
- `memory.promote`
- `memory.reembed`
- `memory.export`
- `memory.import`

## Multi-project e gerarchia scope

Il runtime ora distingue tre bucket logici:

- `project`
  - memoria specifica del progetto corrente
  - priorita di retrieval piu alta
  - in multi-project mode richiede `scope.project_id` esplicito
- `workspace`
  - convenzioni o memorie condivise nello stesso workspace
  - recuperata dopo lo scope `project`
- `global`
  - preferenze e regole non legate a un singolo progetto
  - recuperata per ultima

La ricerca puo comporre gli scope in modo esplicito con:

- `include_project`
- `include_workspace`
- `include_global`

Il comportamento consigliato e:

- `project` sempre attivo
- `workspace` attivo quando vuoi convenzioni comuni allo stesso workspace
- `global` attivo solo per memorie trasversali davvero riusabili

## Discovery e creazione progetti

I progetti non dovrebbero piu comparire "per caso" solo perche qualcuno ha scritto una memoria.

Flusso previsto:

1. `memory.list_projects` per discovery
2. `memory.get_project_info` per metadata e verifica esistenza
3. `memory.create_project` per creare esplicitamente un nuovo progetto

In multi-project mode i write project-scoped sono pensati per lavorare solo su progetti registrati.

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
python scripts/legacy/migrate_v1_to_v2.py --source-dir ./memories --workspace default --project default
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

Il runtime supportato usa SQLite come unico backend di persistenza.

## Esempi rapidi MCP

Creazione progetto:

```json
{
  "project_id": "crm-api",
  "display_name": "CRM API",
  "agent_id": "codex"
}
```

Write project-scoped:

```json
{
  "content": "La convenzione del progetto usa migration incrementali.",
  "agent_id": "codex",
  "scope": {
    "workspace_id": "default",
    "project_id": "crm-api",
    "scope_level": "project"
  }
}
```

Search con composizione scope:

```json
{
  "query": "migration incrementali",
  "agent_id": "codex",
  "scope": {
    "workspace_id": "default",
    "project_id": "crm-api"
  },
  "include_project": true,
  "include_workspace": true,
  "include_global": false
}
```
