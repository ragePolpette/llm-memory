# Guida Rapida - LLM Memory

## 1. Installazione

```bash
cd <project-root>\llm-memory

# Installa dipendenze
pip install -e .

# Scarica il modello embedding in locale
python scripts/download_model.py
```

## 2. Configurazione

Il file `.env` è già configurato con i valori di default.

## 3. Avvio Server MCP

```bash
python -m src.mcp_server.server
```

Il server è ora in ascolto su stdio per connessioni MCP.

## 4. Test del Sistema

```bash
# Esegui tutti i test
pytest tests/ -v

# Test specifici
pytest tests/test_storage.py -v
pytest tests/test_integration.py -v
```

## 5. Uso da Agente AI

Gli agenti possono connettersi al server MCP e usare i seguenti tools:

### `memory_write`
```json
{
  "content": "Contenuto della memoria",
  "context": "contesto_semantico",
  "agent_id": "agent-alpha",
  "scope": "shared",
  "tags": ["tag1", "tag2"]
}
```

### `memory_search`
```json
{
  "query": "Cosa ricordi su Python?",
  "agent_id": "agent-alpha",
  "scope": "all",
  "limit": 10
}
```

### `memory_read`
```json
{
  "memory_id": "uuid-della-memoria",
  "agent_id": "agent-alpha"
}
```

### `memory_list`
```json
{
  "agent_id": "agent-alpha",
  "scope": "shared",
  "limit": 50
}
```

## 6. Struttura Dati

Le memorie vengono salvate in:
- **Filesystem**: `./memories/` (Markdown con YAML frontmatter)
- **Vector DB**: `./data/lancedb/` (indicizzazione semantica)

## 7. Scopes

- **private**: Solo l'agente proprietario può accedere
- **shared**: Tutti gli agenti possono leggere/scrivere
- **global**: Tutti possono leggere, nessuno può scrivere

## 8. Modalità Indicizzazione

- **sync** (default): Indicizzazione immediata (~50-200ms)
- **async**: Coda background (latenza 1-5 sec)
- **hybrid**: Sync per contenuti <1KB, async per grandi

Cambia in `.env`:
```
INDEXING_MODE=async
```
