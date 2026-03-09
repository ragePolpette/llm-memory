# Guida Docker per LLM Memory

## Prerequisiti

- Docker Desktop installato
- Docker Compose (incluso in Docker Desktop)

## Build dell'Immagine

### Opzione 1: Build Semplice

```bash
cd <project-root>\llm-memory
docker build -t llm-memory:latest .
```

### Opzione 2: Build con Pre-download Modello

Per includere il modello embedding nell'immagine (build più lenta ma avvio più veloce):

1. Decommentare la riga nel `Dockerfile`:
   ```dockerfile
   RUN python scripts/download_model.py
   ```

2. Build:
   ```bash
   docker build -t llm-memory:latest .
   ```

## Avvio con Docker Compose (Consigliato)

### Primo Avvio

```bash
# Build e avvio
docker-compose up -d

# Visualizza log
docker-compose logs -f llm-memory

# Verifica stato
docker-compose ps
```

### Comandi Utili

```bash
# Ferma il container
docker-compose down

# Riavvia
docker-compose restart

# Ferma e rimuovi volumi (ATTENZIONE: cancella i dati!)
docker-compose down -v

# Rebuild dopo modifiche al codice
docker-compose up -d --build

# Accedi al container
docker-compose exec llm-memory bash
```

## Avvio con Docker Run (Alternativa)

```bash
# Avvio base
docker run -d \
  --name llm-memory \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/logs:/data/logs" \
  llm-memory:latest

# Con variabili di ambiente custom
docker run -d \
  --name llm-memory \
  -e INDEXING_MODE=async \
  -e EMBEDDING_MODEL=intfloat/multilingual-e5-small \
  -v "$(pwd)/data:/data" \
  llm-memory:latest

# Interattivo (per debug)
docker run -it --rm \
  -v "$(pwd)/data:/data" \
  llm-memory:latest
```

## Connessione al Server MCP

### Da Host (Windows)

Il server MCP gira su stdio all'interno del container. Per connettersi:

```bash
# Attach al container
docker attach llm-memory-server

# Oppure exec
docker exec -it llm-memory-server python -m src.mcp_server.server
```

### Futura Configurazione HTTP (Opzionale)

Se in futuro vuoi esporre MCP via HTTP invece di stdio:

1. Modifica `docker-compose.yml`:
   ```yaml
   ports:
     - "8080:8080"
   ```

2. Modifica il server per ascoltare su HTTP invece di stdio

## Persistenza Dati

I dati vengono salvati in:

```
./data/
├── memories/       # File Markdown
├── lancedb/        # Database vettoriale
└── logs/           # Log del server
```

**IMPORTANTE**: Non cancellare la directory `data/` se vuoi mantenere le memorie!

## Configurazione Avanzata

### Variabili di Ambiente

Modifica `docker-compose.yml`:

```yaml
environment:
  - MEMORY_STORAGE_DIR=/data/memories
  - LANCEDB_DIR=/data/lancedb
  - EMBEDDING_MODEL=intfloat/multilingual-e5-small
  - EMBEDDING_DIM=384
  - INDEXING_MODE=sync  # o async, hybrid
  - HYBRID_THRESHOLD_BYTES=1024
```

### Limiti Risorse

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## Troubleshooting

### Container non si avvia

```bash
# Visualizza log
docker-compose logs llm-memory

# Verifica build
docker-compose build --no-cache
```

### Modello embedding non trovato

Il modello si scarica automaticamente al primo avvio. Attendi qualche minuto.

Per pre-scaricare:
```bash
docker-compose exec llm-memory python scripts/download_model.py
```

### Permessi su volumi (Linux/Mac)

```bash
# Crea directory con permessi corretti
mkdir -p data logs
chmod -R 777 data logs
```

### Reset completo

```bash
# Ferma e rimuovi tutto
docker-compose down -v
rm -rf data/ logs/

# Rebuild da zero
docker-compose up -d --build
```

## Backup e Restore

### Backup

```bash
# Backup directory dati
tar -czf llm-memory-backup-$(date +%Y%m%d).tar.gz data/

# Oppure copia
cp -r data/ data-backup/
```

### Restore

```bash
# Estrai backup
tar -xzf llm-memory-backup-20260205.tar.gz

# Riavvia container
docker-compose restart
```

## Aggiornamento

```bash
# Pull nuovo codice
git pull  # se usi git

# Rebuild
docker-compose up -d --build

# Verifica
docker-compose logs -f
```

## Uso in Produzione (Futuro)

### Con Docker Swarm

```bash
docker stack deploy -c docker-compose.yml llm-memory
```

### Con Kubernetes

Converti docker-compose in manifests:
```bash
kompose convert -f docker-compose.yml
kubectl apply -f .
```

## Monitoring

### Log in tempo reale

```bash
docker-compose logs -f llm-memory
```

### Statistiche risorse

```bash
docker stats llm-memory-server
```

### Inspect container

```bash
docker inspect llm-memory-server
```

## Note Importanti

1. **Stdio vs HTTP**: Attualmente il server usa stdio. Per uso remoto, considera di implementare un wrapper HTTP.

2. **Modello embedding**: ~120MB, si scarica al primo avvio. Considera di includerlo nell'immagine per deployment più veloci.

3. **Volumi**: Usa sempre volumi per `/data` altrimenti perdi le memorie quando ricrei il container.

4. **Sicurezza**: In produzione, non esporre il container direttamente. Usa un reverse proxy (nginx, traefik).

## Esempio Completo

```bash
# 1. Build
cd <project-root>\llm-memory
docker-compose build

# 2. Avvio
docker-compose up -d

# 3. Verifica
docker-compose ps
docker-compose logs -f

# 4. Test (da altro terminale)
docker exec -it llm-memory-server python -c "from src.config import get_config; print(get_config())"

# 5. Stop
docker-compose down
```

Il container è pronto per uso locale e futuro deployment!
