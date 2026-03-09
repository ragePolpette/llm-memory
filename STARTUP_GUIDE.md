# Script di Avvio LLM Memory Server

Sono stati creati 3 script batch per gestire il server MCP:

## 1. `start_server.bat` - Avvio Interattivo

Avvia il server con output visibile nella console.

**Uso:**
```bash
cd <project-root>\llm-memory
start_server.bat
```

**Caratteristiche:**
- Output in tempo reale
- Log salvato in `logs/mcp_server_YYYYMMDD_HHMMSS.log`
- Ideale per debugging e monitoraggio

## 2. `start_server_silent.bat` - Avvio Minimizzato (Autostart)

Avvia il server in finestra minimizzata.

**Uso:**
```bash
start_server_silent.bat
```

**Caratteristiche:**
- Finestra minimizzata (visibile nella barra applicazioni)
- Log salvato automaticamente
- **Ideale per autostart di Windows**
- Puoi ripristinare la finestra per vedere l'output

### Come Aggiungere all'Autostart

#### Metodo 1: Cartella Autostart (Consigliato)

1. Premi `Win + R`
2. Digita: `shell:startup`
3. Crea un collegamento a `start_server_silent.bat` in quella cartella

#### Metodo 2: Task Scheduler

1. Apri "Utilità di pianificazione" (Task Scheduler)
2. Crea attività di base
3. Nome: "LLM Memory Server"
4. Trigger: "All'avvio del computer"
5. Azione: Avvia programma
6. Programma: `<project-root>\llm-memory\start_server_silent.bat`
7. Opzioni avanzate:
   - ✅ Esegui con privilegi più elevati (se necessario)
   - ✅ Esegui indipendentemente dall'accesso dell'utente

## 3. `stop_server.bat` - Ferma Server

Ferma tutti i processi del server MCP in esecuzione.

**Uso:**
```bash
stop_server.bat
```

**Caratteristiche:**
- Trova e termina automaticamente i processi del server
- Chiude la finestra minimizzata se presente

## Log Files

I log vengono salvati in:
```
logs/
├── mcp_server_20260205_165357.log
├── mcp_server_20260205_170123.log
└── last_start.txt
```

## Verifica Server in Esecuzione

```powershell
# Verifica processi Python
tasklist | findstr python

# Controlla ultimo avvio
type logs\last_start.txt

# Visualizza log più recente
type logs\mcp_server_*.log | more
```

## Troubleshooting

### Server non si avvia

1. Verifica Python nel PATH:
   ```bash
   python --version
   ```

2. Controlla i log in `logs/`

3. Prova avvio interattivo per vedere errori:
   ```bash
   start_server.bat
   ```

### Server già in esecuzione

```bash
stop_server.bat
start_server_silent.bat
```

## Esempio Autostart Completo

1. **Crea collegamento:**
   - Tasto destro su `start_server_silent.bat`
   - "Crea collegamento"

2. **Sposta in autostart:**
   - `Win + R` → `shell:startup`
   - Sposta il collegamento

3. **Riavvia Windows**

4. **Verifica:**
   ```bash
   tasklist | findstr pythonw
   type logs\last_start.txt
   ```

Il server si avvierà automaticamente ad ogni login!
