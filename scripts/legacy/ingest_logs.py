#!/usr/bin/env python3
"""
Script di ingest per Daily Log -> llm-memory.

Estrae fatti atomici dai file markdown del Decision Log e li ingesta
nel sistema llm-memory per ricerca semantica.

Uso:
    python scripts/ingest_logs.py --source "..\\Decision_Log" --dry-run
    python scripts/ingest_logs.py --source "..\\Decision_Log"
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Aggiungi src al path per import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MemoryScope, get_config
from src.embedding.embedding_service import get_embedding_provider
from src.models import Memory
from src.storage.markdown_store import MarkdownStore
from src.vectordb.lance_store import LanceVectorStore


# Pattern per estrarre sezioni dai daily log
DECISION_PATTERN = re.compile(
    r"###\s+(.+?)\n"  # Titolo della decisione
    r"(?:\*\*Context\*\*:\s*\n?(.*?))?"  # Context (opzionale)
    r"(?:\*\*Rationale\*\*:\s*\n?(.*?))?"  # Rationale (opzionale)
    r"(?:\*\*Trade-offs?\*\*:\s*\n?(.*?))?"  # Trade-offs (opzionale)
    r"(?:\*\*Impact\*\*:\s*\n?(.*?))?"  # Impact (opzionale)
    r"(?=###|\n---|\Z)",  # Fine sezione
    re.DOTALL | re.IGNORECASE
)

NOTES_PATTERN = re.compile(
    r"##\s+📝\s*Notes\s*(?:&|and)?\s*Synthesized Thoughts?\s*\n(.*?)(?=\n##|\Z)",
    re.DOTALL | re.IGNORECASE
)

# Tag comuni da estrarre automaticamente
TAG_KEYWORDS = {
    "SQL": ["SQL", "query", "INSERT", "UPDATE", "DELETE", "SELECT", "JOIN"],
    "BPOFH": ["BPOFH", "ambiente BPOFH", "studi", "IdAziendaCollegata"],
    "security": ["XSS", "security", "vulnerabilità", "injection"],
    "performance": ["performance", "ottimizzazione", "N+1", "cache"],
    "refactoring": ["refactoring", "pulizia", "rimozione", "cleanup"],
    "architecture": ["architettura", "pattern", "design", "modello"],
    "database": ["database", "tabella", "schema", "migration"],
    "api": ["API", "endpoint", "controller", "REST"],
}


def extract_tags(text: str) -> list[str]:
    """Estrae tag rilevanti dal testo."""
    tags = set()
    text_lower = text.lower()
    
    for tag, keywords in TAG_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                tags.add(tag)
                break
    
    return sorted(tags)


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Estrae la data dal nome file (es. 2026-02-06.md)."""
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if match:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            tzinfo=timezone.utc
        )
    return None


def extract_decisions(content: str, log_date: datetime) -> list[dict]:
    """Estrae le decisioni dal contenuto del log."""
    facts = []
    
    # Trova sezione Decision Log
    decision_section_match = re.search(
        r"##\s+🧠\s*Decision Log\s*\n(.*?)(?=\n##[^#]|\Z)",
        content,
        re.DOTALL
    )
    
    if not decision_section_match:
        return facts
    
    decision_section = decision_section_match.group(1)
    
    # Estrai ogni decisione (### heading)
    current_decision = None
    current_content = []
    
    lines = decision_section.split("\n")
    for line in lines:
        if line.startswith("### "):
            # Salva decisione precedente
            if current_decision and current_content:
                fact_content = "\n".join(current_content).strip()
                if fact_content:
                    facts.append({
                        "title": current_decision,
                        "content": fact_content,
                        "date": log_date,
                        "tags": extract_tags(fact_content),
                    })
            # Nuova decisione
            current_decision = line[4:].strip()
            current_content = []
        elif current_decision:
            current_content.append(line)
    
    # Ultima decisione
    if current_decision and current_content:
        fact_content = "\n".join(current_content).strip()
        if fact_content:
            facts.append({
                "title": current_decision,
                "content": fact_content,
                "date": log_date,
                "tags": extract_tags(fact_content),
            })
    
    return facts


def extract_notes(content: str, log_date: datetime) -> list[dict]:
    """Estrae le note sintetizzate dal log."""
    facts = []
    
    match = NOTES_PATTERN.search(content)
    if not match:
        return facts
    
    notes_content = match.group(1).strip()
    if not notes_content:
        return facts
    
    # Estrai bullet points individuali
    bullets = re.findall(r"\*\s+\*\*([^:]+)\*\*:\s*(.+?)(?=\n\*|\Z)", notes_content, re.DOTALL)
    
    for title, body in bullets:
        body = body.strip()
        if body:
            facts.append({
                "title": title.strip(),
                "content": f"**{title.strip()}**: {body}",
                "date": log_date,
                "tags": extract_tags(body),
            })
    
    return facts


def parse_log_file(file_path: Path) -> list[dict]:
    """Parse un singolo file di log e ritorna i fatti estratti."""
    content = file_path.read_text(encoding="utf-8")
    log_date = extract_date_from_filename(file_path.name) or datetime.now(timezone.utc)
    
    facts = []
    facts.extend(extract_decisions(content, log_date))
    facts.extend(extract_notes(content, log_date))
    
    return facts


async def ingest_facts(
    facts: list[dict],
    markdown_store: MarkdownStore,
    vector_store: LanceVectorStore,
    source_file: str,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[int, int]:
    """Ingesta i fatti nel sistema llm-memory."""
    ingested = 0
    skipped = 0
    
    for fact in facts:
        # Crea memory object
        memory = Memory(
            content=fact["content"],
            context=f"Decision Log: {fact['title']} ({fact['date'].strftime('%Y-%m-%d')})",
            agent_id="decision-ingest",
            scope=MemoryScope.SHARED,
            tags=fact["tags"] + ["decision-log", source_file],
            metadata={
                "source": "decision-log",
                "source_file": source_file,
                "decision_title": fact["title"],
                "log_date": fact["date"].isoformat(),
            },
            created_at=fact["date"],
        )
        
        if dry_run:
            print(f"  [DRY-RUN] Would ingest: {fact['title'][:50]}...")
            print(f"            Tags: {fact['tags']}")
            print(f"            Content: {fact['content'][:100]}...")
            print()
            ingested += 1
            continue
        
        # Check deduplicazione
        if not force:
            existing = await markdown_store.find_by_hash(memory.content_hash)
            if existing:
                print(f"  [SKIP] Already exists: {fact['title'][:50]}...")
                skipped += 1
                continue
        
        # Salva in markdown store
        await markdown_store.write(memory)
        
        # Indicizza in vector store
        await vector_store.index(memory)
        
        print(f"  [OK] Ingested: {fact['title'][:50]}...")
        ingested += 1
    
    return ingested, skipped


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest Daily Logs into llm-memory"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory sorgente con i file .md del Decision Log"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Ingesta un singolo file invece di tutta la directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra cosa verrebbe ingerito senza scrivere"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora deduplicazione (re-ingesta tutto)"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"ERROR: Directory non trovata: {source_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("LLM-Memory Decision Log Ingest")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print()
    
    # Inizializza componenti (solo se non dry-run o per test)
    if not args.dry_run:
        print("Initializing llm-memory components...")
        config = get_config()
        embedding_provider = get_embedding_provider(
            model_name=config.embedding_model,
            device=config.embedding_device
        )
        markdown_store = MarkdownStore(config.storage_dir)
        vector_store = LanceVectorStore(config.lancedb_dir, embedding_provider)
        print("Components initialized.\n")
    else:
        markdown_store = None
        vector_store = None
    
    # Trova file da processare
    if args.file:
        files = [source_dir / args.file]
    else:
        # Solo file con pattern data (YYYY-MM-DD.md)
        files = sorted([
            f for f in source_dir.glob("*.md")
            if re.match(r"\d{4}-\d{2}-\d{2}\.md", f.name)
        ])
    
    print(f"Found {len(files)} log file(s) to process.\n")
    
    total_ingested = 0
    total_skipped = 0
    
    for file_path in files:
        print(f"Processing: {file_path.name}")
        
        facts = parse_log_file(file_path)
        if not facts:
            print("  (No facts extracted)\n")
            continue
        
        print(f"  Extracted {len(facts)} fact(s)")
        
        ingested, skipped = await ingest_facts(
            facts,
            markdown_store,
            vector_store,
            file_path.name,
            dry_run=args.dry_run,
            force=args.force,
        )
        
        total_ingested += ingested
        total_skipped += skipped
        print()
    
    print("=" * 60)
    print(f"SUMMARY: {total_ingested} ingested, {total_skipped} skipped")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
