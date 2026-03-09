"""Persistenza Markdown append-only per memorie."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import frontmatter
from filelock import FileLock

from ..config import MemoryScope
from ..models import Memory

logger = logging.getLogger(__name__)


class MarkdownStore:
    """Store per memorie in formato Markdown con YAML frontmatter."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Crea le directory base per gli scope."""
        for scope in [MemoryScope.GLOBAL, MemoryScope.SHARED]:
            (self.base_path / scope.value).mkdir(parents=True, exist_ok=True)
        (self.base_path / "agents").mkdir(parents=True, exist_ok=True)
    
    def _get_memory_path(self, memory: Memory) -> Path:
        """Calcola il path del file per una memoria."""
        timestamp = memory.created_at.strftime("%Y-%m-%dT%H-%M-%S")
        short_id = str(memory.id)[:8]
        filename = f"{timestamp}_{short_id}.md"
        
        if memory.scope == MemoryScope.PRIVATE:
            # private/<agent_id>/<year>/<month>/
            year = memory.created_at.strftime("%Y")
            month = memory.created_at.strftime("%m")
            dir_path = self.base_path / "agents" / memory.agent_id / "private" / year / month
        else:
            # shared/ o global/ con <year>/<month>/
            year = memory.created_at.strftime("%Y")
            month = memory.created_at.strftime("%m")
            dir_path = self.base_path / memory.scope.value / year / month
        
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / filename
    
    async def write(self, memory: Memory) -> Path:
        """
        Scrive una memoria su file Markdown.
        
        Il file è append-only: una volta scritto non viene mai modificato.
        Usa file locking per evitare race condition.
        """
        path = self._get_memory_path(memory)
        lock_path = path.with_suffix(".lock")
        
        # Prepara il contenuto con frontmatter YAML
        post = frontmatter.Post(
            content=memory.content,
            id=str(memory.id),
            created_at=memory.created_at.isoformat(),
            agent_id=memory.agent_id,
            session_id=memory.session_id,
            scope=memory.scope.value if isinstance(memory.scope, MemoryScope) else memory.scope,
            context=memory.context,
            tags=memory.tags,
            content_hash=memory.content_hash,
            metadata=memory.metadata,
        )
        
        # File locking per scrittura sicura
        lock = FileLock(str(lock_path), timeout=10)
        try:
            with lock:
                path.write_text(frontmatter.dumps(post), encoding="utf-8")
                logger.info(f"Written memory {memory.id} to {path}")
        finally:
            # Cleanup lock file
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except OSError:
                    pass
        
        return path
    
    async def read(self, memory_id: str) -> Optional[Memory]:
        """
        Legge una memoria da file dato il suo ID.
        
        Cerca ricorsivamente nei file .md per trovare l'ID.
        """
        # Cerca in tutte le directory
        for md_file in self.base_path.rglob("*.md"):
            try:
                post = frontmatter.load(md_file)
                if post.get("id") == memory_id:
                    return self._post_to_memory(post)
            except Exception as e:
                logger.warning(f"Error reading {md_file}: {e}")
                continue
        
        return None
    
    async def read_by_path(self, path: Path) -> Optional[Memory]:
        """Legge una memoria da un path specifico."""
        try:
            post = frontmatter.load(path)
            return self._post_to_memory(post)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None
    
    async def list_memories(
        self,
        scope: Optional[MemoryScope] = None,
        agent_id: Optional[str] = None,
        limit: int = 50
    ) -> list[Path]:
        """
        Lista i file di memoria per scope/agente.
        
        Ritorna i path ordinati per data decrescente.
        """
        paths: list[Path] = []
        
        if scope == MemoryScope.PRIVATE and agent_id:
            search_dir = self.base_path / "agents" / agent_id / "private"
        elif scope:
            search_dir = self.base_path / scope.value
        else:
            search_dir = self.base_path
        
        if search_dir.exists():
            paths = list(search_dir.rglob("*.md"))
        
        # Ordina per nome file (che contiene timestamp) decrescente
        paths.sort(key=lambda p: p.name, reverse=True)
        
        return paths[:limit]
    
    def _post_to_memory(self, post: frontmatter.Post) -> Memory:
        """Converte un Post frontmatter in Memory."""
        scope_value = post.get("scope", "shared")
        if isinstance(scope_value, str):
            scope = MemoryScope(scope_value)
        else:
            scope = scope_value
            
        return Memory(
            id=post.get("id"),
            content=post.content,
            context=post.get("context", ""),
            agent_id=post.get("agent_id", "unknown"),
            session_id=post.get("session_id"),
            scope=scope,
            tags=post.get("tags", []),
            metadata=post.get("metadata", {}),
            created_at=datetime.fromisoformat(post.get("created_at")),
            content_hash=post.get("content_hash", ""),
        )
    
    async def find_by_hash(self, content_hash: str) -> Optional[str]:
        """
        Cerca una memoria esistente con lo stesso content hash.
        
        Usato per deduplicazione.
        """
        for md_file in self.base_path.rglob("*.md"):
            try:
                post = frontmatter.load(md_file)
                if post.get("content_hash") == content_hash:
                    return post.get("id")
            except Exception:
                continue
        return None
