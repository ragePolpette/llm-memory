"""Gestione scopes e permessi per multi-agent."""

from __future__ import annotations

import logging

from ..config import MemoryScope
from ..models import Memory

logger = logging.getLogger(__name__)


class ScopeManager:
    """
    Gestisce i permessi di accesso alle memorie basati su scope e agent_id.
    
    Regole:
    - PRIVATE: solo l'agente proprietario può leggere/scrivere
    - SHARED: tutti gli agenti possono leggere/scrivere
    - GLOBAL: tutti possono leggere, nessuno può scrivere (read-only)
    """
    
    def can_read(self, agent_id: str, memory: Memory) -> bool:
        """
        Verifica se un agente può leggere una memoria.
        
        Args:
            agent_id: ID dell'agente richiedente
            memory: Memoria da verificare
            
        Returns:
            True se l'agente può leggere la memoria
        """
        if memory.scope == MemoryScope.PRIVATE:
            return memory.agent_id == agent_id
        
        # SHARED e GLOBAL sono leggibili da tutti
        return True
    
    def can_write(self, agent_id: str, scope: MemoryScope) -> bool:
        """
        Verifica se un agente può scrivere in uno scope.
        
        Args:
            agent_id: ID dell'agente richiedente
            scope: Scope target
            
        Returns:
            True se l'agente può scrivere nello scope
        """
        # GLOBAL è read-only
        if scope == MemoryScope.GLOBAL:
            return False
        
        # PRIVATE e SHARED sono scrivibili
        return True
    
    def filter_by_scope(
        self,
        agent_id: str,
        memories: list[Memory],
        requested_scope: str = "all"
    ) -> list[Memory]:
        """
        Filtra memorie per scope accessibile dall'agente.
        
        Args:
            agent_id: ID dell'agente richiedente
            memories: Lista di memorie da filtrare
            requested_scope: Scope richiesto ("all", "private", "shared", "global")
            
        Returns:
            Lista filtrata di memorie accessibili
        """
        filtered = []
        
        for memory in memories:
            # Verifica permesso lettura
            if not self.can_read(agent_id, memory):
                continue
            
            # Filtra per scope richiesto
            if requested_scope == "all":
                filtered.append(memory)
            elif requested_scope == memory.scope.value:
                filtered.append(memory)
        
        return filtered
