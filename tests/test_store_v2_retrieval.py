"""Embedding adapter, background indexer and hybrid recall of the v2 store."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.store_v2.embeddings import EmbeddingError, EmbeddingSettings, OpenAICompatibleEmbeddings
from src.store_v2.indexer import EmbeddingIndexer
from src.store_v2.retrieval import HybridRetriever
from src.store_v2.schema import GLOBAL_PROJECT
from src.store_v2.store import AuditRecord, Episode, Memory, MemoryStore

PROJECT = "prj-a"


# -- a fake OpenAI-compatible server ---------------------------------------------


class FakeServer:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.requests: list[dict] = []
        self.script: list[int] = []  # status codes to answer before succeeding
        self.wrong_dim = False
        self.reverse = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                if server.script:
                    code = server.script.pop(0)
                    self.send_response(code)
                    if code == 429:
                        self.send_header("Retry-After", "0")
                    self.end_headers()
                    self.wfile.write(b'{"error": "nope"}')
                    return
                dim = server.dim + (1 if server.wrong_dim else 0)
                data = [{"index": i, "embedding": [float(i + 1)] * dim} for i in range(len(body["input"]))]
                if server.reverse:
                    data.reverse()
                payload = json.dumps({"data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = FakeServer()
    yield srv
    srv.close()


def _client(server, **overrides):
    settings = EmbeddingSettings(base_url=server.url, model="test-model", dim=4, api_key_env="LLM_MEMORY_TEST_KEY",
                                 max_retries=2, **overrides)
    return OpenAICompatibleEmbeddings(settings, _sleep=lambda _s: None)


def test_adapter_sends_openai_compatible_request(server, monkeypatch):
    monkeypatch.setenv("LLM_MEMORY_TEST_KEY", "k-123")
    client = _client(server, batch_size=2)
    vectors = client.embed_documents(["uno", "due", "tre"])
    assert vectors == [[1.0] * 4, [2.0] * 4, [1.0] * 4]  # two batches: 2 + 1
    first = server.requests[0]
    assert first["path"] == "/v1/embeddings" and first["auth"] == "Bearer k-123"
    assert first["body"]["model"] == "test-model" and first["body"]["dimensions"] == 4
    assert first["body"]["input"] == ["uno", "due"]


def test_adapter_query_prefix_and_no_auth_without_key(server, monkeypatch):
    monkeypatch.delenv("LLM_MEMORY_TEST_KEY", raising=False)
    client = _client(server, query_prefix="Q: ", send_dimensions=False)
    client.embed_query("dove sta il db?")
    request = server.requests[0]
    assert request["auth"] is None
    assert request["body"]["input"] == ["Q: dove sta il db?"] and "dimensions" not in request["body"]


def test_adapter_orders_by_index(server):
    server.reverse = True
    assert _client(server).embed_documents(["a", "b"]) == [[1.0] * 4, [2.0] * 4]


def test_adapter_retries_rate_limits_and_server_errors(server):
    server.script = [429, 503]
    assert _client(server).embed_query("x") == [1.0] * 4
    assert len(server.requests) == 3


def test_adapter_gives_up_and_does_not_retry_client_errors(server):
    server.script = [500, 500, 500]
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        _client(server).embed_query("x")
    server.requests.clear()
    server.script = [400]
    with pytest.raises(EmbeddingError, match="HTTP 400"):
        _client(server).embed_query("x")
    assert len(server.requests) == 1


def test_adapter_rejects_wrong_dimension(server):
    server.wrong_dim = True
    with pytest.raises(EmbeddingError, match="dimension"):
        _client(server).embed_query("x")


def test_adapter_unreachable_provider():
    settings = EmbeddingSettings(base_url="http://127.0.0.1:9/v1", dim=4, max_retries=1, timeout_s=1)
    with pytest.raises(EmbeddingError):
        OpenAICompatibleEmbeddings(settings, _sleep=lambda _s: None).embed_query("x")


def test_settings_from_env():
    settings = EmbeddingSettings.from_env({
        "MEMORY_EMBEDDING_BASE_URL": "http://localhost:11434/v1/",
        "MEMORY_EMBEDDING_MODEL": "nomic-embed-text",
        "MEMORY_EMBEDDING_DIM": "768",
        "MEMORY_EMBEDDING_API_KEY_ENV": "",
        "MEMORY_EMBEDDING_SEND_DIMENSIONS": "false",
    })
    assert settings.base_url == "http://localhost:11434/v1" and settings.provider == "localhost:11434"
    assert settings.dim == 768 and settings.send_dimensions is False
    defaults = EmbeddingSettings.from_env({})
    assert defaults.model == "fireworks/qwen3-embedding-8b" and defaults.dim == 1024
    assert defaults.provider == "api.fireworks.ai"


# -- a deterministic "semantic" client: words map to concepts ---------------------

CONCEPTS = {
    "database": 0, "postgresql": 0, "db": 0, "dati": 0, "archivio": 0,
    "notifiche": 1, "ntfy": 1, "avvisi": 1, "alert": 1, "urgenti": 1,
    "backup": 2, "salvataggio": 2, "copia": 2,
    "timeout": 3, "lento": 3, "scade": 3,
}


class ConceptClient:
    provider, model, dim = "fake", "concepts", 4

    def __init__(self):
        self.fail = False
        self.calls = 0

    def version_config(self):
        return {}

    def _embed(self, text):
        vector = [0.0] * self.dim
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            if word in CONCEPTS:
                vector[CONCEPTS[word]] += 1.0
        vector[-1] += 0.01  # never a zero vector
        return vector

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("provider down")
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        if self.fail:
            raise EmbeddingError("provider down")
        return self._embed(text)


def _audit():
    return AuditRecord(action="write_attempt", actor="test")


@pytest.fixture(params=[False, True], ids=["scan", "vec"])
def store(tmp_path, request):
    if request.param:
        pytest.importorskip("sqlite_vec")
    s = MemoryStore(tmp_path / "m.db", use_sqlite_vec=request.param)
    s.ensure_project(PROJECT)
    s.ensure_project("prj-b")
    s.ensure_project(GLOBAL_PROJECT)
    yield s
    s.close()


def _memory(store, statement, project=PROJECT, type="fact", **kw):
    memory_id, _ = store.add_memory(Memory(project_id=project, type=type, statement=statement, actor="t", **kw),
                                    audit=_audit())
    return memory_id


def test_indexer_embeds_queue_and_survives_provider_failure(store):
    ids = [_memory(store, f"Il sistema usa PostgreSQL numero {i}.") for i in range(5)]
    client = ConceptClient()
    client.fail = True
    indexer = EmbeddingIndexer(store, client, batch_size=2)
    outcome = indexer.run_once()
    assert outcome.embedded == 0 and outcome.remaining == 5 and "provider down" in outcome.error

    client.fail = False
    outcome = indexer.run_once(max_items=3)
    assert outcome.embedded == 3 and outcome.remaining == 2
    outcome = indexer.run_once()
    assert outcome.embedded == 2 and outcome.remaining == 0 and outcome.error is None
    assert store.active_embedding_version()["version_id"] == indexer.version_id
    assert len(ids) == 5


def test_indexer_requeues_everything_on_provider_change(store):
    _memory(store, "Il sistema usa PostgreSQL.")
    EmbeddingIndexer(store, ConceptClient()).run_once()

    class Other(ConceptClient):
        model = "concepts-v2"

    other = EmbeddingIndexer(store, Other())
    assert other.run_once().embedded == 1


def test_indexer_never_sends_encrypted_v1_memories(store):
    _memory(store, "gAAAAB-ciphertext", tags=["v1-encrypted"])
    _memory(store, "Il sistema usa PostgreSQL.")
    client = ConceptClient()
    assert EmbeddingIndexer(store, client).run_once().embedded == 1


@pytest.fixture
def corpus(store):
    client = ConceptClient()
    ids = {
        "db": _memory(store, "Il sistema usa PostgreSQL 16 per la fatturazione."),
        "ticket": _memory(store, "Il bug TS-4471 nasce dal pool di connessioni.", type="lesson", why="pool troppo piccolo"),
        "ntfy": _memory(store, "Le notifiche urgenti passano da ntfy.", type="decision", why="self-hosted"),
        "old": _memory(store, "Gli avvisi urgenti passavano da email."),
        "other": _memory(store, "Il progetto B usa PostgreSQL 15.", project="prj-b"),
        "global": _memory(store, "Le dipendenze devono avere almeno 45 giorni.", project=GLOBAL_PROJECT),
    }
    store.supersede(ids["old"], audit=AuditRecord(action="invalidate", actor="t"), replacement_id=ids["ntfy"])
    episode_id = store.add_episode(Episode(project_id=PROJECT, event_type="observation", agent_id="t",
                                           content="Il job notturno scade per timeout."), audit=_audit())
    ids["episode"] = episode_id
    EmbeddingIndexer(store, client).run_once()
    return client, ids


def test_identifier_found_by_text(store, corpus):
    client, ids = corpus
    result = HybridRetriever(store, client).recall("TS-4471", [PROJECT])
    assert result.mode == "hybrid" and result.items[0].id == ids["ticket"]
    assert "text" in result.items[0].matched_by


def test_paraphrase_found_by_vector_only(store, corpus):
    client, ids = corpus
    query = "dove finiscono gli alert?"  # no word in common with the stored memory
    assert all(hit.id != ids["ntfy"] for hit in store.search_text(query, [PROJECT]))
    result = HybridRetriever(store, client).recall(query, [PROJECT])
    assert result.items[0].id == ids["ntfy"] and result.items[0].matched_by == ["vector"]
    assert "Why: self-hosted" in result.items[0].text


def test_agreement_of_both_rankings_wins(store, corpus):
    client, ids = corpus
    result = HybridRetriever(store, client).recall("quale database usa la fatturazione PostgreSQL", [PROJECT])
    assert result.items[0].id == ids["db"] and result.items[0].matched_by == ["text", "vector"]


def test_scoping_closed_and_global(store, corpus):
    client, ids = corpus
    retriever = HybridRetriever(store, client)
    found = {item.id for item in retriever.recall("PostgreSQL avvisi urgenti dipendenze", [PROJECT], limit=20).items}
    assert ids["other"] not in found and ids["old"] not in found and ids["global"] in found
    history = {item.id for item in retriever.recall("avvisi urgenti", [PROJECT], include_closed=True, limit=20).items}
    assert ids["old"] in history
    without_global = retriever.recall("dipendenze 45 giorni", [PROJECT], include_global=False, limit=20)
    assert ids["global"] not in {item.id for item in without_global.items}


def test_episodes_and_kinds(store, corpus):
    client, ids = corpus
    retriever = HybridRetriever(store, client)
    assert retriever.recall("il job è lento", [PROJECT]).items[0].id == ids["episode"]
    only_memories = retriever.recall("il job è lento", [PROJECT], kinds=("memory",))
    assert ids["episode"] not in {item.id for item in only_memories.items}


def test_budget_and_limit(store, corpus):
    client, ids = corpus
    retriever = HybridRetriever(store, client)
    tight = retriever.recall("PostgreSQL notifiche dipendenze timeout", [PROJECT], budget_chars=60)
    assert len(tight.items) == 1 and tight.truncated and tight.used_chars == len(tight.items[0].text)
    limited = retriever.recall("PostgreSQL notifiche dipendenze timeout", [PROJECT], limit=2)
    assert len(limited.items) == 2 and limited.truncated


def test_degrades_to_text_when_provider_fails_or_is_missing(store, corpus):
    client, ids = corpus
    client.fail = True
    result = HybridRetriever(store, client).recall("TS-4471", [PROJECT])
    assert result.mode == "text" and "provider down" in result.degraded
    assert result.items[0].id == ids["ticket"]
    plain = HybridRetriever(store).recall("TS-4471", [PROJECT])
    assert plain.mode == "text" and plain.degraded and plain.items[0].id == ids["ticket"]


def test_unembedded_items_still_found_by_text(store, corpus):
    client, ids = corpus
    fresh = _memory(store, "Il ticket TS-9001 riguarda il salvataggio notturno.")
    result = HybridRetriever(store, client).recall("TS-9001", [PROJECT])
    assert result.items[0].id == fresh and result.items[0].matched_by == ["text"]
