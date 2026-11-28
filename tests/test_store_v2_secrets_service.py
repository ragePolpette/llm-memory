"""Secret detection, service validation and runtime wiring of the v2 store.

The fake secrets below are assembled at run time (J joins the parts) so that secret
scanners such as gitleaks do not flag the test source itself.
"""

from __future__ import annotations

import pytest

from src.store_v2.retrieval import HybridRetriever
from src.store_v2.runtime import RuntimeSettings, build_runtime
from src.store_v2.secrets import scan_fields, scan_text
from src.store_v2.service import Caller, MemoryServiceV2, ServiceError
from src.store_v2.store import MemoryStore


def J(*parts: str) -> str:
    return "".join(parts)


SECRETS = [
    (J("export OPENAI_API_KEY=sk-", "proj-abcdefghijklmnopqrstuvwxyz", "0123456789ABCD"), "openai_key"),
    ("ghp_" + "a1B2c3D4e5" * 4, "github_token"),
    (J("AKIA", "IOSFODNN7EXAMPLE"), "aws_access_key"),
    (J("postgres://app:", "S3cretPw", "@db.local:5432/x"), "url_credentials"),
    (J("curl -H 'Authorization: Bea", "rer abcDEF1234567890", "abcdef1234567890'"), "bearer_token"),
    (J("pass", "word = 'hX9#kq2L", "mP0z'"), "credential_assignment"),
    (J("-----BEGIN OPENSSH PRI", "VATE KEY-----"), "private_key"),
    (J("LLM_MEMORY_API_TOKEN", " ", "=", "4f9a8c", "2e7d1b6a3f9e0c").replace(" ", ""), "credential_assignment"),
    (J("fw_", "3ZmJ1bWJhc2VzZWNy", "ZXRrZXk1234"), "fireworks_key"),
    (J("sk-", "ant-api03-abcdefghij", "klmnopqrstuvwxyz"), "anthropic_key"),
    (J("eyJhbGciOiJIUzI1NiJ9", ".eyJzdWIiOiIxMjM0NTY3ODkwIn0", ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"), "jwt"),
    (J("xo", "xb-1234567890-abcdefghij"), "slack_token"),
]

CLEAN = [
    "La password policy richiede 12 caratteri.",
    "token: vedi il vault",
    "LLM_MEMORY_API_TOKEN=${LLM_MEMORY_API_TOKEN}",
    "Imposta MaximumIterationsPerRequest a 50.",
    "commit d00a49c3f1e2 merged",
    "api_key: <your-key>",
    "postgres://user:${PW}@db/x",
    "Il token di sessione scade dopo 30 minuti.",
    "Usa Authorization: Bearer <token> per /admin/audit",
    "password: changeme",
    "secret_key = os.environ['SECRET_KEY']",
    "sk-learn è la libreria",
    "",
]


@pytest.mark.parametrize("text,kind", SECRETS)
def test_secrets_detected(text, kind):
    assert kind in scan_text(text)


@pytest.mark.parametrize("text", CLEAN)
def test_ordinary_text_passes(text):
    assert scan_text(text) == []


def test_scan_fields_names_the_field_and_walks_structures():
    findings = scan_fields([
        ("content", "nulla"),
        ("metadata", {"api_key": J("Zk29dm", "Q81xLq"), "nested": {"how": J("AKIA", "IOSFODNN7EXAMPLE")}}),
        ("tags", ["ok", "ghp_" + "a1B2c3D4e5" * 4]),
        ("none", None),
    ])
    assert {(f.field, f.kind) for f in findings} == {
        ("metadata.api_key", "credential_assignment"),
        ("metadata.nested.how", "aws_access_key"),
        ("tags[1]", "github_token"),
    }


@pytest.fixture
def service(tmp_path):
    store = MemoryStore(tmp_path / "m.db", use_sqlite_vec=False)
    store.ensure_project("prj")
    store.ensure_project("__global__")
    yield MemoryServiceV2(store, HybridRetriever(store), episode_ttl_days=1)
    store.close()


CALLER = Caller(actor="t")


@pytest.mark.parametrize("call,code", [
    (lambda s: s.note(CALLER, project_id="", content="x"), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="Bad Id", content="x"), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x", event_type="Not Valid"), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content=5), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x" * 8001), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x", tags="a"), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x", tags=["has space"]), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x", tags=[f"t{i}" for i in range(21)]), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x", metadata=[1]), "invalid_argument"),
    (lambda s: s.save(CALLER, project_id="prj", type="fact", statement="x", impact="huge"), "invalid_argument"),
    (lambda s: s.save(CALLER, project_id="prj", type="fact", statement="x", confidence=2), "invalid_argument"),
    (lambda s: s.save(CALLER, project_id="prj", type="fact", statement="x", evidence=[{"kind": "web"}]),
     "invalid_argument"),
    (lambda s: s.save(CALLER, project_id="prj", type="fact", statement="x",
                      evidence=[{"kind": "episode", "ref": "missing"}]), "not_found"),
    (lambda s: s.save(CALLER, project_id="prj", type="fact", statement="x", supersedes="missing"), "not_found"),
    (lambda s: s.save(Caller(actor="d", source="distilled"), project_id="prj", type="fact", statement="x"),
     "invalid_argument"),
    (lambda s: s.supersede(CALLER, memory_id="missing", reason="r"), "not_found"),
    (lambda s: s.recall(CALLER, project_id="prj", query="x", kinds=["file"]), "invalid_argument"),
    (lambda s: s.recall(CALLER, project_id="prj", query="x", budget_chars=10), "invalid_argument"),
    (lambda s: s.projects(CALLER, action="delete"), "invalid_argument"),
    (lambda s: s.list(CALLER, project_id="prj", kind="file"), "invalid_argument"),
    (lambda s: s.list(CALLER, project_id="prj", limit=0), "invalid_argument"),
    (lambda s: s.list(CALLER, project_id="prj", since="ieri"), "invalid_argument"),
    (lambda s: s.list(CALLER, project_id="prj", type="fact"), "invalid_argument"),
    (lambda s: s.list(CALLER, project_id="prj", kind="memory", type="opinion"), "invalid_argument"),
    (lambda s: s.note(CALLER, project_id="prj", content="x", ttl_days=True), "invalid_argument"),
    (lambda s: s.projects(CALLER, action="create"), "invalid_argument"),
    (lambda s: s.projects(CALLER, action="create", project_id="ok-id",
                          description=J("tok", "en=Qm9vZ2xl", "MTIzNDU2")), "secret_rejected"),
])
def test_service_errors(service, call, code):
    with pytest.raises(ServiceError) as info:
        call(service)
    assert info.value.code == code and info.value.message


def test_supersede_with_unknown_replacement(service):
    memory = service.save(CALLER, project_id="prj", type="fact", statement="Uno.")
    with pytest.raises(ServiceError) as info:
        service.supersede(CALLER, memory_id=memory["id"], reason="r", replacement_id="missing")
    assert info.value.code == "not_found"


def test_save_in_another_projects_memory_is_refused(service):
    service.store.ensure_project("other")
    foreign = service.save(CALLER, project_id="other", type="fact", statement="Altro.")
    with pytest.raises(ServiceError) as info:
        service.save(CALLER, project_id="prj", type="fact", statement="Nuovo.", supersedes=foreign["id"])
    assert info.value.code == "invalid_argument"


def test_expiry_keeps_cited_episodes(service):
    cited = service.note(CALLER, project_id="prj", content="Citato.")
    service.note(CALLER, project_id="prj", content="Dimenticato.")
    service.save(CALLER, project_id="prj", type="fact", statement="Fatto.",
                 evidence=[{"kind": "episode", "ref": cited["id"]}])
    service.store._conn.execute("UPDATE episodes SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert service.expire_episodes() == 1
    assert service.get(CALLER, id=cited["id"])["item"]["content"] == "Citato."


def test_runtime_settings_from_env(tmp_path):
    settings = RuntimeSettings.from_env({"MEMORY_STORE_PATH": str(tmp_path / "s.db"), "MEMORY_EMBEDDING": "off",
                                         "MEMORY_SQLITE_VEC": "off", "MEMORY_EPISODE_TTL_DAYS": "7"})
    assert settings.embedding is None and settings.use_sqlite_vec is False and settings.episode_ttl_days == 7
    assert settings.diagnostics()["embedding"] is None
    on = RuntimeSettings.from_env({"MEMORY_STORE_PATH": "rel/s.db"})
    assert on.store_path.is_absolute() and on.embedding is not None
    assert on.diagnostics()["embedding"]["model"] == "fireworks/qwen3-embedding-8b"


async def test_runtime_builds_and_runs_background(tmp_path):
    runtime = build_runtime(RuntimeSettings(store_path=tmp_path / "s.db", v1_path=tmp_path / "none.db",
                                            embedding=None, use_sqlite_vec=False))
    try:
        assert runtime.store.get_project("__global__") and runtime.indexer is None
        runtime.start_background()
        await runtime.stop_background()
    finally:
        runtime.close()


def test_list_since_filters(service):
    service.note(CALLER, project_id="prj", content="Vecchio.")
    service.save(CALLER, project_id="prj", type="fact", statement="Fatto.")
    assert service.list(CALLER, project_id="prj", since="2999-01-01")["items"] == []
    assert service.list(CALLER, project_id="prj", kind="memory", since="2999-01-01T00:00:00Z")["items"] == []
    assert service.list(CALLER, project_id="prj", kind="memory", since="2000-01-01")["count"] == 1
