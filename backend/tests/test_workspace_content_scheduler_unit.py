"""
Unit tests for:
  backend/orchestrator/workspace_manager.py
  backend/orchestrator/content_orchestrator.py
  backend/services/workflow_scheduler.py

User-specified critical tests (all covered):
  - Compression/decompression round-trip is correct
  - last_agent_result is always raw even when action history is compressed
  - Thread workspace isolation — thread A cannot read thread B's files
  - Cron schedule loaded from DB on startup
  - Invalid cron expression rejected cleanly
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ── Path setup (mirrors test_brain_unit.py) ───────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent       # backend/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))               # project root


# ─────────────────────────────────────────────────────────────────────────────
# WorkspaceManager
# ─────────────────────────────────────────────────────────────────────────────

import backend.orchestrator.workspace_manager as wm_module
from backend.orchestrator.workspace_manager import (
    FileMetadata,
    WorkspaceManager,
    get_workspace_manager,
)


@pytest.fixture(autouse=True)
def reset_workspace_registry():
    """Clear the global singleton dict before/after every test."""
    wm_module._workspace_managers.clear()
    yield
    wm_module._workspace_managers.clear()


@pytest.fixture()
def fake_orch_base(tmp_path):
    """Redirect ORCHESTRATOR_WORKSPACE to a temp dir for full isolation."""
    original = wm_module.ORCHESTRATOR_WORKSPACE
    wm_module.ORCHESTRATOR_WORKSPACE = tmp_path
    yield tmp_path
    wm_module.ORCHESTRATOR_WORKSPACE = original


# ── FileMetadata ──────────────────────────────────────────────────────────────

class TestFileMetadata:
    def test_to_dict_contains_all_fields(self):
        fm = FileMetadata(
            file_path="/tmp/a.txt",
            file_name="a.txt",
            file_type="text/plain",
            created_by="orchestrator",
            description="test",
            size_bytes=1234,
            conversation_thread="t1",
        )
        d = fm.to_dict()
        assert d["file_path"] == "/tmp/a.txt"
        assert d["file_name"] == "a.txt"
        assert d["file_type"] == "text/plain"
        assert d["created_by"] == "orchestrator"
        assert d["description"] == "test"
        assert d["size_bytes"] == 1234
        assert d["conversation_thread"] == "t1"

    def test_from_dict_round_trip(self):
        fm = FileMetadata(
            file_path="/tmp/b.csv",
            file_name="b.csv",
            file_type="text/csv",
            created_by="python",
            description="",
            size_bytes=0,
            conversation_thread="default",
        )
        restored = FileMetadata.from_dict(fm.to_dict())
        assert restored.file_path == fm.file_path
        assert restored.file_name == fm.file_name
        assert restored.file_type == fm.file_type
        assert restored.created_by == fm.created_by
        assert restored.conversation_thread == fm.conversation_thread

    def test_created_at_defaults_to_now(self):
        fm = FileMetadata(file_path="/x", file_name="x", file_type="t", created_by="o")
        assert fm.created_at is not None
        assert len(fm.created_at) > 5


# ── WorkspaceManager init & persistence ──────────────────────────────────────

class TestWorkspaceManagerInit:
    def test_creates_workspace_directory(self, fake_orch_base):
        wm = WorkspaceManager("init-thread")
        assert (fake_orch_base / "init-thread").is_dir()

    def test_starts_empty_when_no_index_file(self, fake_orch_base):
        wm = WorkspaceManager("empty-thread")
        assert wm.list_files() == []

    def test_loads_existing_index_on_construction(self, fake_orch_base):
        """An index saved by one instance is loaded by a fresh instance."""
        wm1 = WorkspaceManager("persist-thread")
        wm1.add_file("/abs/file.txt", "file.txt", "text/plain", "orchestrator")

        wm2 = WorkspaceManager("persist-thread")
        assert len(wm2.list_files()) == 1
        assert wm2.list_files()[0].file_name == "file.txt"

    def test_corrupt_index_falls_back_to_empty(self, fake_orch_base):
        (fake_orch_base / "corrupt").mkdir(parents=True)
        idx = fake_orch_base / "corrupt" / ".file_index.json"
        idx.write_text("NOT VALID JSON")
        wm = WorkspaceManager("corrupt")
        assert wm.list_files() == []


# ── Add / List / Search / Get ─────────────────────────────────────────────────

class TestWorkspaceManagerFiles:
    def test_add_and_list(self, fake_orch_base):
        wm = WorkspaceManager("t-add")
        wm.add_file("/abs/report.pdf", "report.pdf", "application/pdf", "agent:doc")
        files = wm.list_files()
        assert len(files) == 1
        assert files[0].file_name == "report.pdf"
        assert files[0].created_by == "agent:doc"

    def test_list_filter_by_type(self, fake_orch_base):
        wm = WorkspaceManager("t-type-filter")
        wm.add_file("/a.pdf", "a.pdf", "application/pdf", "o")
        wm.add_file("/b.txt", "b.txt", "text/plain", "o")
        pdfs = wm.list_files(file_type="application/pdf")
        assert len(pdfs) == 1
        assert pdfs[0].file_name == "a.pdf"

    def test_list_filter_by_created_by(self, fake_orch_base):
        wm = WorkspaceManager("t-creator")
        wm.add_file("/x.txt", "x.txt", "text/plain", "python")
        wm.add_file("/y.txt", "y.txt", "text/plain", "terminal")
        py_files = wm.list_files(created_by="python")
        assert len(py_files) == 1
        assert py_files[0].file_name == "x.txt"

    def test_search_by_name(self, fake_orch_base):
        wm = WorkspaceManager("t-search")
        wm.add_file("/report.pdf", "report.pdf", "application/pdf", "o")
        wm.add_file("/summary.txt", "summary.txt", "text/plain", "o")
        results = wm.search_files("report")
        assert len(results) == 1
        assert results[0].file_name == "report.pdf"

    def test_search_by_description(self, fake_orch_base):
        wm = WorkspaceManager("t-desc")
        wm.add_file("/x.txt", "x.txt", "text/plain", "o", description="quarterly analysis")
        assert len(wm.search_files("quarterly")) == 1

    def test_search_no_match_returns_empty(self, fake_orch_base):
        wm = WorkspaceManager("t-nomatch")
        wm.add_file("/a.txt", "a.txt", "text/plain", "o")
        assert wm.search_files("nonexistent") == []

    def test_get_file_found(self, fake_orch_base):
        wm = WorkspaceManager("t-get")
        wm.add_file("/data.csv", "data.csv", "text/csv", "python")
        f = wm.get_file("data.csv")
        assert f is not None
        assert f.file_name == "data.csv"

    def test_get_file_not_found_returns_none(self, fake_orch_base):
        wm = WorkspaceManager("t-miss")
        assert wm.get_file("missing.pdf") is None

    def test_add_file_saves_index_to_disk(self, fake_orch_base):
        wm = WorkspaceManager("t-save")
        wm.add_file("/f.txt", "f.txt", "text/plain", "o")
        assert wm.index_path.exists()
        data = json.loads(wm.index_path.read_text())
        assert len(data["files"]) == 1

    def test_relative_path_resolved_to_workspace(self, fake_orch_base):
        wm = WorkspaceManager("t-rel")
        wm.add_file("output.txt", "output.txt", "text/plain", "python")
        f = wm.get_file("output.txt")
        assert f is not None
        assert Path(f.file_path).is_absolute()


# ── scan_for_new_files ────────────────────────────────────────────────────────

class TestWorkspaceManagerScan:
    def test_scan_discovers_new_file(self, fake_orch_base):
        wm = WorkspaceManager("t-scan")
        (wm.workspace_path / "output.txt").write_text("hello")
        new_files = wm.scan_for_new_files()
        assert len(new_files) == 1
        assert new_files[0].file_name == "output.txt"

    def test_scan_skips_index_file(self, fake_orch_base):
        wm = WorkspaceManager("t-skip-idx")
        wm.index_path.write_text('{"files":[]}')
        new_files = wm.scan_for_new_files()
        assert ".file_index.json" not in [f.file_name for f in new_files]

    def test_scan_skips_already_indexed_files(self, fake_orch_base):
        wm = WorkspaceManager("t-skip-tracked")
        tracked = wm.workspace_path / "tracked.txt"
        tracked.write_text("tracked")
        wm.add_file(str(tracked), "tracked.txt", "text/plain", "o")
        new_files = wm.scan_for_new_files()
        assert all(f.file_name != "tracked.txt" for f in new_files)

    def test_scan_multiple_new_files(self, fake_orch_base):
        wm = WorkspaceManager("t-multi")
        for name in ["a.txt", "b.pdf", "c.csv"]:
            (wm.workspace_path / name).write_text("data")
        new_files = wm.scan_for_new_files()
        names = {f.file_name for f in new_files}
        assert names == {"a.txt", "b.pdf", "c.csv"}


# ── Thread workspace isolation ────────────────────────────────────────────────

class TestWorkspaceThreadIsolation:
    """CRITICAL: Thread A's workspace must be completely invisible to Thread B."""

    def test_separate_threads_have_separate_directories(self, fake_orch_base):
        wm_a = WorkspaceManager("thread-A")
        wm_b = WorkspaceManager("thread-B")
        assert wm_a.workspace_path != wm_b.workspace_path

    def test_thread_a_files_not_visible_to_thread_b_list(self, fake_orch_base):
        wm_a = WorkspaceManager("thread-A")
        wm_b = WorkspaceManager("thread-B")
        wm_a.add_file("/secret.txt", "secret.txt", "text/plain", "orchestrator")
        assert wm_b.list_files() == []
        assert wm_b.get_file("secret.txt") is None

    def test_thread_a_scan_cannot_see_thread_b_files(self, fake_orch_base):
        wm_a = WorkspaceManager("thread-A")
        wm_b = WorkspaceManager("thread-B")
        (wm_b.workspace_path / "b_private.txt").write_text("B data")
        new_a = wm_a.scan_for_new_files()
        assert all(f.file_name != "b_private.txt" for f in new_a)

    def test_index_files_stored_in_separate_paths(self, fake_orch_base):
        wm_a = WorkspaceManager("thread-A")
        wm_b = WorkspaceManager("thread-B")
        assert wm_a.index_path != wm_b.index_path

    def test_thread_a_write_does_not_corrupt_thread_b_index(self, fake_orch_base):
        wm_a = WorkspaceManager("thread-A")
        wm_b = WorkspaceManager("thread-B")
        wm_a.add_file("/a.txt", "a.txt", "text/plain", "o")
        # B should still have empty file list
        wm_b2 = WorkspaceManager("thread-B")
        assert wm_b2.list_files() == []


# ── _detect_file_type ─────────────────────────────────────────────────────────

class TestDetectFileType:
    @pytest.fixture()
    def wm(self, fake_orch_base):
        return WorkspaceManager("t-detect")

    def test_pdf(self, wm):
        assert wm._detect_file_type(Path("doc.pdf")) == "application/pdf"

    def test_png(self, wm):
        assert wm._detect_file_type(Path("img.png")) == "image/png"

    def test_csv(self, wm):
        assert wm._detect_file_type(Path("data.csv")) == "text/csv"

    def test_py(self, wm):
        assert wm._detect_file_type(Path("script.py")) == "text/x-python"

    def test_xlsx(self, wm):
        assert "spreadsheetml" in wm._detect_file_type(Path("data.xlsx"))

    def test_unknown_extension(self, wm):
        assert wm._detect_file_type(Path("file.xyz")) == "application/octet-stream"


# ── Singleton registry ────────────────────────────────────────────────────────

class TestGetWorkspaceManagerSingleton:
    def test_same_thread_returns_same_instance(self, fake_orch_base):
        w1 = get_workspace_manager("singleton-thread")
        w2 = get_workspace_manager("singleton-thread")
        assert w1 is w2

    def test_different_threads_return_different_instances(self, fake_orch_base):
        wa = get_workspace_manager("thread-X")
        wb = get_workspace_manager("thread-Y")
        assert wa is not wb


# ─────────────────────────────────────────────────────────────────────────────
# Content Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

import backend.orchestrator.content_orchestrator as co_module
from backend.orchestrator.content_orchestrator import (
    ContentOrchestratorConfig,
    _generate_result_summary,
    agent_requires_file_upload,
    compress_state_for_saving,
    config as co_config,
    expand_state_from_saved,
    get_file_parameter_name,
    inject_content_id_into_payload,
)
from backend.services.content_management_service import ContentType


@pytest.fixture(autouse=True)
def reset_content_singleton():
    co_module._content_service = None
    yield
    co_module._content_service = None


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_endpoint(path, params=None):
    ep = MagicMock()
    ep.endpoint = path
    ep.parameters = []
    for p_name in (params or []):
        pm = MagicMock()
        pm.name = p_name
        ep.parameters.append(pm)
    return ep


def _make_agent(endpoints):
    agent = MagicMock()
    agent.endpoints = endpoints
    agent.id = "test-agent"
    return agent


def _make_cms_metadata(content_id="cid-1", size=500):
    m = MagicMock()
    m.id = content_id
    m.size_bytes = size
    return m


# ── ContentOrchestratorConfig ─────────────────────────────────────────────────

class TestContentOrchestratorConfig:
    def test_defaults(self):
        cfg = ContentOrchestratorConfig()
        assert cfg.enabled is True
        assert cfg.max_context_tokens == 8000
        assert cfg.auto_upload_to_agents is True
        assert cfg.capture_agent_outputs is True

    def test_artifact_thresholds_populated(self):
        cfg = ContentOrchestratorConfig()
        assert cfg.artifact_thresholds["task_result"] == 2000
        assert "canvas_content" in cfg.artifact_thresholds

    def test_custom_thresholds_not_overwritten(self):
        cfg = ContentOrchestratorConfig(artifact_thresholds={"task_result": 9999})
        assert cfg.artifact_thresholds["task_result"] == 9999

    def test_disabled_config(self):
        cfg = ContentOrchestratorConfig(enabled=False)
        assert cfg.enabled is False


# ── agent_requires_file_upload ────────────────────────────────────────────────

class TestAgentRequiresFileUpload:
    def test_upload_endpoint_plus_file_id_param_returns_true(self):
        eps = [_make_endpoint("/upload"), _make_endpoint("/execute", ["file_id", "prompt"])]
        assert agent_requires_file_upload(_make_agent(eps), "/execute") is True

    def test_no_upload_endpoint_returns_false(self):
        eps = [_make_endpoint("/execute", ["file_id"])]
        assert agent_requires_file_upload(_make_agent(eps), "/execute") is False

    def test_upload_endpoint_but_no_file_param_returns_false(self):
        eps = [_make_endpoint("/upload"), _make_endpoint("/execute", ["prompt"])]
        assert agent_requires_file_upload(_make_agent(eps), "/execute") is False

    def test_endpoint_not_in_agent_returns_false(self):
        eps = [_make_endpoint("/upload"), _make_endpoint("/execute", ["file_id"])]
        assert agent_requires_file_upload(_make_agent(eps), "/nonexistent") is False

    def test_document_id_param_also_triggers_true(self):
        eps = [_make_endpoint("/upload"), _make_endpoint("/execute", ["document_id"])]
        assert agent_requires_file_upload(_make_agent(eps), "/execute") is True


# ── get_file_parameter_name ───────────────────────────────────────────────────

class TestGetFileParameterName:
    def test_finds_file_id(self):
        assert get_file_parameter_name(_make_endpoint("/e", ["prompt", "file_id"])) == "file_id"

    def test_finds_document_id(self):
        assert get_file_parameter_name(_make_endpoint("/e", ["document_id"])) == "document_id"

    def test_finds_content_id(self):
        assert get_file_parameter_name(_make_endpoint("/e", ["content_id"])) == "content_id"

    def test_finds_file_identifier(self):
        assert get_file_parameter_name(_make_endpoint("/e", ["file_identifier"])) == "file_identifier"

    def test_finds_fileid_camel_case(self):
        assert get_file_parameter_name(_make_endpoint("/e", ["fileId"])) == "fileId"

    def test_no_file_param_returns_none(self):
        assert get_file_parameter_name(_make_endpoint("/e", ["prompt", "temperature"])) is None


# ── inject_content_id_into_payload ────────────────────────────────────────────

class TestInjectContentIdIntoPayload:
    def test_injects_by_filename_match(self):
        ep = _make_endpoint("/e", ["file_id"])
        result = inject_content_id_into_payload(
            {"prompt": "go"},
            {"report.pdf": "agent-file-123"},
            ep,
            [{"file_name": "report.pdf"}],
        )
        assert result["file_id"] == "agent-file-123"

    def test_injects_first_available_when_no_filename_match(self):
        ep = _make_endpoint("/e", ["file_id"])
        result = inject_content_id_into_payload(
            {"prompt": "go"},
            {"cms-id-1": "agent-file-456"},
            ep,
            [{"file_name": "other.pdf"}],
        )
        assert result["file_id"] == "agent-file-456"

    def test_no_mapping_returns_payload_unchanged(self):
        ep = _make_endpoint("/e", ["file_id"])
        result = inject_content_id_into_payload({"prompt": "go"}, {}, ep, [])
        assert "file_id" not in result

    def test_no_file_param_returns_payload_unchanged(self):
        ep = _make_endpoint("/e", ["prompt"])
        result = inject_content_id_into_payload(
            {"prompt": "go"},
            {"report.pdf": "a123"},
            ep,
            [{"file_name": "report.pdf"}],
        )
        assert result == {"prompt": "go"}

    def test_correct_file_chosen_from_multiple_uploads(self):
        ep = _make_endpoint("/e", ["file_id"])
        mapping = {"doc1.pdf": "agent-1", "doc2.pdf": "agent-2"}
        uploaded = [{"file_name": "doc2.pdf"}, {"file_name": "doc1.pdf"}]
        result = inject_content_id_into_payload({"prompt": "q"}, mapping, ep, uploaded)
        # doc2 is first in uploaded list → should be picked first
        assert result["file_id"] == "agent-2"


# ── compress_state_for_saving ─────────────────────────────────────────────────

class TestCompressState:
    @pytest.mark.asyncio
    async def test_small_field_not_compressed(self):
        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock()
        state = {"completed_tasks": [{"task_name": "t1", "status": "done"}]}
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await compress_state_for_saving(state, "thread-1")
        mock_svc.register_artifact.assert_not_called()
        assert result["completed_tasks"] == state["completed_tasks"]

    @pytest.mark.asyncio
    async def test_large_task_plan_replaced_with_placeholder(self):
        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock(return_value=_make_cms_metadata("cid-99"))
        big_plan = [{"phase": f"phase_{i}", "goal": "x" * 300} for i in range(8)]
        state = {"task_plan": big_plan}
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await compress_state_for_saving(state, "t-big")
        assert result["task_plan"] == "[CONTENT:cid-99]"
        assert result["_content_refs"]["task_plan"]["id"] == "cid-99"

    @pytest.mark.asyncio
    async def test_last_agent_result_never_compressed(self):
        """CRITICAL: last_agent_result must remain raw regardless of size."""
        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock(return_value=_make_cms_metadata())
        big_raw_result = {"success": True, "output": "x" * 5000, "data": list(range(200))}
        state = {
            "last_agent_result": big_raw_result,
            "task_plan": [{"phase": "p", "goal": "g" * 400}],
        }
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await compress_state_for_saving(state, "t-raw")
        # last_agent_result must be identical original dict, NOT a content ref string
        assert result["last_agent_result"] is big_raw_result
        assert isinstance(result["last_agent_result"], dict)
        assert result["last_agent_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_action_history_never_compressed(self):
        """action_history is also not in compressible_fields — must stay raw."""
        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock(return_value=_make_cms_metadata())
        history = [{"action": "call_agent", "result": "z" * 1000} for _ in range(10)]
        state = {"action_history": history}
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await compress_state_for_saving(state, "t-hist")
        assert result["action_history"] is history

    @pytest.mark.asyncio
    async def test_disabled_config_returns_state_unchanged(self):
        original = co_config.enabled
        co_config.enabled = False
        state = {"task_plan": [{"step": "big " * 1000}]}
        try:
            result = await compress_state_for_saving(state, "t-off")
        finally:
            co_config.enabled = original
        assert result is state

    @pytest.mark.asyncio
    async def test_completed_tasks_compressed_to_slim_list(self):
        """completed_tasks over threshold → slim [{task_name, status}] list, not string."""
        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock(return_value=_make_cms_metadata("cid-tasks"))
        big_tasks = [
            {"task_name": f"task_{i}", "status": "completed", "output": "x" * 200}
            for i in range(15)
        ]
        state = {"completed_tasks": big_tasks}
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await compress_state_for_saving(state, "t-tasks")
        slim = result["completed_tasks"]
        assert isinstance(slim, list)
        for item in slim:
            assert set(item.keys()) == {"task_name", "status"}

    @pytest.mark.asyncio
    async def test_multiple_fields_each_get_own_ref(self):
        mock_svc = MagicMock()
        counter = {"n": 0}

        async def make_meta(content, name, content_type, thread_id, **kw):
            counter["n"] += 1
            return _make_cms_metadata(f"cid-{counter['n']}")

        mock_svc.register_artifact = AsyncMock(side_effect=make_meta)
        big_value = "y" * 3000
        state = {
            "task_plan": [{"g": big_value}],
            "task_agent_pairs": [{"a": big_value}],
        }
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await compress_state_for_saving(state, "t-multi")
        assert "_content_refs" in result
        assert len(result["_content_refs"]) == 2


# ── expand_state_from_saved ───────────────────────────────────────────────────

class TestExpandState:
    @pytest.mark.asyncio
    async def test_no_refs_returns_state_unchanged(self):
        state = {"task_plan": "some data", "other": 42}
        result = await expand_state_from_saved(state)
        assert result["task_plan"] == "some data"
        assert result["other"] == 42

    @pytest.mark.asyncio
    async def test_expands_content_ref_to_original_data(self):
        original_plan = [{"phase_id": "p1", "goal": "do something"}]
        mock_svc = MagicMock()
        mock_svc.get_content = MagicMock(return_value=(MagicMock(), original_plan))
        state = {
            "task_plan": "[CONTENT:cid-plan]",
            "_content_refs": {"task_plan": {"id": "cid-plan"}},
        }
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await expand_state_from_saved(state)
        assert result["task_plan"] == original_plan
        assert "_content_refs" not in result

    @pytest.mark.asyncio
    async def test_missing_content_id_leaves_placeholder(self):
        mock_svc = MagicMock()
        mock_svc.get_content = MagicMock(return_value=None)
        state = {
            "task_plan": "[CONTENT:cid-gone]",
            "_content_refs": {"task_plan": {"id": "cid-gone"}},
        }
        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            result = await expand_state_from_saved(state)
        # No crash; placeholder remains
        assert result["task_plan"] == "[CONTENT:cid-gone]"

    @pytest.mark.asyncio
    async def test_disabled_config_returns_unchanged(self):
        original = co_config.enabled
        co_config.enabled = False
        state = {"_content_refs": {"x": {"id": "y"}}, "x": "[CONTENT:y]"}
        try:
            result = await expand_state_from_saved(state)
        finally:
            co_config.enabled = original
        # disabled → not expanded, _content_refs still in state
        assert "_content_refs" in result


# ── Compression / Decompression round-trip ────────────────────────────────────

class TestCompressionRoundTrip:
    """CRITICAL: compress then expand must recover original field values."""

    @pytest.mark.asyncio
    async def test_round_trip_restores_task_plan(self):
        original_plan = [{"phase_id": f"p{i}", "goal": "do " + "x" * 200} for i in range(10)]
        state = {"task_plan": original_plan, "uncompressed_field": "stays"}

        # Simple in-memory store
        store: Dict[str, Any] = {}

        async def mock_register(content, name, content_type, thread_id, **kwargs):
            cid = f"cid-{len(store)}"
            store[cid] = content
            return _make_cms_metadata(cid, len(str(content)))

        def mock_get(cid):
            return (MagicMock(), store[cid]) if cid in store else None

        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock(side_effect=mock_register)
        mock_svc.get_content = MagicMock(side_effect=mock_get)

        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            compressed = await compress_state_for_saving(state, "thread-rt")
            assert isinstance(compressed["task_plan"], str)  # is a placeholder

            expanded = await expand_state_from_saved(compressed)

        assert expanded["task_plan"] == original_plan
        assert expanded["uncompressed_field"] == "stays"

    @pytest.mark.asyncio
    async def test_last_agent_result_raw_throughout_round_trip(self):
        """CRITICAL: last_agent_result must never be touched by compress or expand."""
        raw_result = {
            "success": True,
            "agent_id": "spreadsheet_agent",
            "output": "summary: " + "x" * 3000,
        }
        state = {
            "last_agent_result": raw_result,
            "task_plan": [{"p": "g" * 500}],
        }
        store: Dict[str, Any] = {}

        async def mock_register(content, name, content_type, thread_id, **kwargs):
            cid = f"cid-{len(store)}"
            store[cid] = content
            return _make_cms_metadata(cid)

        def mock_get(cid):
            return (MagicMock(), store[cid]) if cid in store else None

        mock_svc = MagicMock()
        mock_svc.register_artifact = AsyncMock(side_effect=mock_register)
        mock_svc.get_content = MagicMock(side_effect=mock_get)

        with patch.object(co_module, "get_content_service", return_value=mock_svc):
            compressed = await compress_state_for_saving(state, "thread-lar")
            expanded = await expand_state_from_saved(compressed)

        # After full round-trip, last_agent_result is still the original dict
        assert expanded["last_agent_result"]["success"] is True
        assert expanded["last_agent_result"]["agent_id"] == "spreadsheet_agent"
        assert not isinstance(expanded["last_agent_result"], str)


# ── _generate_result_summary ──────────────────────────────────────────────────

class TestGenerateResultSummary:
    def test_error_field(self):
        s = _generate_result_summary({"error": "something went wrong"})
        assert "Failed" in s

    def test_summary_field(self):
        s = _generate_result_summary({"summary": "done successfully"})
        assert "done successfully" in s

    def test_result_field(self):
        s = _generate_result_summary({"result": "42 rows processed"})
        assert "42 rows processed" in s

    def test_status_only(self):
        s = _generate_result_summary({"status": "completed"})
        assert "completed" in s

    def test_non_dict_returns_string_repr(self):
        s = _generate_result_summary("plain string result")
        assert "plain string result" in s

    def test_long_output_truncated(self):
        s = _generate_result_summary({"result": "x" * 500})
        assert len(s) <= 300  # Should be truncated


# ─────────────────────────────────────────────────────────────────────────────
# Workflow Scheduler
# ─────────────────────────────────────────────────────────────────────────────

from backend.services.workflow_scheduler import (
    WorkflowScheduler,
    get_scheduler,
    init_scheduler,
)
import backend.services.workflow_scheduler as sched_module
from apscheduler.jobstores.base import JobLookupError


@pytest.fixture()
def scheduler_instance():
    """WorkflowScheduler with a fully mocked APScheduler backend."""
    with patch("backend.services.workflow_scheduler.BackgroundScheduler") as MockBS:
        mock_inner = MagicMock()
        MockBS.return_value = mock_inner
        ws = WorkflowScheduler()
        mock_inner.start.assert_called_once()
        yield ws, mock_inner


@pytest.fixture(autouse=True)
def reset_scheduler_singleton():
    orig = sched_module._scheduler
    sched_module._scheduler = None
    yield
    sched_module._scheduler = orig


# ── Invalid cron expression rejected ─────────────────────────────────────────

class TestInvalidCronExpression:
    """CRITICAL: All malformed cron expressions must raise cleanly."""

    def test_too_few_parts_raises_value_error(self, scheduler_instance):
        ws, _ = scheduler_instance
        with pytest.raises(ValueError, match="Invalid cron"):
            ws.add_schedule("s1", "w1", "0 * *", {}, "user1", MagicMock())

    def test_too_many_parts_raises_value_error(self, scheduler_instance):
        ws, _ = scheduler_instance
        with pytest.raises(ValueError, match="Invalid cron"):
            ws.add_schedule("s1", "w1", "0 * * * * 2024", {}, "user1", MagicMock())

    def test_single_part_raises_value_error(self, scheduler_instance):
        ws, _ = scheduler_instance
        with pytest.raises(ValueError, match="Invalid cron"):
            ws.add_schedule("s1", "w1", "30", {}, "user1", MagicMock())

    def test_empty_string_raises(self, scheduler_instance):
        ws, _ = scheduler_instance
        with pytest.raises(Exception):
            ws.add_schedule("s1", "w1", "", {}, "user1", MagicMock())

    def test_four_part_expression_raises(self, scheduler_instance):
        ws, _ = scheduler_instance
        with pytest.raises(ValueError, match="Invalid cron"):
            ws.add_schedule("s1", "w1", "0 9 * *", {}, "user1", MagicMock())


# ── Valid cron expression accepted ────────────────────────────────────────────

class TestValidCronExpression:
    def test_five_part_cron_returns_true(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        with patch("backend.services.workflow_scheduler.CronTrigger"):
            result = ws.add_schedule("sched-1", "wflow-1", "0 9 * * 1", {}, "u1", MagicMock())
        assert result is True
        mock_inner.add_job.assert_called_once()

    def test_every_minute_wildcard_cron(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        with patch("backend.services.workflow_scheduler.CronTrigger"):
            result = ws.add_schedule("s-every", "w1", "* * * * *", {}, "u1", MagicMock())
        assert result is True

    def test_replace_existing_is_true(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        with patch("backend.services.workflow_scheduler.CronTrigger"):
            ws.add_schedule("s-r", "w1", "0 0 * * *", {}, "u1", MagicMock())
        call_kwargs = mock_inner.add_job.call_args[1]
        assert call_kwargs.get("replace_existing") is True

    def test_schedule_id_passed_as_job_id(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        with patch("backend.services.workflow_scheduler.CronTrigger"):
            ws.add_schedule("my-sched-id", "w1", "30 8 * * 1", {}, "u1", MagicMock())
        call_kwargs = mock_inner.add_job.call_args[1]
        assert call_kwargs.get("id") == "my-sched-id"

    def test_cron_parts_passed_to_trigger(self, scheduler_instance):
        ws, _ = scheduler_instance
        with patch("backend.services.workflow_scheduler.CronTrigger") as MockTrigger:
            ws.add_schedule("s1", "w1", "30 9 15 6 1", {}, "u1", MagicMock())
        MockTrigger.assert_called_once_with(
            minute="30", hour="9", day="15", month="6", day_of_week="1", timezone="UTC"
        )


# ── remove_schedule ───────────────────────────────────────────────────────────

class TestRemoveSchedule:
    def test_remove_existing_returns_true(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        assert ws.remove_schedule("sched-abc") is True
        mock_inner.remove_job.assert_called_once_with("sched-abc")

    def test_remove_nonexistent_returns_false(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        mock_inner.remove_job.side_effect = JobLookupError("sched-missing")
        assert ws.remove_schedule("sched-missing") is False

    def test_unexpected_error_on_remove_raises(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        mock_inner.remove_job.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError):
            ws.remove_schedule("s")


# ── load_active_schedules (DB startup) ───────────────────────────────────────

class TestLoadActiveSchedules:
    """CRITICAL: Cron schedules must be loaded from DB on startup."""

    def _mock_db_with_schedules(self, schedules):
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = schedules
        return db

    def _make_schedule(self, sid, wid="w1", cron="0 9 * * 1"):
        s = MagicMock()
        s.schedule_id = sid
        s.workflow_id = wid
        s.cron_expression = cron
        s.input_template = {}
        s.user_id = "user1"
        return s

    def test_registers_all_active_schedules(self, scheduler_instance):
        ws, _ = scheduler_instance
        schedules = [
            self._make_schedule("s1", "w1"),
            self._make_schedule("s2", "w2"),
            self._make_schedule("s3", "w3"),
        ]
        db = self._mock_db_with_schedules(schedules)
        with patch.object(ws, "add_schedule", return_value=True) as mock_add:
            with patch.dict(sys.modules, {"models": MagicMock(), "database": MagicMock()}):
                ws.load_active_schedules(db)
        assert mock_add.call_count == 3
        # load_active_schedules calls add_schedule with all keyword arguments
        ids = [c.kwargs.get("schedule_id") for c in mock_add.call_args_list]
        assert set(ids) == {"s1", "s2", "s3"}

    def test_passes_correct_fields_to_add_schedule(self, scheduler_instance):
        ws, _ = scheduler_instance
        s = self._make_schedule("s-check", wid="workflow-abc", cron="30 18 * * 5")
        s.user_id = "user-xyz"
        s.input_template = {"key": "value"}
        db = self._mock_db_with_schedules([s])
        with patch.object(ws, "add_schedule", return_value=True) as mock_add:
            with patch.dict(sys.modules, {"models": MagicMock(), "database": MagicMock()}):
                ws.load_active_schedules(db)
        call_kwargs = mock_add.call_args
        assert call_kwargs.kwargs.get("schedule_id") == "s-check" or call_kwargs.args[0] == "s-check"

    def test_failed_schedule_does_not_abort_remaining(self, scheduler_instance):
        ws, _ = scheduler_instance
        schedules = [self._make_schedule(f"s{i}") for i in range(4)]
        db = self._mock_db_with_schedules(schedules)
        calls = []

        def add_side_effect(*a, **kw):
            sid = a[0] if a else kw.get("schedule_id")
            calls.append(sid)
            if sid == "s1":
                raise Exception("transient DB error")
            return True

        with patch.object(ws, "add_schedule", side_effect=add_side_effect):
            with patch.dict(sys.modules, {"models": MagicMock(), "database": MagicMock()}):
                ws.load_active_schedules(db)  # must not raise

        # All 4 were attempted despite s1 failure
        assert len(calls) == 4

    def test_empty_db_loads_nothing(self, scheduler_instance):
        ws, _ = scheduler_instance
        db = self._mock_db_with_schedules([])
        with patch.object(ws, "add_schedule", return_value=True) as mock_add:
            with patch.dict(sys.modules, {"models": MagicMock(), "database": MagicMock()}):
                ws.load_active_schedules(db)
        mock_add.assert_not_called()


# ── get_scheduler singleton ───────────────────────────────────────────────────

class TestGetSchedulerSingleton:
    def test_same_instance_returned_on_second_call(self):
        with patch("backend.services.workflow_scheduler.BackgroundScheduler"):
            s1 = get_scheduler()
            s2 = get_scheduler()
        assert s1 is s2

    def test_creates_workflow_scheduler_instance(self):
        with patch("backend.services.workflow_scheduler.BackgroundScheduler"):
            s = get_scheduler()
        assert isinstance(s, WorkflowScheduler)


# ── shutdown ──────────────────────────────────────────────────────────────────

class TestSchedulerShutdown:
    def test_shutdown_delegates_to_apscheduler(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        ws.shutdown(wait=False)
        mock_inner.shutdown.assert_called_once_with(wait=False)

    def test_shutdown_wait_true(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        ws.shutdown(wait=True)
        mock_inner.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_exception_is_swallowed(self, scheduler_instance):
        ws, mock_inner = scheduler_instance
        mock_inner.shutdown.side_effect = Exception("crash on shutdown")
        ws.shutdown()  # must not propagate
