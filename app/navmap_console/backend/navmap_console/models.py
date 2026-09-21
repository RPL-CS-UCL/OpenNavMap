"""Pydantic models shared by routers and services (API schema = disk schema)."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class VprConfig(BaseModel):
    method: str = "cosplace"
    backbone: str = "ResNet18"
    dim: int = 256


class FileCheck(BaseModel):
    name: str
    required: bool
    status: Literal["ok", "missing", "incomplete", "invalid"]
    lines: Optional[int] = None
    detail: str = ""


class ValidationReport(BaseModel):
    ok: bool
    num_frames: int = 0
    has_gt: bool = False
    has_gps: bool = False
    has_iqa: bool = False
    descriptor_dim: Optional[int] = None
    files: List[FileCheck] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    checked_at: str = Field(default_factory=now_iso)


class RegionHead(BaseModel):
    run_id: str
    step_index: int
    session_ids: List[str] = Field(default_factory=list)
    lineage: List[str] = Field(default_factory=list)


class Region(BaseModel):
    id: str
    name: str
    description: str = ""
    vpr: VprConfig = Field(default_factory=VprConfig)
    image_size: List[int] = Field(default_factory=lambda: [512, 288])
    created_at: str = Field(default_factory=now_iso)
    head: Optional[RegionHead] = None


class RegionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    vpr: Optional[VprConfig] = None
    image_size: List[int] = Field(default_factory=lambda: [512, 288])


class RegionPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    description: Optional[str] = None
    vpr: Optional[VprConfig] = None
    image_size: Optional[List[int]] = None


class Session(BaseModel):
    id: str
    region_id: str
    name: str
    kind: Literal["submap", "raw"] = "submap"
    source: Literal["upload", "path", "imported"]
    path: str
    num_frames: int = 0
    has_gt: bool = False
    has_gps: bool = False
    has_iqa: bool = False
    validation: Optional[ValidationReport] = None
    created_at: str = Field(default_factory=now_iso)


class RegisterSessionRequest(BaseModel):
    path: str
    name: Optional[str] = None


# --- jobs and runs (spec §5.3 / §5.4) --------------------------------------

JobKind = Literal["merge", "append", "consolidate", "official_eval", "per_step_eval", "import_results", "export",
                  "export_verify"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "orphaned"]
QueueName = Literal["gpu", "cpu"]


class JobProgress(BaseModel):
    step: Optional[int] = None          # global step index currently being merged
    total: Optional[int] = None         # steps this job will produce
    stage_index: Optional[int] = None   # 1..8, see jobs/progress.py STAGE_NAMES
    stage: Optional[str] = None
    completed_steps: int = 0
    detail: str = ""


class Job(BaseModel):
    id: str
    kind: JobKind
    queue: QueueName
    status: JobStatus = "queued"
    region_id: Optional[str] = None
    run_id: Optional[str] = None
    argv: List[str]
    cwd: str
    env: Dict[str, str] = Field(default_factory=dict)  # overrides applied on top of os.environ
    cpu_list: Optional[str] = None
    log_path: str
    pid: Optional[int] = None
    process_create_time: Optional[float] = None
    log_offset: int = 0  # bytes of the log already parsed for progress; adoption resumes from here
    created_at: str = Field(default_factory=now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    crash_kind: Optional[str] = None
    error: Optional[str] = None
    progress: JobProgress = Field(default_factory=JobProgress)


class StepRecord(BaseModel):
    index: int
    session_id: str
    dir_name: str = ""
    status: Literal["running", "done", "failed"] = "running"
    id_offset: Optional[int] = None
    odom_nodes: Optional[int] = None
    covis_nodes: Optional[int] = None
    components: Optional[int] = None
    registry_edges: Optional[int] = None
    pgo_error_initial: Optional[float] = None
    pgo_error_final: Optional[float] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class LoopStats(BaseModel):
    total: int = 0
    new: int = 0
    hist: int = 0
    accepted: int = 0
    rejected_new: int = 0
    overturned_hist: int = 0


class StepSummary(BaseModel):
    """Per-step numbers derived from a merge_* directory (spec §5.6 GET .../steps summaries)."""
    index: int
    dir_name: str
    session_id: str
    status: str
    id_offset: int
    num_nodes: int
    num_covis_nodes: int
    num_new: int
    num_culled: int
    component_sizes: List[int] = Field(default_factory=list)
    edge_counts: Dict[str, int] = Field(default_factory=dict)
    history: Dict[str, int] = Field(default_factory=dict)
    precision: List[float] = Field(default_factory=list)
    recall: List[float] = Field(default_factory=list)
    loops: LoopStats = Field(default_factory=LoopStats)
    pgo_error_initial: Optional[float] = None
    pgo_error_final: Optional[float] = None
    max_displacement: float = 0.0
    num_moved: int = 0
    duration_s: Optional[float] = None
    has_dmatrix: bool = False
    has_pre_pgo: bool = False
    # per-step ATE from evaluations/per_step/step_XX/eval.json; reason is None when the numbers are valid
    ate_trans_rmse: Optional[float] = None
    ate_rot_rmse: Optional[float] = None
    ate_frames: Optional[int] = None
    ate_reason: Optional[str] = None


class RunParent(BaseModel):
    run_id: str
    step_index: int


RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "orphaned"]


class Run(BaseModel):
    id: str
    region_id: str
    name: str
    kind: Literal["merge", "append", "imported"] = "merge"
    parent: Optional[RunParent] = None
    start_step: int = 0
    session_ids: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)  # free-form: order preset, seed, notes
    status: RunStatus = "queued"
    job_id: Optional[str] = None
    git_commit: Optional[str] = None
    num_steps_expected: int = 0
    last_step_index: Optional[int] = None
    final_dir: Optional[str] = None
    final_error: Optional[str] = None
    created_at: str = Field(default_factory=now_iso)
    finished_at: Optional[str] = None


class RunCreate(BaseModel):
    name: Optional[str] = None
    kind: Literal["merge", "append"] = "merge"
    parent: Optional[RunParent] = None
    session_ids: List[str] = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ImportRequest(BaseModel):
    """Register a result directory written by scripts/run_map_merging.sh as an imported run."""
    result_dir: str
    sessions_root: Optional[str] = None
    name: Optional[str] = None
    promote: bool = False
