"""Pydantic models shared by routers and services (API schema = disk schema)."""
from datetime import datetime, timezone
from typing import List, Optional

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
