import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT, SYNTHETIC_MAP


@pytest.fixture
def real_settings(tmp_path: Path):
    from navmap_console.config import load_settings

    return load_settings({"NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path), "NAVMAP_CONSOLE_REPO": str(REPO_ROOT),
                          "NAVMAP_CONSOLE_PYTHON": "/opt/conda/envs/x/bin/python", "MERGE_CPU_LIST": "2,3"})


def test_build_env_pins_preload_and_pythonpath(real_settings):
    from navmap_console.jobs.env import build_env, command_prefix, resolve_cpu_list

    env = build_env(real_settings)
    assert env["LD_PRELOAD"] == "/opt/conda/envs/x/lib/libstdc++.so.6"
    assert env["MKL_THREADING_LAYER"] == "GNU"
    assert env["PYTHONPATH"].split(":") == [str(REPO_ROOT / "python"), str(REPO_ROOT / "third_party/litevloc_code/python"),
                                            str(REPO_ROOT / "third_party/pose_estimation_models")]
    assert env["PYTHONUNBUFFERED"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert resolve_cpu_list(real_settings) == "2,3"
    assert command_prefix("2,3")[:2] == ["taskset", "-c"] or command_prefix("2,3") == []
    assert command_prefix(None) == []


def test_params_schema_and_validation():
    from navmap_console.jobs.specs import MERGE_PARAMS, params_schema, validate_params

    schema = params_schema()
    by_name = {p["name"]: p for p in schema}
    assert by_name["pgo_robust"]["default"] == "gnc_gm" and by_name["pgo_robust"]["choices"] == ["none", "huber", "gnc_tls", "gnc_gm"]
    assert by_name["pgo_loop_sigma_trans"]["default"] == 0.1 and by_name["use_iqa"]["default"] is True
    assert len(schema) == len(MERGE_PARAMS)
    full = validate_params({"pgo_loop_sigma_trans": "0.2", "use_td": False})
    assert full["pgo_loop_sigma_trans"] == 0.2 and full["use_td"] is False and full["pgo_robust"] == "gnc_gm"
    with pytest.raises(ValueError, match="unknown"):
        validate_params({"nope": 1})
    with pytest.raises(ValueError, match="pgo_robust"):
        validate_params({"pgo_robust": "magic"})
    with pytest.raises(ValueError, match="vpr_match_seq_len"):
        validate_params({"vpr_match_seq_len": "ten"})


def test_merge_argv_reference_config(real_settings, tmp_path: Path):
    from navmap_console.jobs.specs import merge_argv

    argv = merge_argv(real_settings, submap_list=tmp_path / "inputs.txt", result_dir=tmp_path / "out",
                      image_size=[512, 288], params={"pgo_loop_sigma_rot": 2.0}, rerun_viz_dir=tmp_path / "rv")
    assert argv[:2] == ["/opt/conda/envs/x/bin/python", str(REPO_ROOT / "python" / "map_merge_pipeline.py")]
    s = " ".join(argv)
    for flag in ("--submap_list", "--result_dir", "--step_dir_style indexed", "--image_size 512 288",
                 "--pose_estimation_method master", "--vpr_match_model vpr_dp", "--vpr_match_seq_len 10",
                 "--pgo_robust gnc_gm", "--pgo_loop_sigma_trans 0.1", "--pgo_loop_sigma_rot 2.0",
                 "--pgo_loop_conf_scaling inverse", "--use_iqa", "--use_ig", "--use_td", "--viz", "--rerun-viz",
                 "--rerun-viz-dir"):
        assert flag in s, flag
    assert "--append_from" not in s
    argv2 = merge_argv(real_settings, submap_list=tmp_path / "i.txt", result_dir=tmp_path / "o", image_size=[512, 288],
                       params={"use_iqa": False}, append_from=tmp_path / "base", start_step=7)
    s2 = " ".join(argv2)
    assert "--append_from" in s2 and "--start_step 7" in s2 and "--use_iqa" not in s2 and "--use_ig" in s2


def test_fake_pipeline_produces_steps(tmp_path: Path):
    from navmap_console.jobs.specs import FAKE_SCRIPT, merge_argv
    from navmap_console.config import load_settings

    settings = load_settings({"NAVMAP_CONSOLE_DATA_ROOT": str(tmp_path), "NAVMAP_CONSOLE_REPO": str(REPO_ROOT),
                              "NAVMAP_CONSOLE_FAKE_PIPELINE": "1"})
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(f"{SYNTHETIC_MAP}\n{SYNTHETIC_MAP}\n")
    argv = merge_argv(settings, submap_list=inputs, result_dir=tmp_path / "out", image_size=[512, 288], params={})
    assert argv[1] == str(FAKE_SCRIPT)
    out = subprocess.run(argv, capture_output=True, text=True, env={"NAVMAP_CONSOLE_FAKE_STEP_SECONDS": "0.01", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert "--- Merging submap 0: synthetic_map ---" in out.stdout
    assert "STEP_DONE index=1 sid=synthetic_map" in out.stdout
    assert (tmp_path / "out" / "merge_001_synthetic_map" / "poses.txt").is_file()
    assert (tmp_path / "out" / "merge_finalmap").resolve() == (tmp_path / "out" / "merge_001_synthetic_map").resolve()
