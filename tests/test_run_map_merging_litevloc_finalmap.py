"""End-to-end wrapper test for delivering a LiteVLoc-ready final map."""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_map_merging.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(textwrap.dedent(contents))
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("consolidate", "expects_symlink"),
    [("1", False), ("0", True)],
)
def test_final_map_delivery_mode(tmp_path: Path, consolidate: str, expects_symlink: bool) -> None:
    """The wrapper promotes a completed, self-contained directory at its public path."""
    output_root = tmp_path / "out"
    traj_root = tmp_path / "traj"
    fake_python = tmp_path / "fake_python.py"
    converter_log = tmp_path / "converter_inputs.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    _write_executable(
        fake_python,
        f"""\
        #!/usr/bin/env python3
        import os
        import shutil
        import sys
        from pathlib import Path

        script = Path(sys.argv[1]).name
        args = sys.argv[2:]
        if script == 'map_merge_pipeline.py':
            root = Path(args[args.index('--output_root') + 1])
            result = root / 's_results_in_method_iqaigtd'
            first, final = result / 'merge_0', result / 'merge_0_1'
            for step, image_id in ((first, 0), (final, 1)):
                (step / 'seq').mkdir(parents=True)
                (step / 'preds').mkdir()
                (step / 'seq' / f'{{image_id:06d}}.color.jpg').write_bytes(b'image')
            lines = {{
                'timestamps.txt': 'seq/000000.color.jpg 0\\nseq/000001.color.jpg 1\\n',
                'intrinsics.txt': ('seq/000000.color.jpg 1 1 0 0 1 1\\n'
                                   'seq/000001.color.jpg 1 1 0 0 1 1\\n'),
                'poses.txt': ('seq/000000.color.jpg 1 0 0 0 0 0 0\\n'
                              'seq/000001.color.jpg 1 0 0 0 0 0 0\\n'),
                'poses_abs_gt.txt': ('seq/000000.color.jpg 1 0 0 0 0 0 0\\n'
                                     'seq/000001.color.jpg 1 0 0 0 0 0 0\\n'),
                'gps_data.txt': 'seq/000000.color.jpg 0 0 0\\nseq/000001.color.jpg 0 0 0\\n',
                'iqa_data.txt': 'seq/000000.color.jpg 1\\nseq/000001.color.jpg 1\\n',
                'edges_covis.txt': '', 'edges_odom.txt': '', 'edges_trav.txt': '',
                'database_descriptors.txt': 'seq/000000.color.jpg 0\\nseq/000001.color.jpg 0\\n',
            }}
            for name, value in lines.items():
                (final / name).write_text(value)
            (final / 'submap_disc_0').mkdir()
            for name in ('poses.txt', 'poses_abs_gt.txt', 'timestamps.txt'):
                shutil.copy2(final / name, final / 'submap_disc_0' / name)
            (result / 'merge_finalmap').symlink_to(final)
        elif script == 'utils_convert_pose_format.py':
            Path(os.environ['CONVERTER_LOG']).open('a').write(args[args.index('--input_pose') + 1] + '\\n')
        elif script == 'map_merge_pack.py':
            os.execv(sys.executable, [sys.executable, sys.argv[1], *args])
        else:
            raise AssertionError(f'unexpected Python script: {{script}}')
        """,
    )
    _write_executable(
        fake_bin / "bash",
        """\
        #!/bin/sh
        case "$1" in
          */run_evaluation.sh) exit 0 ;;
        esac
        exec /bin/bash "$@"
        """,
    )

    env = dict(os.environ)
    env.update({
        "OUTPUT_ROOT": str(output_root),
        "TRAJ_EVAL_ROOT": str(traj_root),
        "PYTHON_OPENNAVMAP": str(fake_python),
        "EVAL_PYTHON": str(fake_python),
        "CONVERTER_LOG": str(converter_log),
        "MERGE_CPU_LIST": "",
        "CONSOLIDATE_LITEVLOC_MAP": consolidate,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    })
    completed = subprocess.run(
        ["/bin/bash", str(SCRIPT), "s", "0", "method", "master", "1", "1", "1"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result_dir = output_root / "s_results_in_method_iqaigtd"
    final_map = result_dir / "merge_finalmap"
    assert final_map.is_symlink() is expects_symlink
    if not expects_symlink:
        assert final_map.is_dir()
        assert (final_map / "seq" / "000000.color.jpg").is_file()
        assert (final_map / "seq" / "000001.color.jpg").is_file()
    assert converter_log.read_text().splitlines()[0] == str(result_dir / "merge_0_1" / "submap_disc_0" / "poses_abs_gt.txt")
