#!/usr/bin/env python

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from paper_writing.python import viz_vpr_data


def test_find_valid_matches_returns_best_and_all_valid_pairs() -> None:
    test_ds = SimpleNamespace(
        num_queries=1,
        num_database=3,
        queries_image_names=['query0'],
        database_image_names=['db0', 'db1', 'db2'],
        queries_poses={'query0': np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
        database_poses={
            'db0': np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            'db1': np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            'db2': np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        },
    )

    pose_errors = [(3.0, 5.0), (1.0, 5.0), (10.0, 5.0)]

    with (
        patch.object(viz_vpr_data, 'convert_vec_to_matrix', return_value=np.eye(4)),
        patch.object(
            viz_vpr_data,
            'convert_matrix_to_vec',
            return_value=(np.zeros(3), np.array([0, 0, 0, 1])),
        ),
        patch.object(viz_vpr_data, 'compute_pose_error', side_effect=pose_errors),
    ):
        best_valid_pairs, valid_pairs = viz_vpr_data.find_valid_matches(
            test_ds,
            trans_thresh=5.0,
            rot_thresh=10.0,
        )

    assert best_valid_pairs == [(0, 1)]
    assert valid_pairs == [(0, 0), (0, 1)]


def test_parse_submission_pairs_maps_names_to_indices(tmp_path: Path) -> None:
    submission_path = tmp_path / 'submission.txt'
    submission_path.write_text(
        '\n'.join([
            'query_a db_b 0.9 1',
            'unknown_query db_a 0.8 1',
            '',
            'query_b db_a 0.7 0',
            'query_b db_b',
            'query_a',
        ]),
        encoding='utf-8',
    )

    pairs = viz_vpr_data.parse_submission_pairs(
        submission_path,
        query_names=['query_a', 'query_b'],
        db_names=['db_a', 'db_b'],
    )

    assert pairs == [(0, 1), (1, 0)]


def test_parse_submission_pairs_can_filter_discarded_master_matches(tmp_path: Path) -> None:
    submission_path = tmp_path / 'submission.txt'
    submission_path.write_text(
        '\n'.join([
            'query_a db_a 908.000000 1',
            'query_b db_b 10.000000 0',
        ]),
        encoding='utf-8',
    )

    pairs = viz_vpr_data.parse_submission_pairs(
        submission_path,
        query_names=['query_a', 'query_b'],
        db_names=['db_a', 'db_b'],
        keep_only_positive_label=True,
    )

    assert pairs == [(0, 0)]


def test_visualize_vpr_data_dmatrix_writes_png_and_pdf(tmp_path: Path) -> None:
    output_path = tmp_path / 'dmatrix'

    viz_vpr_data.visualize_vpr_data_dmatrix(
        D_all=np.array([[0.1, 0.2], [0.3, 0.4]]),
        gt_pairs=[(0, 0), (1, 1)],
        singlematch_pairs=[(1, 0)],
        seqslam_pairs=[(0, 1)],
        proposed_pairs=[(1, 0)],
        proposed_gv_pairs=[(0, 0)],
        output_path=output_path,
    )

    assert output_path.with_suffix('.png').exists()
    assert output_path.with_suffix('.pdf').exists()


def test_visualize_graph_master_dmatrix_writes_png_and_pdf(tmp_path: Path) -> None:
    output_path = tmp_path / 'dmatrix_graph_master'

    viz_vpr_data.visualize_graph_master_dmatrix(
        D_all=np.array([[0.1, 0.2], [0.3, 0.4]]),
        gt_pairs=[(0, 0), (1, 1)],
        none_pairs=[(0, 1)],
        master_pairs=[(1, 0)],
        output_path=output_path,
    )

    assert output_path.with_suffix('.png').exists()
    assert output_path.with_suffix('.pdf').exists()


def test_evaluate_vpr_system_uses_precomputed_dmatrix_and_best_pairs(tmp_path: Path) -> None:
    database_folder = tmp_path / 'out_map_db'
    query_folder = tmp_path / 'out_map_query'
    dmatrix_dir = tmp_path / 'dmatrix'
    singlematch_dir = tmp_path / 'singlematch'
    seqmatch_dir = tmp_path / 'seqmatch'
    graph_dir = tmp_path / 'graph'
    graph_none_dir = tmp_path / 'graph_none'
    graph_master_dir = tmp_path / 'graph_master'
    output_dir = tmp_path / 'output'
    for folder in (
        database_folder,
        query_folder,
        dmatrix_dir,
        singlematch_dir,
        seqmatch_dir,
        graph_dir,
        graph_none_dir,
        graph_master_dir,
    ):
        folder.mkdir()

    D_all = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.save(dmatrix_dir / 'D_all_out_map_query.npy', D_all)
    submission_name = 'submission-query-db.txt'
    (singlematch_dir / submission_name).write_text('query_a db_a 0.9 1\n', encoding='utf-8')
    (seqmatch_dir / submission_name).write_text('query_a db_b 0.9 1\n', encoding='utf-8')
    (graph_dir / submission_name).write_text('query_b db_a 0.8 1\n', encoding='utf-8')
    (graph_none_dir / submission_name).write_text('query_a db_b 0.8 0\n', encoding='utf-8')
    (graph_master_dir / submission_name).write_text(
        'query_b db_b 0.9 1\n'
        'query_a db_a 0.1 0\n',
        encoding='utf-8',
    )

    test_ds = SimpleNamespace(
        num_queries=2,
        num_database=2,
        queries_image_names=['query_a', 'query_b'],
        database_image_names=['db_a', 'db_b'],
    )
    args = SimpleNamespace(
        database_folder=database_folder,
        queries_folder=[query_folder],
        image_size=224,
        trans_thresh=5.0,
        rot_thresh=10.0,
        z_ratio=0.2,
        output_path=output_dir,
        dmatrix_dir=dmatrix_dir,
        singlematch_dir=singlematch_dir,
        seqmatch_dir=seqmatch_dir,
        graph_dir=graph_dir,
        graph_none_dir=graph_none_dir,
        graph_master_dir=graph_master_dir,
    )

    with (
        patch.object(viz_vpr_data, 'TestDataset', return_value=test_ds),
        patch.object(
            viz_vpr_data,
            'load_poses',
            return_value=(np.zeros((2, 7)), np.zeros((2, 7))),
        ),
        patch.object(
            viz_vpr_data,
            'find_valid_matches',
            return_value=([(0, 1)], [(0, 0), (0, 1)]),
        ),
        patch.object(viz_vpr_data, 'visualize_vpr_data_dmatrix') as mock_dmatrix,
        patch.object(viz_vpr_data, 'visualize_graph_master_dmatrix') as mock_graph_master,
        patch.object(viz_vpr_data, 'visualize_vpr_data_queries') as mock_queries,
    ):
        viz_vpr_data.evaluate_vpr_system(args)

    mock_dmatrix.assert_called_once()
    dmatrix_args = mock_dmatrix.call_args.args
    np.testing.assert_array_equal(dmatrix_args[0], D_all)
    assert dmatrix_args[1] == [(0, 0), (0, 1)]
    assert dmatrix_args[2] == [(0, 0)]
    assert dmatrix_args[3] == [(0, 1)]
    assert dmatrix_args[4] == [(1, 0)]
    assert dmatrix_args[5] == [(1, 1)]
    assert dmatrix_args[6] == output_dir / 'dmatrix_out_map_query.png'

    mock_graph_master.assert_called_once()
    graph_master_args = mock_graph_master.call_args.args
    np.testing.assert_array_equal(graph_master_args[0], D_all)
    assert graph_master_args[1] == [(0, 0), (0, 1)]
    assert graph_master_args[2] == [(0, 1)]
    assert graph_master_args[3] == [(1, 1)]
    assert graph_master_args[4] == output_dir / 'dmatrix_graph_master_out_map_query.png'

    mock_queries.assert_called_once()
    assert mock_queries.call_args.args[2] == [[(0, 1)]]
