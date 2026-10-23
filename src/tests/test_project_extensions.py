"""Small synthetic tests for project-only preprocessing extensions.

These tests intentionally do not test xIV-LDDMM optimization.  That requires the
configured repository and PyKeOps environment.  They test the pieces that this project
adds around xmodmap: affine-world geometry, 3-D block mass conservation, dense scalar
moments/bins, and the VTK coordinate adapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dense_image_utils import IntensityTransform, construct_dense_image_particles
from particle_utils import FeatureDefinition, construct_particles
from xmodmap_compat import XmodmapAPI, write_vtk_xyz, save_atlas_outputs, save_target_outputs


def test_categorical_mass_conservation() -> None:
    labels = np.zeros((5, 6, 7), dtype=np.int16)
    labels[1:4, 1:5, 1:6] = 1
    labels[2:5, 3:6, 2:7] = 2
    affine = np.array(
        [[0.2, 0, 0, 10], [0, 0.3, 0, -4], [0, 0, 0.4, 7], [0, 0, 0, 1]],
        dtype=float,
    )
    features = FeatureDefinition((1, 2), ("one", "two"), ("both",), (0, 0))
    for factor in (1, 2, 5):
        _, nu_sub, nu_all, diagnostics = construct_particles(
            labels, affine, features, factor, "affine_world"
        )
        assert np.allclose(nu_all.sum(), diagnostics["expected_total_mass_mm3"])
        assert np.allclose(nu_sub.sum(), nu_all.sum())


def test_dense_histogram_mass_conservation() -> None:
    data = np.arange(5 * 6 * 7, dtype=float).reshape(5, 6, 7)
    mask = data > 20
    affine = np.array(
        [[0.2, 0, 0, 10], [0, 0.3, 0, -4], [0, 0, 0.4, 7], [0, 0, 0, 1]],
        dtype=float,
    )
    transform = IntensityTransform("fixed_range", 0.0, float(data.max()), True)
    expected = mask.sum() * abs(np.linalg.det(affine[:3, :3]))
    for factor in (1, 2, 5):
        _, nu_intensity, nu_bin, _, _ = construct_dense_image_particles(
            data,
            affine,
            mask,
            transform,
            factor,
            "affine_world",
            "block_center",
            {"enabled": True, "bins": 8, "assignment": "linear"},
        )
        assert nu_bin is not None
        assert np.allclose(nu_intensity[:, 0].sum(), expected)
        assert np.allclose(nu_bin.sum(), expected, rtol=1e-6)


def test_vtk_xyz_adapter_cancels_xmodmap_swap() -> None:
    captured: dict[str, np.ndarray] = {}

    def historical_write(points, arrays, names, filename):
        points = np.asarray(points)
        captured["vtk_points"] = points[:, [1, 0, 2]]

    api = XmodmapAPI(
        xmodmap=None,
        get_from_file=None,
        make_from_single_channel_image=None,
        make_bins_from_multichannel_image=None,
        get_entropy=None,
        get_mass_ratio=None,
        write_particle_vtk=None,
        write_vtk=historical_write,
        make_pq=None,
        resize_data=None,
    )
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_vtk_xyz(api, xyz, [np.ones(2)], ["test"], "unused.vtk")
    assert np.array_equal(captured["vtk_points"], xyz)


def test_atlas_reference_fields_use_mapped_features_without_support_censoring(tmp_path):
    D = torch.tensor([[1., 2.], [3., 4.]])
    atlas = torch.tensor([[0., 4.], [2., 0.]])
    mapped = atlas @ torch.tensor([[1., 3.], [2., 0.]])
    ratio = torch.tensor([2., .5])
    summary = dict(D=D, nu_D=atlas, nu_Dpi=mapped, mass_ratio_proxy=ratio,
                   relative_support_weight=torch.tensor([[0.], [.25]]))
    particles, scalars = {}, {}
    def particle(points, mass, filename, **kwargs):
        particles[Path(filename).name] = (points, mass, kwargs)
    def vtk(points, values, names, filename):
        scalars.update(zip(names, values))
    def mapped_entropy(masses):
        np.testing.assert_array_equal(masses, mapped.numpy())
        return np.array([0., -(0.25*np.log(.25)+.75*np.log(.75))])
    save_atlas_outputs(particle, vtk, mapped_entropy, summary, tmp_path, ['a', 'b'], ['x', 'y'])
    np.testing.assert_array_equal(scalars['Weight_AtlasFeatures'], [4., 2.])
    np.testing.assert_array_equal(scalars['MaxVal_AtlasFeatures'], [2, 1])
    np.testing.assert_array_equal(scalars['Weight_TargetFeatures'], [8., 8.])
    np.testing.assert_array_equal(scalars['MaxVal_TargetFeatures'], [1, 2])
    np.testing.assert_allclose(scalars['Entropy'], [0., -(0.25*np.log(.25)+.75*np.log(.75))])
    np.testing.assert_array_equal(scalars['Jacobian'], ratio)
    np.testing.assert_array_equal(scalars['MassRatioProxy'], ratio)
    assert particles['atlas_nu_Dpi.vtk'][2]['featNames'] == ['x', 'y']
    assert particles['atlas_nu_Dpi.vtk'][2]['sW'] is None
    saved = torch.load(tmp_path/'atlas_deformationSummary.pt', weights_only=False)
    torch.testing.assert_close(saved['D'], D)
    torch.testing.assert_close(saved['nu_Dpi'], mapped)
    torch.testing.assert_close(saved['jac'], ratio)


def test_single_target_particle_keeps_vector_fields_and_suffix(tmp_path):
    calls = []
    def vtk(points, values, names, filename):
        assert all(v.shape == (1,) for v in values)
        calls.append((Path(filename).name, dict(zip(names, values))))
    state = dict(Td=torch.tensor([[4., 5.]]), nu_Td=torch.tensor([[0., 2.]]),
                 mass_ratio_proxy=torch.tensor([.5]))
    save_target_outputs(lambda *args, **kwargs: None, vtk, lambda masses: np.zeros(len(masses)), state, tmp_path, '_qc')
    name, values = calls[0]
    assert name == 'target_deformationSummary_qc.vtk'
    np.testing.assert_array_equal(values['MaxVal_TargetFeatures'], [2])
    np.testing.assert_array_equal(values['Jacobian'], [.5])
    np.testing.assert_array_equal(values['Entropy'], [0.])
    saved = torch.load(tmp_path/'target_deformationSummary_qc.pt', weights_only=False)
    torch.testing.assert_close(saved['jac'], state['mass_ratio_proxy'])
