"""Synthetic mathematical checks before the bounded NeuN pilot."""
import numpy as np
import pytest
import torch

from preprocess.particle_approximation import optimize_particles, varifold_squared
from preprocess.particle_utils import varifold_score_mm


def test_reference_formula_and_existing_score():
    rng = np.random.default_rng(4)
    x, z = rng.normal(size=(6, 3)), rng.normal(size=(3, 3))
    nx, nz = rng.uniform(size=(6, 2)), rng.uniform(size=(3, 2))
    sigma = 1.3

    def inner(a, u, b, v):
        return sum(np.exp(-np.sum((a[i]-b[j])**2)/sigma**2) * np.dot(u[i], v[j])
                   for i in range(len(a)) for j in range(len(b)))

    c = inner(x, nx, x, nx)
    expected = c + inner(z, nz, z, nz) - 2*inner(z, nz, x, nx)
    args = [torch.tensor(a, dtype=torch.float64) for a in (x, nx, z, nz)]
    assert float(varifold_squared(*args, sigma)) == pytest.approx(expected, abs=1e-12)
    assert expected / c == pytest.approx(varifold_score_mm(z, nz, x, nx, [sigma/np.sqrt(2)]))
    assert float(varifold_squared(args[0], args[1], args[0], args[1], sigma)) == pytest.approx(0, abs=1e-12)


def test_position_and_squared_mass_gradients():
    x = torch.tensor([[0., 0., 0.], [0.8, 0.2, 0.]], dtype=torch.float64)
    nx = torch.tensor([[1., 0.4], [0.2, 0.9]], dtype=torch.float64)
    xy = torch.tensor([[0.3, -0.1]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[0.9, 0.8]], dtype=torch.float64, requires_grad=True)

    def loss(xy, a):
        z = torch.cat([xy, torch.zeros((len(xy), 1), dtype=xy.dtype)], dim=1)
        return varifold_squared(x, nx, z, a.square(), 1.)

    assert torch.autograd.gradcheck(loss, (xy, a), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_optimization_recovers_single_particle_and_joint_beats_fixed():
    x, nx = np.array([[1., 2., 0.]]), np.array([[2., 1.]])
    z, nz = np.array([[1.3, 1.8, 0.]]), np.array([[0.5, 0.4]])
    original = [a.copy() for a in (x, nx, z, nz)]
    fixed = optimize_particles(x, nx, z, nz, sigma_mm=1., outer_steps=5)
    joint = optimize_particles(x, nx, z, nz, sigma_mm=1., optimize_positions=True, outer_steps=5)
    np.testing.assert_array_equal(fixed.X, z)
    # Analytic optimum at a fixed offset: mass = kernel(z,x) * source mass.
    np.testing.assert_allclose(fixed.nu, np.exp(-0.13)*nx, atol=1e-5)
    np.testing.assert_allclose(joint.X, x, atol=1e-5)
    np.testing.assert_allclose(joint.nu, nx, atol=1e-5)
    assert joint.objective_history[-1] < 1e-9 < fixed.objective_history[-1]
    for result in (fixed, joint):
        assert np.all(np.diff(result.objective_history) <= 1e-9)
        assert result.nu.shape == nz.shape and np.all(result.nu >= 0)
        assert np.all(result.X[:, 2] == 0)
        assert np.isfinite(result.closure_history).all()
    for a, before in zip((x, nx, z, nz), original):
        np.testing.assert_array_equal(a, before)


def test_reference_zero_mass_lock_and_no_mass_penalty():
    x = np.array([[0., 0., 0.]])
    result = optimize_particles(x, [[2., 1.]], x, [[0.5, 0.]], sigma_mm=1., outer_steps=5)
    np.testing.assert_allclose(result.nu, [[2., 0.]], atol=1e-5)
    assert result.objective_history[-1] == pytest.approx(1., abs=1e-9)


@pytest.mark.parametrize('kind', ['sigma', 'nonfinite', 'negative', 'plane', 'features', 'bound'])
def test_invalid_inputs_and_dense_bound(kind):
    x = np.zeros((2, 3)); nx = np.ones((2, 2)); z = x.copy(); nz = nx.copy()
    kwargs = {'sigma_mm': 1., 'outer_steps': 0}
    if kind == 'sigma': kwargs['sigma_mm'] = float('nan')
    if kind == 'nonfinite': x[0, 0] = np.inf
    if kind == 'negative': nz[0, 0] = -1
    if kind == 'plane': z[0, 2] = 1e-10
    if kind == 'features': nz = nz[:, :1]
    if kind == 'bound': kwargs['max_pairs'] = 3
    with pytest.raises(ValueError):
        optimize_particles(x, nx, z, nz, **kwargs)
