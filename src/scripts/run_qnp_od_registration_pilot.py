#!/usr/bin/env python3
"""Bounded OD-only ASYN/I6 -> NeuN pilot using unchanged production fitting.

Run with PYTHONPATH=src:src/config_setup:src/preprocess. Writes only to the
pilot analysis directory. Native particles, saved baselines and figures are read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from config_setup.project_config import load_config, resolve_path
from preprocess.section_alignment import FootprintScorer, Similarity2D
from scripts.align_qnp_multistain import (
    MAP_CENTER, dice, fit_registration_stages, homogeneous,
    registration_particles, resample_map,
)
from visualization import scene as S
from visualization.figures_qnp import ribbon_mask

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'results/analysis/qnp_od_registration_pilot'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def delta(a, b, points):
    distances = np.linalg.norm(b.moving_to_section(points) - a.moving_to_section(points), axis=1)
    return dict(rotation_deg=float((b.theta_deg-a.theta_deg+180)%360-180),
                reflection_changed=b.flip != a.flip,
                tau_xy_mm=(np.asarray(b.tau)-a.tau).tolist(),
                point_displacement_median_mm=float(np.median(distances)),
                point_displacement_max_mm=float(distances.max()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT / 'configs/project_config.yaml')
    job = config['qnp_section_inputs']['H20.33.034_MTG']
    settings = job['alignment']
    native = resolve_path(config, job['particle_dir'])
    baseline_path = resolve_path(config, settings['output_dir']) / 'transforms.json'
    baseline = json.loads(baseline_path.read_text())
    inventory_path = resolve_path(config, job['qc_dir']) / 'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    slides = {Path(s['file']).stem.split('-')[-1].upper(): s for s in inventory['slides']}
    stains = ('NEUN', 'ASYN', 'I6')
    stems = {s: Path(slides[s]['file']).stem for s in stains}
    annotated = set(inventory['readiness']['slides_with_annotations_on_own_tissue']) - set(job['annotation_misfit'])
    assert set(stains) <= annotated
    scratch = Path(os.environ['TMPDIR']).resolve()
    assert scratch.is_relative_to(Path.home()) and scratch.is_dir()

    # Hash every existing native/aligned particle product, plus existing figure families,
    # numerical prototype products, and prior comparison products. Never write to them.
    protected = set(native.rglob('*'))
    for directory in ('results/visualization_prototypes', 'results/analysis/qnp_iv_structural_registration'):
        protected.update((PROJECT / directory).rglob('*'))
    protected.update([inventory_path, PROJECT / 'configs/project_config.yaml'])
    protected.update((PROJECT / 'src/visualization').glob('*.py'))
    protected.update(Path(slides[s]['annotation_file']) for s in stains)
    protected_hash = {str(p): digest(p) for p in sorted(protected) if p.is_file()}
    inputs = {}
    particles = {}
    edges = None
    for stain in stains:
        inputs[stain], particles[stain] = {}, {}
        for stage, tag in [('coarse', settings['search_bin_tag']), ('fine', settings['refine_bin_tag'])]:
            source = native / f'{stems[stain]}_od_bins_{tag}.npz'
            X, nu, meta = registration_particles(native, stems[stain], tag, 'od_bins')
            assert nu.shape == (len(X), 8) and np.isfinite(nu).all() and (nu >= 0).all()
            assert np.isfinite(X).all() and meta['coordinate_space'] == 'qnp_slide_mm'
            with np.load(source, allow_pickle=False) as a:
                assert (a['X'][:, 2] == 0).all()
                if edges is None:
                    edges = a['bin_edges'].copy()
                np.testing.assert_array_equal(a['bin_edges'], edges)
            particles[stain][stage] = (X, nu)
            inputs[stain][stage] = dict(path=str(source), sha256=digest(source), feature_key='nu_bin',
                shape=list(nu.shape), per_channel_mass_mm2=nu.sum(0).tolist(), metadata=meta)
    preflight = dict(job='H20.33.034_MTG', reference='NEUN', inputs=inputs,
        settings=settings, bin_edges=edges.tolist(), baseline_path=str(baseline_path),
        scratch=str(scratch), protected_sha256=protected_hash,
        objective='Unchanged FootprintScorer, identity feature dot product on all eight nu_bin channels. '
                  'No feature normalization beyond the stored OD representation. No p_wm or HALO in fitting.',
        archive_metadata_note='Native OD metadata contains inherited gray/white diagnostics; these are not read by the objective.',
        threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    dump('preflight.json', preflight)
    print('Preflight passed: six unchanged eight-channel OD archives.', flush=True)

    coarse = FootprintScorer(*particles['NEUN']['coarse'], settings['score_sigmas_mm'])
    fine = FootprintScorer(*particles['NEUN']['fine'], settings['score_sigmas_mm'])
    # QC reads tissue masks only, never inferred gray/white memberships.
    maps = {}
    polys = {}
    for s in stains:
        with np.load(native / f'{stems[s]}_maps_32um.npz') as a:
            maps[s] = {k: a[k] for k in ('mask', 'px_mm')}
        polys[s] = S.load_halo_polygons(slides[s]['annotation_file'], slides[s]['mpp_baseline_um'])
    px = maps['NEUN']['px_mm']
    pad = int(settings.get('grid_pad_px', 60))
    shape = tuple(n+2*pad for n in maps['NEUN']['mask'].shape)
    rr, cc = np.indices(shape)
    origin = np.array([(MAP_CENTER-pad)*px[1], (MAP_CENTER-pad)*px[0]])
    grid = origin + np.column_stack([cc.ravel()*px[1], rr.ravel()*px[0]])
    refmask = resample_map(maps['NEUN'], None, grid, 'mask', 0).reshape(shape) > .5
    refhalo = ribbon_mask(polys['NEUN'], None, origin, px, shape)

    def qc(stain, transform):
        affine = S.Affine2D(np.asarray(homogeneous(transform, 'to_local')), 'slide', 'qnp_local_mm')
        tissue = resample_map(maps[stain], transform, grid, 'mask', 0).reshape(shape) > .5
        halo = ribbon_mask(polys[stain], affine, origin, px, shape)
        return dict(tissue_dice=dice(tissue, refmask), halo_polygon_ribbon_dice=dice(halo, refhalo),
                    reference_tissue_coverage=float((tissue & refmask).sum()/refmask.sum()))

    result = dict(qc_grid=dict(origin_mm=origin.tolist(), px_mm=px.tolist(), shape=list(shape)),
        halo_method='Original HALO XML polygon union with holes removed; transformed then rasterized '
                    'using existing ribbon_mask on the padded 32um reference grid.', stains={})
    for stain in ('ASYN', 'I6'):
        start = time.perf_counter()
        P, mu = particles[stain]['coarse']
        Pf, muf = particles[stain]['fine']
        ranked, score, best, runner, free_score, free = fit_registration_stages(
            coarse, fine, P, mu, Pf, muf, settings)
        saved = Similarity2D.from_dict(baseline['edges'][stain]['transform'])
        identity = Similarity2D(0., 1., False, (0., 0.))
        def record(t):
            return dict(transform=t.as_dict(), slide_mm_to_local_mm=homogeneous(t, 'to_local'),
                score_od_coarse=coarse.score_transform(t, P, mu),
                score_od_fine=fine.score_transform(t, Pf, muf), qc=qc(stain, t),
                delta_vs_saved_baseline=delta(saved, t, Pf))
        entry = dict(stages={name: record(t) for name,t in
                     [('untransformed', identity), ('saved_baseline', saved),
                      ('coarse', ranked[0][1]), ('refined', best)]},
                     coarse_candidates=[dict(rank=i+1, coarse_score=v, **record(t)) for i,(v,t) in enumerate(ranked)],
                     runner_up_gap_coarse=None if runner is None else runner[0]-ranked[0][0],
                     runner_up_distinct_coarse=None if runner is None else runner[1].as_dict(),
                     coarse_to_refined=delta(ranked[0][1], best, Pf),
                     free_scale_diagnostic=dict(score=free_score, transform=free.as_dict(), applied=False),
                     runtime_seconds=time.perf_counter()-start)
        result['stains'][stain] = entry
        dump('comparison.json', result)
        print(stain, json.dumps(entry), flush=True)
    unchanged = {p: digest(p) == h for p,h in protected_hash.items()}
    assert all(unchanged.values()), [p for p,ok in unchanged.items() if not ok]
    dump('preservation.json', dict(all_unchanged=True, checked_files=len(unchanged), sha256_verified=unchanged))
    print('Protected products unchanged:', len(unchanged), flush=True)


if __name__ == '__main__':
    main()
