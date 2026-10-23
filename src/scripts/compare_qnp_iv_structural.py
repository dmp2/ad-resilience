#!/usr/bin/env python3
"""Bounded two-condition within-QNP comparison; no compression, exports or figures.

Uses the production rigid fitting sequence and native 32 um QC maps. Optimized
archives store the retained nu_Sub matrix under `nu`; this key is explicit and
validated against the archive's retained_features and ordered feature_labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from config_setup.project_config import load_config, resolve_path
from preprocess.particle_utils import load_particle_archive
from preprocess.section_alignment import FootprintScorer, Similarity2D
from scripts.align_qnp_multistain import MAP_CENTER, dice, fit_registration_stages, homogeneous, resample_map

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'results/analysis/qnp_iv_structural_registration'
IV = PROJECT / 'results/visualization_prototypes/qnp/iv_structural/numerical'
LABELS = ['gray_matter', 'white_matter']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def particle_source(native, stem, stain, stage, condition, iv_dir=IV):
    """Only the coarse stage changes; fine paths/keys are identical in both arms."""
    if stage not in ('coarse', 'fine') or condition not in ('baseline', 'iv_structural'):
        raise ValueError((stage, condition))
    if stage == 'coarse' and condition == 'iv_structural':
        return iv_dir / f'{stain}_1000um.npz', 'nu'
    tag = '1000um' if stage == 'coarse' else '500um'
    return native / f'{stem}_tissue_classes_{tag}.npz', 'nu_Sub'


def read(source):
    x, nu, _ = load_particle_archive(*source)
    return x[:, :2].astype(np.float64), nu.astype(np.float64)


def archive_record(path, key):
    x, nu, meta = load_particle_archive(path, key)
    assert x.ndim == 2 and x.shape[1] == 3 and nu.shape == (len(x), 2)
    assert np.isfinite(x).all() and np.isfinite(nu).all() and (nu >= 0).all()
    assert (x[:, 2] == 0).all()
    return dict(path=str(path.resolve()), sha256=digest(path), feature_key=key,
                count=len(x), xy_bounds_mm=[x[:, :2].min(0).tolist(), x[:, :2].max(0).tolist()],
                per_feature_mass_mm2=nu.sum(0).tolist(), finite=True, nonnegative=True,
                planar_z_zero=True, metadata=meta)


def preflight(config, job, stems, out):
    native = resolve_path(config, job['particle_dir'])
    records = {}
    for stain, stem in stems.items():
        records[stain] = {}
        for stage, condition in [('fine', 'baseline'), ('coarse', 'baseline'), ('coarse', 'iv_structural')]:
            source = particle_source(native, stem, stain, stage, condition)
            r = archive_record(*source)
            records[stain][f'{condition}_{stage}'] = r
            if condition == 'baseline':
                assert r['metadata']['coordinate_space'] == 'qnp_slide_mm'
                assert 'mm^2' in r['metadata']['mass_units']
                with np.load(source[0]) as a:
                    assert a['coarse_feature_names'].tolist() == LABELS
            else:
                m = r['metadata']
                fine = particle_source(native, stem, stain, 'fine', 'baseline')[0]
                coarse = particle_source(native, stem, stain, 'coarse', 'baseline')[0]
                assert m['frame'] == f'qnp_slide_mm:{stem}'
                assert m['feature_labels'] == LABELS and m['retained_features'] == 'nu_Sub'
                assert m['mass_units'] == 'mm^2' and m['variant'] == 'iv_structural'
                assert Path(m['reference']).resolve() == fine.resolve()
                assert Path(m['initialization']).resolve() == coarse.resolve()
                assert m['input_sha256'] == [digest(fine), digest(coarse)]
                assert r['count'] == records[stain]['baseline_coarse']['count']
                # Independently check saved compression reference and initialization.
                refpath = IV / f'{stain}_1000um_reference.npz'
                with np.load(refpath) as a, np.load(fine) as f, np.load(coarse) as c:
                    for saved, original, key in [('X', f, 'X'), ('nu', f, 'nu_Sub'),
                                                  ('Z0', c, 'X'), ('nu_Z0', c, 'nu_Sub')]:
                        np.testing.assert_array_equal(a[saved], original[key])
                assert json.loads(source[0].with_suffix('.json').read_text()) == m
                r['compression_reference_archive'] = {'path': str(refpath), 'sha256': digest(refpath)}
                r['mass_delta_percent'] = 100 * (sum(r['per_feature_mass_mm2']) /
                    sum(records[stain]['baseline_coarse']['per_feature_mass_mm2']) - 1)
        maps = native / f'{stem}_maps_32um.npz'
        records[stain]['qc_maps'] = {'path': str(maps), 'sha256': digest(maps)}
        with np.load(maps) as a:
            assert np.isfinite(a['p_wm']).all()
            records[stain]['qc_maps'].update(shape=list(a['mask'].shape), px_mm=a['px_mm'].tolist(),
                                             has_halo='layers' in a.files)
    baseline = resolve_path(config, job['alignment']['output_dir']) / 'transforms.json'
    graph = PROJECT / 'results/registrations/qnp_H20.33.034_MTG_atlas_plane_scan/transform_graph.json'
    protected = [baseline, graph, PROJECT/'configs/project_config.yaml']
    protected += list((baseline.parent).glob('*'))
    for root in ('qnp', 'omics'):
        protected += list((PROJECT/'results/visualization_prototypes'/root).rglob('*.png'))
        protected += list((PROJECT/'results/visualization_prototypes'/root).rglob('*.pdf'))
        protected += list((PROJECT/'results/visualization_prototypes'/root).rglob('*.state.json'))
    protected += [PROJECT/'src/visualization'/f for f in ('figures_qnp.py', 'figures_omics.py', 'ng_export.py')]
    report = dict(archives=records, settings=job['alignment'], reference='NEUN',
                  structural_stains=[s for s,c in job['stains'].items() if c.get('structural')],
                  annotation_exclusions=job['annotation_misfit'],
                  baseline_transform_path=str(baseline), baseline_transform_sha256=digest(baseline),
                  protected_sha256={str(p): digest(p) for p in protected if p.is_file()},
                  resource_preflight={'available_memory': next(l.strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')),
                      'largest_fine_count': max(r['baseline_fine']['count'] for r in records.values()),
                      'largest_float32_pair_matrix_mib': max(r['baseline_fine']['count'] for r in records.values())**2*4/2**20,
                      'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                      'scratch': os.environ.get('TMPDIR'), 'sequential_fits': True},
                  baseline_rerun_reason='Winning coarse transform absent from saved baseline; matched rerun records all coarse candidates. Original final outputs retained for numerical validation.',
                  mass_handling='Load unchanged masses as float64; no normalization/rescaling. All fitted rigid scales are 1; free scale is diagnostic only.')
    dump(out/'preflight.json', report)
    print('Preflight passed: six native-slide structural archives; identical deterministic fine inputs.', flush=True)
    return report


def delta(a, b, points):
    """b minus a; displacement evaluates the two transforms on unchanged slide points."""
    shift = b.moving_to_section(points) - a.moving_to_section(points)
    return dict(rotation_deg=float((b.theta_deg-a.theta_deg+180)%360-180),
                tau_xy_mm=(np.array(b.tau)-a.tau).tolist(),
                tau_norm_mm=float(np.linalg.norm(np.array(b.tau)-a.tau)),
                reflection_changed=b.flip != a.flip, scale_delta=b.scale-a.scale,
                point_displacement_median_mm=float(np.median(np.linalg.norm(shift,axis=1))),
                point_displacement_max_mm=float(np.max(np.linalg.norm(shift,axis=1))))


def run(config, job, stems, inventory, out, pf):
    native = resolve_path(config, job['particle_dir'])
    settings = job['alignment']; reference = job['reference_stain']
    original = json.loads(Path(pf['baseline_transform_path']).read_text())
    graph = json.loads((PROJECT/'results/registrations/qnp_H20.33.034_MTG_atlas_plane_scan/transform_graph.json').read_text())
    status = {e['from'].split(':')[-1]:e['status'] for e in graph['edges'] if e['to']=='qnp_local_mm'}
    maps = {s:dict(np.load(native/f'{stem}_maps_32um.npz')) for s,stem in stems.items()}
    px = maps[reference]['px_mm']; pad = int(settings.get('grid_pad_px',60))
    shape = tuple(n+2*pad for n in maps[reference]['mask'].shape)
    rr,cc = np.indices(shape)
    origin = np.array([(MAP_CENTER-pad)*px[1],(MAP_CENTER-pad)*px[0]])
    grid = origin + np.column_stack([cc.ravel()*px[1],rr.ravel()*px[0]])
    def sample(stain, t):
        m = maps[stain]
        return {k:resample_map(m,t,grid,k,1 if k=='p_wm' else 0).reshape(shape)
                for k in ('mask','p_wm','layers') if k in m}
    ref = sample(reference,None)
    refmask = ref['mask'] > .5
    refwm = refmask & (ref['p_wm'] > .5)
    annotated = set(inventory['readiness']['slides_with_annotations_on_own_tissue']) - set(job['annotation_misfit'])
    fine_ref = read(particle_source(native,stems[reference],reference,'fine','baseline'))
    common = FootprintScorer(*fine_ref,settings['score_sigmas_mm'])
    result = {'conditions':{}, 'between_conditions':{}, 'baseline_reproduction':{},
              'common_objective':'FootprintScorer on unchanged original deterministic 500um NeuN AND moving-stain nu_Sub at all stages; sigmas [2,1,0.5] mm',
              'qc_grid':{'origin_mm':origin.tolist(),'px_mm':px.tolist(),'shape':list(shape)},
              'reliability':'Original graph status retained; GFAP stays unreliable. HALO exclusions unchanged.'}
    for condition in ('baseline','iv_structural'):
        refc = read(particle_source(native,stems[reference],reference,'coarse',condition))
        coarse_scorer = FootprintScorer(*refc,settings['score_sigmas_mm'])
        records = result['conditions'][condition] = {}
        for stain,stem in stems.items():
            start=time.perf_counter()
            coarse_source = particle_source(native,stem,stain,'coarse',condition)
            fine_source = particle_source(native,stem,stain,'fine',condition)
            P,mu = read(coarse_source); Pf,muf = read(fine_source)
            if stain==reference:
                best=Similarity2D(0.,1.,False,(0.,0.)); ranked=[(0.,best)]
                score=free_score=0.; free=best; runner=None
            else:
                ranked,score,best,runner,free_score,free = fit_registration_stages(coarse_scorer,common,P,mu,Pf,muf,settings)
            entry = dict(coarse_candidates=[{'score':v,'transform':t.as_dict(),
                       'common_objective':common.score_transform(t,Pf,muf)} for v,t in ranked],
                       fine_initial_transform=ranked[0][1].as_dict(),
                       inputs={stage:{'path':str(source[0]),'feature_key':source[1], 'sha256':digest(source[0])}
                               for stage,source in [('coarse',coarse_source),('fine',fine_source)]},
                       reference_inputs={stage:{'path':str(particle_source(native,stems[reference],reference,stage,condition)[0]),
                                                'feature_key':particle_source(native,stems[reference],reference,stage,condition)[1]}
                                         for stage in ('coarse','fine')},
                       runner_up_gap_coarse=None if runner is None else runner[0]-ranked[0][0],
                       runner_up_distinct_coarse=None if runner is None else {'score':runner[0],'transform':runner[1].as_dict()},
                       free_scale_diagnostic={'scale':free.scale,'score':free_score},
                       reliability_status='reference' if stain==reference else status[stem],
                       structural=stain in pf['structural_stains'], stages={})
            for stage,t,loss in [('coarse',ranked[0][1],ranked[0][0]),('fine',best,score)]:
                sampled=sample(stain,t); mask=sampled['mask']>.5
                qc={'tissue_dice_vs_reference':dice(mask,refmask),
                    'white_matter_dice_vs_reference':dice(mask & (sampled['p_wm']>.5),refwm)}
                if stain in annotated and reference in annotated and 'layers' in sampled:
                    qc['halo_ribbon_dice_vs_reference']=dice(sampled['layers']>0,ref['layers']>0)
                matrix=np.array(homogeneous(t,'to_local'))
                assert np.allclose(matrix@np.array(homogeneous(t,'to_slide')),np.eye(3),rtol=0,atol=1e-9)
                entry['stages'][stage]={'transform':t.as_dict(),'slide_mm_to_local_mm':matrix.tolist(),
                                       'fit_objective':loss,'common_objective':common.score_transform(t,Pf,muf),'qc':qc}
            entry['coarse_to_fine']=delta(ranked[0][1],best,Pf)
            entry['seconds']=time.perf_counter()-start
            records[stain]=entry
            if condition=='baseline' and stain!=reference:
                old=original['edges'][stain]
                result['baseline_reproduction'][stain]={'transform_delta':delta(Similarity2D.from_dict(old['transform']),best,Pf),
                    'fine_score_delta':score-old['score_fine'],'coarse_score_delta':ranked[0][0]-old['score_coarse_best'],
                    'qc_delta':{k:v-old['independent_checks'][k] for k,v in entry['stages']['fine']['qc'].items()}}
            dump(out/condition/f'{stain}.json',entry)
            dump(out/'comparison.json',result)
            print(f'{condition} {stain}: coarse {ranked[0][0]:.7f}; fine {score:.7f}; {entry["seconds"]:.1f}s',flush=True)
    for stain,stem in stems.items():
        Pf,_=read(particle_source(native,stem,stain,'fine','baseline'))
        result['between_conditions'][stain]={stage:delta(
            Similarity2D.from_dict(result['conditions']['baseline'][stain]['stages'][stage]['transform']),
            Similarity2D.from_dict(result['conditions']['iv_structural'][stain]['stages'][stage]['transform']),Pf)
            for stage in ('coarse','fine')}
    protected={p:digest(p)==h for p,h in pf['protected_sha256'].items()}
    inputs={r['path']:digest(r['path'])==r['sha256'] for s in pf['archives'].values() for r in s.values() if 'path' in r}
    assert all(protected.values()) and all(inputs.values())
    result['preservation_validation']={'protected_files':protected,'input_archives_and_maps':inputs}
    dump(out/'comparison.json',result)
    return result


def validate_halo(config, job, inventory, out):
    """Existing polygon-union QC on original HALO XML, without rendering figures."""
    from visualization import scene as S
    from visualization.figures_qnp import ribbon_mask

    result = json.loads((out/'comparison.json').read_text())
    grid = result['qc_grid']
    origin, px, shape = np.array(grid['origin_mm']), np.array(grid['px_mm']), tuple(grid['shape'])
    slides = {Path(s['file']).stem.split('-')[-1].upper():s for s in inventory['slides']}
    polys = {}
    for stain in ('NEUN', 'ASYN', 'I6'):
        slide = slides[stain]
        polys[stain] = S.load_halo_polygons(slide['annotation_file'], slide['mpp_baseline_um'])
    ref = ribbon_mask(polys['NEUN'], None, origin, px, shape)
    report = {'method':'Existing figures_qnp.ribbon_mask; original native HALO XML polygons, holes removed, transformed then rasterized on unchanged 32um NeuN grid. No figure rendered.',
              'annotations':{s:{'path':slides[s]['annotation_file'], 'sha256':digest(slides[s]['annotation_file'])} for s in polys},
              'conditions':{}, 'untransformed':{s:dice(ribbon_mask(polys[s],None,origin,px,shape),ref) for s in ('ASYN','I6')}}
    for condition,records in result['conditions'].items():
        report['conditions'][condition]={}
        for stain in ('ASYN','I6'):
            report['conditions'][condition][stain]={}
            for stage,entry in records[stain]['stages'].items():
                t=Similarity2D.from_dict(entry['transform'])
                affine=S.Affine2D.from_similarity(t,'qnp_local_mm','slide','rigid','QC only').inverse()
                value=dice(ribbon_mask(polys[stain],affine,origin,px,shape),ref)
                report['conditions'][condition][stain][stage]={'halo_polygon_ribbon_dice':value,
                    'difference_from_native_raster_qc':value-entry['qc']['halo_ribbon_dice_vs_reference']}
    dump(out/'independent_halo.json',report)
    print(json.dumps(report['conditions'],indent=2))
    # Read serialized transforms back and check the common endpoint through the
    # existing float64 particle scorer, independent of the fitting scorer's float32 inners.
    from preprocess.particle_utils import varifold_score_mm
    pf=json.loads((out/'preflight.json').read_text())
    ref=pf['archives']['NEUN']['baseline_fine']
    X,nu=read((Path(ref['path']),ref['feature_key']))
    scorer=FootprintScorer(X,nu,job['alignment']['score_sigmas_mm'])
    checks=[]
    for condition,records in result['conditions'].items():
        for stain,record in records.items():
            source=record['inputs']['fine']
            P,mu=read((Path(source['path']),source['feature_key']))
            assert record['fine_initial_transform']==record['coarse_candidates'][0]['transform']
            for stage,entry in record['stages'].items():
                t=Similarity2D.from_dict(entry['transform'])
                assert t.scale==1.0
                matrix=np.array(entry['slide_mm_to_local_mm'])
                independent=t.moving_to_section(P)
                np.testing.assert_allclose(independent,P@matrix[:2,:2].T+matrix[:2,2],atol=1e-12)
                y,m,_=scorer.restricted(independent,mu)
                value=varifold_score_mm(y,m,X,nu,job['alignment']['score_sigmas_mm'])
                error=abs(value-entry['common_objective'])
                assert error<1e-5,(condition,stain,stage,error)
                checks.append({'condition':condition,'stain':stain,'stage':stage,
                               'float64_common_objective':value,'absolute_difference':error})
    protected={p:digest(p)==h for p,h in pf['protected_sha256'].items()}
    assert all(protected.values())
    dump(out/'numerical_validation.json',{'checks':checks,'all_passed':True,
        'float32_vs_float64_tolerance':1e-5,'max_objective_absolute_difference':max(c['absolute_difference'] for c in checks),
        'transform_direction_and_initialization_passed':True,'protected_file_count':len(protected),
        'protected_files_unchanged':all(protected.values()),
        'source_sha256':{str(PROJECT/p):digest(PROJECT/p) for p in (
            'src/scripts/align_qnp_multistain.py','src/scripts/compare_qnp_iv_structural.py',
            'src/tests/test_qnp_iv_structural_comparison.py')},
        'disk_free_gib':os.statvfs(out).f_bavail*os.statvfs(out).f_frsize/2**30})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','run','validate'],required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR'):
        path=Path(os.environ[name]).resolve()
        assert path.is_relative_to(Path.home().resolve()),(name,path)
    config=load_config(PROJECT/'configs/project_config.yaml')
    job=config['qnp_section_inputs']['H20.33.034_MTG']
    inv=json.loads((resolve_path(config,job['qc_dir'])/'inventory.json').read_text())
    stems={Path(s['file']).stem.split('-')[-1].upper():Path(s['file']).stem for s in inv['slides']}
    if args.stage=='preflight':
        preflight(config,job,stems,OUT)
        return
    if args.stage=='validate':
        validate_halo(config,job,inv,OUT)
        return
    pf=json.loads((OUT/'preflight.json').read_text())
    for s in pf['archives'].values():
        for r in s.values():
            assert digest(r['path'])==r['sha256'],r['path']
    start=time.perf_counter()
    run(config,job,stems,inv,OUT,pf)
    dump(OUT/'resources.json',{'wall_seconds':time.perf_counter()-start,
                              'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})


if __name__=='__main__':
    main()
