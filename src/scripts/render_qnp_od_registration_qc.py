#!/usr/bin/env python3
"""Render saved QNP OD fits only; never invoke registration or publication figures."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scipy import ndimage as ndi
from matplotlib.lines import Line2D

from preprocess.qnp_particles import read_slide
from preprocess.section_alignment import Similarity2D
from scripts.align_qnp_multistain import dice, resample_map
from visualization import scene as S, style as st, ng_export as nx
from visualization.figures_qnp import ribbon_mask
from visualization.ng_scenes import _closed

PROJECT = S.PROJECT
PILOT = PROJECT / 'results/analysis/qnp_od_registration_pilot'
OUT = PILOT / 'visual_qc'
BLUE, ORANGE = st.CAT3[:2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')


def affine(entry, source):
    """Read saved slide -> reference matrix; cross-check its inverse convention."""
    t = Similarity2D.from_dict(entry['transform'])
    a = S.Affine2D(np.asarray(entry['slide_mm_to_local_mm']), source, 'qnp_local_mm',
                   'saved rigid + optional reflection', 'diagnostic', 'saved fit; no fitting here')
    independent = S.Affine2D.from_similarity(t, 'qnp_local_mm', source, '', '').inverse()
    np.testing.assert_allclose(a.matrix, independent.matrix, atol=1e-12)
    return a


def warp_rgb(slide, to_local, origin, px, shape):
    """Pull back native pyramid RGB via the existing physical Affine2D convention.

    No shared RGB warper exists in the figure suite (it displays native block means).
    This bounded map_coordinates adapter keeps native pixel centre = index * px_mm.
    """
    rr, cc = np.indices(shape)
    local = origin + np.column_stack([cc.ravel()*px[1], rr.ravel()*px[0]])
    native = to_local.inverse().apply(local)
    coords = np.vstack([native[:, 1]/slide.px_mm[0], native[:, 0]/slide.px_mm[1]])
    return np.stack([ndi.map_coordinates(slide.rgb[..., c], coords, order=1,
        mode='constant', cval=255, prefilter=False).reshape(shape) for c in range(3)], axis=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['TMPDIR']).resolve()
    assert scratch.is_relative_to(Path.home()) and scratch.is_dir()
    config = S.load_project_config(PROJECT/'configs/project_config.yaml')
    job = S.qnp_job(config)
    comparison = json.loads((PILOT/'comparison.json').read_text())
    baseline_path = job['_aligned_dir']/'transforms.json'
    baseline = json.loads(baseline_path.read_text())
    # Protect existing registration products, figures, states, source and pilot results.
    protected = set(job['_particle_dir'].rglob('*'))
    for folder in ('results/visualization_prototypes', 'results/analysis/figures', 'results/figures',
                   'src/visualization'):
        protected.update(p for p in (PROJECT/folder).rglob('*') if p.suffix in ('.py','.png','.pdf','.svg','.json'))
    protected.update(p for p in PILOT.iterdir() if p.is_file())
    protected.update((PROJECT/'results/registrations').rglob('*.json'))
    protected = {str(p):sha(p) for p in sorted(protected) if p.is_file()}
    dump('protected_before.json', protected)
    stains = ('NEUN','ASYN','I6')
    slides, maps, polys, provenance = {}, {}, {}, {}
    for s in stains:
        path = job['_slides'][s]
        # Only smallest native Aperio level (~8 um), not a newly generated derivative.
        slides[s] = read_slide(path, level=-1)
        with np.load(job['_particle_dir']/f'{path.stem}_maps_32um.npz') as a:
            maps[s] = {k:a[k] for k in ('mask','px_mm')}
        polys[s] = S.load_halo_polygons(path.with_suffix('.annotations'), slides[s].mpp_baseline_um)
        provenance[s] = dict(slide=slides[s].describe(), svs_bytes=path.stat().st_size,
            svs_mtime_ns=path.stat().st_mtime_ns, selected_native_rgb_sha256=hashlib.sha256(slides[s].rgb.tobytes()).hexdigest(),
            annotations=dict(path=str(path.with_suffix('.annotations')),sha256=sha(path.with_suffix('.annotations')),
                             rings=len(polys[s].rings),holes=int(polys[s].holes.sum())),
            tissue_mask=dict(path=str(job['_particle_dir']/f'{path.stem}_maps_32um.npz'),
                             sha256=sha(job['_particle_dir']/f'{path.stem}_maps_32um.npz')))
        print(s, 'native RGB', slides[s].rgb.shape, 'holes',int(polys[s].holes.sum()), flush=True)
    identity = S.Affine2D(np.eye(3), S.slide_frame(job['_slides']['NEUN']), 'qnp_local_mm')
    cases = {}
    for s in ('ASYN','I6'):
        records = comparison['stains'][s]['stages']
        # Baseline is loaded from original production output, then checked against pilot copy.
        b = baseline['edges'][s]
        np.testing.assert_array_equal(b['slide_mm_to_local_mm'], records['saved_baseline']['slide_mm_to_local_mm'])
        cases[s] = [('baseline','Production baseline',records['saved_baseline']),
                    ('coarse','OD coarse',records['coarse']),('refined','OD refined',records['refined'])]
        cases[s].append(('alternative_rank2','Coarse rank 2: alternative',comparison['stains'][s]['coarse_candidates'][1])
                        if s=='I6' else ('untransformed','Native identity: control',records['untransformed']))
    # Same square physical field for all panels and both plates; include misplaced I6 tissue.
    bounds = []
    for s in stains:
        rr,cc=np.nonzero(maps[s]['mask']); p=maps[s]['px_mm']
        corners = np.array([[cc.min(),rr.min()],[cc.max(),rr.min()],[cc.max(),rr.max()],[cc.min(),rr.max()]])
        corners=(corners+.375)*np.array([p[1],p[0]])
        bounds.append(corners)
        if s!='NEUN':
            bounds.extend(affine(e,S.slide_frame(job['_slides'][s])).apply(corners) for _,_,e in cases[s])
    points=np.vstack(bounds); lo=points.min(0)-1.2; hi=points.max(0)+1.2
    width=float(np.ceil(max(hi-lo))); centre=(lo+hi)/2
    extent=[float(centre[0]-width/2),float(centre[0]+width/2),float(centre[1]-width/2),float(centre[1]+width/2)]
    # 24 um display sampling is bounded and finer than a printed overview pixel.
    shape=(int(np.ceil(width/.024)),)*2
    px=np.array([width/shape[0]]*2)
    origin=np.array([extent[0]+px[1]/2,extent[2]+px[0]/2])
    raster=S.Raster('display','qnp_local_mm',np.zeros(shape),tuple(origin),tuple(px[::-1]),'derived','RGB sampling')
    xs=origin[0]+np.arange(shape[1])*px[1];ys=origin[1]+np.arange(shape[0])*px[0]
    rr,cc=np.indices(shape)
    grid=origin+np.column_stack([cc.ravel()*px[1],rr.ravel()*px[0]])
    ref_rgb=warp_rgb(slides['NEUN'],identity,origin,px,shape)
    refmask=resample_map(maps['NEUN'],None,grid,'mask',0).reshape(shape)>.5
    checker=((np.floor(xs[None,:]/1.5)+np.floor(ys[:,None]/1.5))%2==0)
    q=comparison['qc_grid'];qorigin=np.array(q['origin_mm']);qpx=np.array(q['px_mm']);qshape=tuple(q['shape'])
    qr,qc=np.indices(qshape)
    qgrid=qorigin+np.column_stack([qc.ravel()*qpx[1],qr.ravel()*qpx[0]])
    qrefmask=resample_map(maps['NEUN'],None,qgrid,'mask',0).reshape(qshape)>.5
    qrefhalo=ribbon_mask(polys['NEUN'],None,qorigin,qpx,qshape)
    checks=[]; cached={}; outputs=[]

    def tissue(ax,mask,color):
        ax.contour(xs,ys,mask.astype(float),[.5],colors=[color],linewidths=.55,linestyles='--')

    def rings(ax,p,trans,color,lw=.6):
        # Draw every original ring, including holes; never fill holes as tissue.
        for ring,hole in zip(p.rings,p.holes):
            closed=_closed(ring); moved=closed if trans is None else trans.apply(closed)
            ax.plot(*moved.T,color=color,lw=lw,ls=':' if hole else '-',alpha=.9)

    def decorate(ax,box=extent):
        st.map_axes(ax,box);st.scale_bar(ax,5 if box==extent else 1,'5 mm' if box==extent else '1 mm')

    for s in ('ASYN','I6'):
        fig=st.figure(320,330)
        gs=fig.add_gridspec(4,4,left=.04,right=.99,top=.92,bottom=.055,wspace=.08,hspace=.26)
        axes=np.array([[fig.add_subplot(gs[i,j]) for j in range(4)]for i in range(4)])
        fig.suptitle(f'{s} → NeuN | saved registration visual QC',x=.04,y=.985,ha='left',fontsize=14,fontweight='bold')
        fig.text(.04,.955,'Native RGB histology • identical physical field across both plates • no registration rerun',fontsize=8)
        for ax,ss in zip(axes[0,:2],('NEUN',s)):
            slide=slides[ss]
            native=S.Raster(ss,S.slide_frame(slide.path),slide.rgb,(0,0),slide.px_mm[::-1],'measured',str(slide.path))
            ax.imshow(native.array,extent=native.extent,interpolation='antialiased',rasterized=True)
            ax.set_title(f'{ss}: native RGB (~8 µm pixels)',fontsize=8)
            decorate(ax);st.axes_glyph(ax,'+x','+y',True)
        axes[0,2].axis('off');axes[0,3].axis('off')
        handles=[Line2D([],[],color=BLUE,lw=1,label='NeuN HALO layers'),
                 Line2D([],[],color=ORANGE,lw=1,label=f'{s} HALO layers'),
                 Line2D([],[],color=BLUE,lw=1,ls='--',label='NeuN tissue boundary'),
                 Line2D([],[],color=ORANGE,lw=1,ls='--',label=f'{s} tissue boundary'),
                 Line2D([],[],color=st.TEXT_2,lw=1,ls=':',label='HALO hole boundary')]
        axes[0,2].legend(handles=handles,loc='upper left',frameon=False,fontsize=8)
        axes[0,3].text(0,.95,'Reading the plate\n\nTop: native slide appearance\nRow 2: transformed moving RGB\nRow 3: 1.5 mm RGB checkerboard\nRow 4: original HALO geometry\n\nBlue = reference, orange = moving\nAll fitted scales are 1\nx right, y down; axes in mm',va='top',fontsize=8,linespacing=1.6)
        for j,(key,title,entry) in enumerate(cases[s]):
            trans=affine(entry,S.slide_frame(job['_slides'][s])); t=Similarity2D.from_dict(entry['transform'])
            rgb=warp_rgb(slides[s],trans,origin,px,shape)
            mask=resample_map(maps[s],t,grid,'mask',0).reshape(shape)>.5
            cached[(s,key)]=(rgb,mask,trans,entry)
            metrics=entry['qc']
            axes[1,j].imshow(rgb,extent=raster.extent,rasterized=True)
            axes[1,j].set_title(f'{title}\nθ={t.theta_deg%360:.2f}°; reflected={t.flip}',fontsize=7.5)
            mixed=np.where(checker[...,None],ref_rgb,rgb)
            axes[2,j].imshow(mixed,extent=raster.extent,rasterized=True)
            axes[2,j].set_title('NeuN / moving RGB checkerboard',fontsize=7.5)
            axes[3,j].imshow(ref_rgb,extent=raster.extent,alpha=.22,rasterized=True)
            rings(axes[3,j],polys['NEUN'],None,BLUE)
            rings(axes[3,j],polys[s],trans,ORANGE)
            axes[3,j].set_title(f"HALO {metrics['halo_polygon_ribbon_dice']:.3f} | tissue {metrics['tissue_dice']:.3f}\nReference coverage {metrics['reference_tissue_coverage']:.1%}",fontsize=7.5)
            for ax in axes[1:,j]:
                tissue(ax,refmask,BLUE);tissue(ax,mask,ORANGE);decorate(ax)
            # Reproduce numerical QC on its original grid, not the larger display field.
            m=resample_map(maps[s],t,qgrid,'mask',0).reshape(qshape)>.5
            h=ribbon_mask(polys[s],trans,qorigin,qpx,qshape)
            observed=dict(tissue_dice=dice(m,qrefmask),halo_polygon_ribbon_dice=dice(h,qrefhalo),
                          reference_tissue_coverage=float((m&qrefmask).sum()/qrefmask.sum()))
            error=max(abs(observed[k]-metrics[k]) for k in observed);assert error<1e-12
            # Vertex and inverse-RGB sample direction: forward overlay -> inverse pullback.
            vertices=np.vstack(polys[s].rings)
            err=float(np.max(np.abs(trans.apply(vertices)-t.moving_to_section(vertices))))
            roundtrip=float(np.max(np.abs(trans.inverse().apply(trans.apply(vertices))-vertices)))
            assert err<1e-12 and roundtrip<1e-12
            checks.append(dict(stain=s,stage=key,metric_error_max=error,observed=observed,
                polygon_vertex_direction_error_mm=err,inverse_rgb_pullback_error_mm=roundtrip,
                landmark_native_mm=vertices[0].tolist(),landmark_local_mm=trans.apply(vertices[:1])[0].tolist(),
                transform=trans.describe()))
        fig.text(.04,.025,'Solid HALO lines retain layer boundaries; dotted lines mark holes. Dashed lines show tissue footprints.\n'
                 'Metrics use the original saved 32 µm QC grid; the larger display field includes displaced I6 tissue.',fontsize=7,color=st.TEXT_2)
        outputs+=st.save(fig,OUT/f'{s.lower()}_registration_comparison')
        print(s,'comparison plate saved',flush=True)

    # Matched baseline/refined detail windows: visualize small ASYN changes at RGB landmarks.
    fig=st.figure(240,180)
    boxes=[[7,13,2,8],[11,17,13,19]]
    for i,box in enumerate(boxes):
        for j,key in enumerate(('baseline','coarse','refined')):
            ax=fig.add_subplot(2,3,i*3+j+1)
            rgb,mask,trans,entry=cached[('ASYN',key)]
            ax.imshow(np.where(checker[...,None],ref_rgb,rgb),extent=raster.extent,rasterized=True)
            rings(ax,polys['NEUN'],None,BLUE,.7);rings(ax,polys['ASYN'],trans,ORANGE,.7)
            tissue(ax,refmask,BLUE);tissue(ax,mask,ORANGE)
            decorate(ax,box);ax.set_title(f'ASYN {key} | window {i+1}',fontsize=8)
    fig.subplots_adjust(left=.03,right=.98,top=.91,bottom=.06,wspace=.1,hspace=.2)
    fig.suptitle('ASYN ribbon / tissue detail | same 1.5 mm RGB checkerboard',fontsize=11)
    outputs+=st.save(fig,OUT/'asyn_landmark_detail')

    # Existing Neuroglancer source assets can be reused directly: states only, no asset exports.
    ng = PROJECT/'results/visualization_prototypes/qnp/neuroglancer'
    original_state=json.loads((ng/'H20.33.034_MTG_multistain.state.json').read_text())
    scene_files=[]
    for s in ('ASYN','I6'):
        layers=[]
        refim=copy.deepcopy(next(l for l in original_state['layers'] if l['type']=='image' and 'NEUN]' in l['name']))
        refhalo=copy.deepcopy(next(l for l in original_state['layers'] if l['name']=='NeuN HALO layers'))
        refim['visible']=True;refim['opacity']=1.;refhalo['visible']=True
        refhalo['shader']=nx.categorical_shader('layer',[BLUE]*5,size=0,line_width=1.5)
        layers.extend([refim,refhalo])
        for key,title,entry in cases[s]:
            trans=cached[(s,key)][2].in_units(1000.,'um')
            for kind,match in [('image',f'{job["_slides"][s].stem}]'),('annotation', 'a-syn HALO layers' if s=='ASYN' else 'Iba1+6E10 HALO layers')]:
                source=next(l for l in original_state['layers'] if l['type']==kind and match in l['name'])
                url=source['source']['url'];local=nx.SERVE_ROOT/urlparse(url.split('://',1)[1]).path.lstrip('/')
                if kind=='image':
                    layer=nx.image_layer(f'{s} {title} RGB',local,trans,visible=key=='refined',opacity=.5)
                else:
                    layer=nx.annotation_layer(f'{s} {title} HALO',local,nx.categorical_shader('layer',[ORANGE]*5,size=0,line_width=1.5),trans,visible=key=='refined')
                assert local.exists(),local
                layers.append(layer)
        state=nx.build_state(layers,centre*1000,width*1000)
        # write_scene only writes JSON and review URLs; all sources remain in their original locations.
        written=nx.write_scene(OUT/'neuroglancer',f'{s.lower()}_saved_registration_review',state,
            dict(description='Diagnostic saved transforms only; alternatives are not validated alignments.',
                 inputs=str(PILOT/'comparison.json'),assets='Reused existing RGB pyramids and HALO polylines; no assets rewritten.',
                 transforms={key:cached[(s,key)][2].describe() for key,_,_ in cases[s]}))
        scene_files.extend(str(p) for p in written.values())
    changed=[p for p,h in protected.items() if sha(p)!=h]
    assert not changed,changed
    dump('validation.json',dict(all_passed=True,checks=checks,protected_count=len(protected),
        protected_files_unchanged=True,display_extent_mm=extent,display_shape=shape,display_px_mm=px.tolist(),
        no_registration_invoked=True,images_and_annotations_not_regenerated=True))
    dump('provenance.json',dict(inputs=provenance,baseline=dict(path=str(baseline_path),sha256=sha(baseline_path)),
        pilot_comparison=dict(path=str(PILOT/'comparison.json'),sha256=sha(PILOT/'comparison.json')),
        outputs=[str(p) for p in outputs],scenes=scene_files,scratch=str(scratch),
        renderer_sha256=sha(Path(__file__)),detail_windows_mm=boxes,
        primitives=['qnp_particles.read_slide','scene.Affine2D','scene.Raster','scene.load_halo_polygons',
                    'align_qnp_multistain.resample_map','figures_qnp.ribbon_mask','style.figure/map_axes/scale_bar/save',
                    'ng_export.image_layer/annotation_layer/build_state/write_scene']))
    print('Validation passed; protected files unchanged:',len(protected),flush=True)


if __name__=='__main__':
    main()
