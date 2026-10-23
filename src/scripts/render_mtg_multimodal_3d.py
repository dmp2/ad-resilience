"""Render existing MTG geometry only; never register, fit, or overwrite outputs.

Run with the wsi-pipeline environment and project PYTHONPATH, e.g.:
PYTHONPATH=src:src/preprocess:src/config_setup:src/scripts python \
    src/scripts/render_mtg_multimodal_3d.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Keep Python, VTK, matplotlib, and temporary artifacts under HOME.
scratch = Path.home() / '.cache/ad-resilience/mtg-multimodal-3d'
scratch.mkdir(parents=True, exist_ok=True)
for key, subdir in [('TMPDIR', 'tmp'), ('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'cache')]:
    directory = scratch / subdir
    directory.mkdir(exist_ok=True)
    os.environ[key] = str(directory)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pyvista as pv
from visualization import scene as S


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render(scene, mesh, output, view, interactive=False):
    plot = pv.Plotter(off_screen=True, window_size=(1800, 1250))
    plot.set_background('#fafbfc')
    plot.add_mesh(mesh, color='#969da5', opacity=.13, smooth_shading=True,
                  label='Template: left MTG', name='MTG')
    for layer in scene['layers']:
        xyz = layer['xyz'].copy()
        meta = layer['metadata']
        if view == 'exploded' and layer['modality'] == 'QNP':
            xyz += layer['normal'] * ((meta['order']-2.5)*2. - meta['z_offset_mm'])
        # Point planes preserve sampled tissue concavities; no convex hull or filled tissue inference.
        plot.add_points(xyz, color=layer['color'], point_size=3.5 if layer['modality']=='QNP' else 2.,
                        opacity=.55 if layer['modality']=='QNP' else .75,
                        render_points_as_spheres=False, name=layer['name'])
        # Explicit 5um bounding slab wireframe, not a tissue boundary estimate.
        lo, hi = xyz.min(0), xyz.max(0)
        lo[1] -= .0025
        hi[1] += .0025
        box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        plot.add_mesh(box, style='wireframe', color=layer['color'], opacity=.45, line_width=1)
    title = dict(overview='A   Oblique overview', side='B   Left lateral view',
                 top='C   Coronal / anterior view', exploded='D   Exploded QNP (100× spacing)')[view]
    plot.add_text(title, position='upper_left',
                  font_size=18, color='#253346')
    anatomical_axes = pv.AxesAssemblySymmetric(
        x_label=('', '') if view=='side' else ('Right', 'Left'),
        y_label=('', '') if view=='top' else ('Rostral', 'Caudal'),
        z_label=('Dorsal', 'Ventral'), label_size=25, label_color='#253346',
        label_position=1.15)
    plot.add_orientation_widget(anatomical_axes, interactive=False, viewport=(0,0,.25,.30))
    center = np.mean(np.array(mesh.bounds).reshape(3,2), axis=1)
    direction = dict(overview=np.array([-1.,1.6,.8]), side=np.array([-1.,0.,0.]),
                     top=np.array([0.,1.,0.]), exploded=np.array([-1.,1.6,.8]))[view]
    plot.camera_position = [center + direction*100, center, (0,0,1)]
    plot.enable_parallel_projection()
    plot.reset_camera()
    plot.camera.zoom(.82)
    plot.show(auto_close=False)
    pixels = plot.screenshot(return_img=True)
    if interactive:
        plot.export_html(str(output.with_name('mtg_multimodal_3d_interactive.html')))
    plot.close()
    return pixels


def write_montage(images, out, scene):
    """Four in-memory views, one PNG; individual view PNGs are never written."""
    from PIL import Image, ImageDraw, ImageFont

    montage = Image.new('RGB', (3600, 2700), '#fafbfc')
    for i, pixels in enumerate(images):
        montage.paste(Image.fromarray(pixels).convert('RGB'), ((i % 2)*1800, (i // 2)*1250))
    labels = {'NEUN':'NeuN', 'AT':'AT8+pTDP43', 'ASYN':'a-syn', 'I6':'Iba1+6E10'}
    legend = [('Template MTG', '#969da5')] + [
        (f"{layer['modality']}  {labels.get(layer['name'], layer['name'])}", layer['color'])
        for layer in scene['layers']]
    draw = ImageDraw.Draw(montage)
    font = ImageFont.truetype('DejaVuSans.ttf', 32)
    for i, (label, color) in enumerate(legend):
        x, y = 60+(i%5)*710, 2535+(i//5)*75
        draw.rectangle((x,y+5,x+32,y+37), fill=color)
        draw.text((x+50,y), label, font=font, fill='#253346')
    montage.save(out/'mtg_multimodal_3d_montage.png')


def _draw_biology(plot, layer, detail=False):
    from visualization import style as st

    if layer['modality'] == 'QNP':
        quad = pv.PolyData(layer['rgb_quad'], [4,0,1,2,3])
        # numpy_to_texture flips image rows internally; original top row maps to v=1.
        quad.active_texture_coordinates = np.array([[0,1],[1,1],[1,0],[0,0]],float)
        texture = pv.numpy_to_texture(layer['rgba'])
        texture.interpolate = True
        plot.add_mesh(quad, texture=texture, lighting=False, name=layer['name'])
        # Frame visible tissue, rather than the empty corners of its rotated RGB canvas.
        rows,cols=np.nonzero(layer['rgba'][:,:,3]); h,w=layer['rgba'].shape[:2]
        q=layer['rgb_quad']
        tissue=q[0]+((cols+.5)/w)[:,None]*(q[1]-q[0])+((rows+.5)/h)[:,None]*(q[3]-q[0])
        pad=np.abs(q[1]-q[0])/w+np.abs(q[3]-q[0])/h
        return np.vstack([tissue.min(0)-pad,tissue.max(0)+pad])
    for code, color in enumerate(st.CAT3):
        selected = layer['cell_class3']==code
        if np.any(selected):
            plot.add_points(layer['cell_xyz'][selected], color=color, point_size=1.8 if detail else 2.,
                            opacity=.85, render_points_as_spheres=False)
    # Provider boundaries only; do not infer a whole-section tissue polygon.
    for ring in layer['polygon_world']:
        line=pv.lines_from_points(np.vstack([ring,ring[:1]]))
        strokes=[(5.,'#fafbfc',-5),(2.5,st.TEXT_2,-10)] if detail else [(1.,st.TEXT_2,-1)]
        for width,color,offset in strokes:
            actor=plot.add_mesh(line,color=color,line_width=width,lighting=False)
            # A raster depth bias keeps annotations above points without moving vertices.
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0,offset)
    return np.vstack([layer['cell_xyz'], *layer['polygon_world']])


def _fit_biological_camera(plot, points, direction, margin=1.12):
    """Fit unchanged world geometry in camera coordinates with a modest safety margin."""
    normal = np.asarray(direction,float)/np.linalg.norm(direction)
    right = np.cross(-normal, [0,0,1]); right /= np.linalg.norm(right)
    up = np.cross(right,-normal)
    projection = points@np.stack([right,up],axis=1)
    middle = (projection.min(0)+projection.max(0))/2
    center = right*middle[0]+up*middle[1]+normal*np.mean(points@normal)
    plot.camera_position = [center+normal*200,center,up]
    plot.enable_parallel_projection()
    viewport = plot.renderer.GetViewport()
    aspect = (plot.window_size[0]*(viewport[2]-viewport[0]) /
              (plot.window_size[1]*(viewport[3]-viewport[1])))
    span = np.ptp(projection,axis=0)
    plot.camera.parallel_scale = .5*max(span[1],span[0]/aspect)*margin
    plot.reset_camera_clipping_range()
    return dict(direction=normal.tolist(), focal_point_mm=center.tolist(),
                parallel_scale_mm=plot.camera.parallel_scale, margin_factor=margin)


def render_biological(scene, mesh, out):
    """Three coordinated panels; all exemplar/explosion changes stay in display copies."""
    from PIL import Image, ImageDraw, ImageFont
    from visualization import style as st

    labels={'NEUN':'NeuN','LFB':'LFB','AT':'AT8 + pTDP-43',
            'ASYN':'α-synuclein','I6':'Iba1 + 6E10','GFAP':'GFAP'}
    modality_colors={'QNP':st.CAT3[0],'OMICS':st.CAT3[1]}
    font=ImageFont.truetype('DejaVuSans.ttf',32)
    small=ImageFont.truetype('DejaVuSans.ttf',27)
    heading=ImageFont.truetype('DejaVuSans.ttf',45)
    figure=Image.new('RGB',(4200,3000),'#fafbfc'); draw=ImageDraw.Draw(figure)
    cameras={}; display={}; outlines={}

    def quad(layer):
        # A common coronal-axis frame conveys the cutting plane, independent of
        # each RGB canvas's in-plane rotation. Underlying image/cell geometry stays put.
        points=layer['rgb_quad'] if layer['modality']=='QNP' else layer['cell_xyz']
        lo,hi=points.min(0),points.max(0)
        return np.array([[lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],
                         [hi[0],lo[1],lo[2]],[lo[0],lo[1],lo[2]]])

    def outline(plot,points,color):
        actor=plot.add_mesh(pv.lines_from_points(np.vstack([points,points[:1]])),
                            color=color,line_width=1.6,lighting=False)
        actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0,-12)

    def project(plot,point):
        plot.renderer.SetWorldPoint(*point,1); plot.renderer.WorldToDisplay()
        x,y,_=plot.renderer.GetDisplayPoint()
        return float(x),float(plot.window_size[1]-1-y)

    def badge(x,y,text,color,radius=25):
        draw.rounded_rectangle((x-radius,y-radius,x+radius,y+radius),radius=radius,fill=color)
        draw.text((x,y),str(text),font=font,fill='white',anchor='mm')

    for number,layer in enumerate(scene['layers'],1):
        layer['section_id']=number
        outlines[str(number)]=quad(layer).tolist()
    # A and C use the original objects. No exemplar rotation or explosion enters here.
    for panel,size,origin,direction in [('A',(1500,1120),(0,100),[-1,1.6,.8]),
                                         ('C',(1500,760),(0,1500),[-1,0,0])]:
        plot=pv.Plotter(off_screen=True,window_size=size); plot.set_background('#fafbfc')
        plot.add_mesh(mesh,color='#969da5',opacity=.18,smooth_shading=True)
        points=[mesh.points]
        for layer in scene['layers']:
            if panel=='A':
                points.append(_draw_biology(plot,layer))
            plane=quad(layer); points.append(plane)
            outline(plot,plane,modality_colors[layer['modality']])
        cameras[panel]=_fit_biological_camera(plot,np.vstack(points),direction,margin=1.14)
        axes=pv.AxesAssemblySymmetric(x_label=('', '') if panel=='C' else ('Right','Left'),
                 y_label=('Rostral','Caudal'),z_label=('Dorsal','Ventral'),label_size=22,
                 label_position=1.15,label_color='#253346')
        plot.add_orientation_widget(axes,interactive=False,viewport=(0,0,.22,.28))
        plot.show(auto_close=False)
        figure.paste(Image.fromarray(plot.screenshot(return_img=True)).convert('RGB'),origin)
        if panel=='A':
            plot.export_html(str(out/'mtg_multimodal_3d_final_interactive.html'))
        else:
            for layer in scene['layers'][6:]:
                plane=quad(layer); anchor=plane.mean(0); anchor[2]=plane[:,2].max()+1.5
                x,y=project(plot,anchor)
                badge(origin[0]+x,origin[1]+y,str(layer['section_id']),modality_colors['OMICS'],22)
            plane=np.vstack([quad(layer) for layer in scene['layers'][:6]])
            anchor=plane.mean(0); anchor[2]=plane[:,2].min()-1.5
            x,y=project(plot,anchor)
            draw.text((x,origin[1]+y),'1–6',font=small,fill=modality_colors['QNP'],anchor='mt')
        plot.close()
        print(f'Rendered composition panel {panel}',flush=True)
    draw.text((30,25),'A   Template-space placement',font=heading,fill='#253346')
    draw.text((45,1240),'1–6  QNP',font=font,fill=modality_colors['QNP'])
    draw.text((45,1300),'7–9  MERFISH',font=font,fill=modality_colors['OMICS'])
    draw.text((30,1425),'C   Spatial sampling architecture',font=heading,fill='#253346')
    draw.text((45,2280),'QNP: 20 µm   |   MERFISH: 5 mm',font=font,fill='#253346')
    draw.text((45,2330),'Thickness: 5 µm',font=small,fill='#52514e')

    # B: nine independently framed biological observations in one 3x3 block.
    plot=pv.Plotter(off_screen=True,window_size=(2700,2160),shape=(3,3),border=False)
    plot.set_background('#fafbfc',all_renderers=True)
    for i,layer in enumerate(scene['layers']):
        plot.subplot(i//3,i%3)
        exemplar=layer
        display[str(i+1)]=dict(name=layer['name'],modality=layer['modality'],rotation_degrees=0.,
                               left_right_reflection=False)
        if layer['name']=='GFAP':
            rows,cols=np.nonzero(layer['rgba'][:,:,3])
            q=layer['rgb_quad']; h,w=layer['rgba'].shape[:2]
            center=q[0]+(cols.mean()+.5)/w*(q[1]-q[0])+(rows.mean()+.5)/h*(q[3]-q[0])
            reflected=q.copy(); reflected[:,0]=2*center[0]-reflected[:,0]
            # In this anterior view, screen right is world -X and screen up is +Z.
            # Positive rotation in world X/Z therefore appears clockwise on screen.
            theta=np.deg2rad(80.)
            rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            adjusted=reflected.copy()
            adjusted[:,[0,2]]=(reflected[:,[0,2]]-center[[0,2]])@rotation.T+center[[0,2]]
            exemplar=dict(layer,rgb_quad=adjusted)
            display[str(i+1)].update(left_right_reflection=True,
                                     rotation_degrees=80.,rotation_direction='clockwise in anterior view',
                                     operation_order=['left-right reflection','80 degree clockwise rotation'],
                                     reflection_axis='dorsal-ventral axis through tissue center',
                                     reflection_matrix_world_xyz=[[-1,0,0],[0,1,0],[0,0,1]],
                                     display_corners_world_mm=adjusted.tolist(),
                                     tissue_center_world_mm=center.tolist(),
                                     scope='Panel B exemplar only; original template placement remains unreliable')
        elif layer['name']=='1217500590':
            center=layer['cell_xyz'].mean(0)
            theta=np.deg2rad(45.)
            rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            def turned(points):
                adjusted=points.copy()
                adjusted[:,[0,2]]=(points[:,[0,2]]-center[[0,2]])@rotation.T+center[[0,2]]
                return adjusted
            exemplar=dict(layer,cell_xyz=turned(layer['cell_xyz']),
                          polygon_world=[turned(ring) for ring in layer['polygon_world']])
            display[str(i+1)].update(rotation_degrees=45.,rotation_direction='clockwise in anterior view',
                                     tissue_center_world_mm=center.tolist(),
                                     rotation_matrix_world_XZ=rotation.tolist(),
                                     scope='Panel B exemplar only; original section placement unchanged')
        points=_draw_biology(plot,exemplar,detail=True)
        cameras[str(i+1)]=_fit_biological_camera(plot,points,[0,1,0],margin=1.13)
    # Equal viewport sizes and one parallel scale enforce a common physical scale per modality.
    for indices in [range(6),range(6,9)]:
        common=max(cameras[str(i+1)]['parallel_scale_mm'] for i in indices)
        for i in indices:
            plot.subplot(i//3,i%3); plot.camera.parallel_scale=common
            cameras[str(i+1)]['parallel_scale_mm']=common
            display[str(i+1)]['millimetres_per_pixel']=2*common/720
    plot.show(auto_close=False); pixels=Image.fromarray(plot.screenshot(return_img=True)).convert('RGB'); plot.close()
    draw.text((1530,25),'B   Biological section exemplars',font=heading,fill='#253346')
    for i,layer in enumerate(scene['layers']):
        x,y=1500+(i%3)*900,100+(i//3)*900
        figure.paste(pixels.crop(((i%3)*900,(i//3)*720,(i%3+1)*900,(i//3+1)*720)),(x,y+85))
        color=modality_colors[layer['modality']]
        badge(x+40,y+35,str(i+1),color)
        draw.text((x+85,y+15),labels.get(layer['name'],layer['name']),font=font,fill='#253346')
        draw.line((x+20,y+70,x+875,y+70),fill=color,width=2)
    print('Rendered unified biological panel B',flush=True)

    # C inset: copied QNP planes separated caudally at 2mm (100x), never applied to A.
    plot=pv.Plotter(off_screen=True,window_size=(500,350)); plot.set_background('#fafbfc')
    exploded=[]
    for layer in scene['layers'][:6]:
        offset=(layer['section_id']-3.5)*2-layer['metadata']['z_offset_mm']
        plane=quad(layer)+layer['normal']*offset
        exploded.append(plane); outline(plot,plane,modality_colors['QNP'])
    cameras['C_exploded']=_fit_biological_camera(plot,np.vstack(exploded),[-1,0,0],margin=1.35)
    plot.show(auto_close=False)
    figure.paste(Image.fromarray(plot.screenshot(return_img=True)).convert('RGB'),(20,2420))
    for i,plane in enumerate(exploded):
        anchor=plane.mean(0); anchor[2]=plane[:,2].max() if i%2==0 else plane[:,2].min()
        x,y=project(plot,anchor)
        draw.text((20+x,2420+y+(-22 if i%2==0 else 6)),str(i+1),font=small,
                  fill=modality_colors['QNP'],anchor='mt')
    plot.close()
    draw.text((535,2440),'QNP 1–6: exploded ×100',font=font,fill=modality_colors['QNP'])
    draw.text((535,2500),'Gap: 2 mm',font=font,fill='#253346')
    draw.text((535,2560),'Thickness: 5 µm',font=small,fill='#52514e')
    legend=[('MTG template','#969da5'),('QNP outlines / IDs',modality_colors['QNP']),
            ('MERFISH outlines / IDs',modality_colors['OMICS']),('Neurons',st.CAT3[0]),
            ('Oligodendroglial lineage',st.CAT3[1]),('Other non-neuronal',st.CAT3[2]),
            ('Provider layer outlines',st.TEXT_2)]
    for i,(text,color) in enumerate(legend):
        x,y=35+(i%4)*1050,2830+(i//4)*65
        if 'outlines' in text:
            draw.rectangle((x,y+5,x+30,y+35),outline=color,width=3)
        else:
            draw.rectangle((x,y+5,x+30,y+35),fill=color)
        draw.text((x+45,y),text,font=small,fill='#253346')
    stem=out/'mtg_multimodal_3d_final_composition'
    montage=figure
    montage.save(stem.with_suffix('.png'))
    pdf=st.figure(4200/300*25.4,3000/300*25.4)
    ax=pdf.add_axes([0,0,1,1]); ax.imshow(montage,interpolation='none'); ax.set_axis_off()
    st.save(pdf,stem,formats=('pdf',))
    return dict(cameras=cameras,exemplars=display,outline_corners_world_mm=outlines,
                outline_convention='common template-coronal X/Z-aligned bounding frames; no change to slice normals or biological coordinates',
                composition='A upper left; C lower left; B right column, three rows',
                QNP_explosion=dict(panel='C inset only',spacing_mm=2.,factor=100.,thickness_mm=.005),
                original_scene_rotations_unchanged=True)


def biological_outputs(scene, mesh, out, prior_metadata):
    """Write a separate, reviewable refinement and provenance without changing prior outputs."""
    prior=json.loads(prior_metadata.read_text())
    for layer in scene['layers']:
        saved=next(r for r in prior['slices'] if r['name']==layer['name'])
        for key in ('transform','plane','z_offset_mm','thickness_mm','rotation_degrees','bounds_world_mm'):
            if layer['metadata'][key]!=saved[key]:
                raise ValueError(f"Geometry differs from prior scene: {layer['name']} {key}")
    S.add_mtg_biological_detail(scene,S.load_project_config())
    prior_outputs=[p for p in prior_metadata.parent.iterdir() if p.is_file()]
    paths=list(dict.fromkeys([*scene['sources'],*prior_outputs,*map(Path,prior['sources_sha256'])]))
    before={str(p):digest(p) for p in paths}
    def array_fingerprints():
        return {layer['name']:hashlib.sha256(b''.join(
            np.asarray(array).tobytes() for array in
            [layer['xyz'],layer.get('rgb_quad',[]),layer.get('rgba',[]),
             layer.get('cell_xyz',[]),layer.get('cell_class3',[]),*layer.get('polygon_world',[])]
        )).hexdigest() for layer in scene['layers']}
    representations=array_fingerprints()
    rendering=render_biological(scene,mesh,out)
    assert representations==array_fingerprints(),'Underlying biological arrays changed'
    assert before=={str(p):digest(p) for p in paths},'Input assets changed during rendering'
    metadata=dict(coordinate_frame=scene['frame'],display_z=scene['display_z'],
                  spacing_mm=scene['spacing_mm'],thickness_mm=scene['thickness_mm'],
                  visualization_only=True,new_registration_run=False,geometry_matches_prior=True,
                  prior_metadata=str(prior_metadata),sources_sha256=before,
                  slices=[dict(section_id=l['section_id'],name=l['name'],modality=l['modality'],**l['metadata']) for l in scene['layers']],
                  display_transformations=dict(
                      A='physical placement and all original geometry unchanged; bounding plane outlines added',
                      B='three-by-three independently centred anterior-coronal exemplars; common scale per modality',
                      GFAP='Panel B only: left-right reflection, then 80 degrees clockwise about tissue center; original placement remains unreliable',
                      MERFISH_1217500590='Panel B only: 45 degrees clockwise about measured-cell centroid',
                      C='true left lateral placement; QNP-only inset uses 2mm spacing (100x), thickness unchanged',
                      layer_annotations='white halo and raster depth bias only; provider vertices unchanged'),
                  rendering=rendering,source_representation_sha256=representations,
                  underlying_arrays_unchanged=True,
                  code_sha256={str(p):digest(p) for p in [Path(__file__),Path(S.__file__)]})
    (out/'scene_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'REPORT.md').write_text('''# MTG final figure composition

**A:** Unchanged oblique template placement, RGB histology, measured MERFISH cells,
and crisp rectangular plane outlines. QNP outlines/IDs are blue; MERFISH are orange.
Outlines use a common template-coronal X/Z-aligned frame around the existing mapped
image/cell extents; they are not tissue boundaries. Earlier QNP outlines followed
rotated RGB-canvas edges, whereas MERFISH outlines followed the template axes. That
mixed border convention, rather than different slice normals, caused the misleading
appearance. Biological geometry and all original cutting-plane normals are unchanged.
The template is slightly more opaque (0.18). Grouped labels identify QNP 1–6 and MERFISH 7–9.

**B:** One 3×3 biological block: 1 NeuN, 2 LFB, 3 AT8+pTDP-43; 4 α-synuclein,
5 Iba1+6E10, 6 GFAP; 7 1194111462, 8 1217500585, 9 1217500590. Equal viewport
sizes and a common orthographic scale per modality preserve physical display scale
within QNP and within MERFISH. These are separately magnified exemplars, not another
template-space placement. Existing measured RGB maps, CLASS3 cell assignments, and
provider polygons are reused. Only section 7 has compatible provider layer outlines;
their white halo and raster depth bias improve contrast without moving any vertices.

GFAP is **reflected left–right, then rotated 80° clockwise** in its Panel B display
copy, about its tissue-mask centroid. Section **1217500590 is rotated 45° clockwise**
in its Panel B display copy about its measured-cell centroid. Clockwise is defined
in the anterior-coronal viewport. These are visual presentation choices only.
Pivots, operation order, matrices, and adjusted GFAP corners are in scene_metadata.json.
No display-adjustment notes are printed on the biological figure tiles. GFAP's production
alignment remains unreliable. Panel A, Panel C, and interactive template placement
retain its original numeric chain. AT8 receives **zero** additional rotation: its
production correction is not applied twice.

**C:** Validated left-lateral placement shows dense QNP versus sparse MERFISH
sampling. Original illustrative intervals remain **20 µm QNP / 5 mm MERFISH**, with
**5 µm display thickness** for both. The separate QNP inset copies the same planes
with **2 mm gaps (100× display exaggeration)** to identify sections 1–6. This change
is isolated from A and the interactive scene. Section numbers link all three panels.

Template coordinates, coronal planes, caudal stacking, placement transforms, and
biological source arrays are unchanged and checked against the prior metadata.
Source/prior-output hashes and array fingerprints are recorded. No registration,
affine/similarity fit, xIV-LDDMM, biological analysis, feature encoding, particle
recomputation/compression, atlas changes, or earlier publication Figure A–C pipeline
was run. Spacings are illustrative conventions, not acquisition positions. All
template placements remain provisional; two MERFISH sections use the shared C5
display affine. QNP and MERFISH are from different donors. This is a qualitative
figure, not a physical reconstruction or quantitative model.

PNG/PDF use the existing publication exporter. Interactive HTML directly exports A,
including original biological placement and modality outlines; it intentionally does
not incorporate the GFAP exemplar reflection or QNP inset explosion. Montage legends,
number badges, and orientation widgets are static annotations.
''')
    print(f'Biological refinement written to {out}; geometry preserved',flush=True)


def export_neuroglancer(scene, mesh, out):
    from visualization import ng_export as nx

    # This standalone process serves only this bundle, using the existing CORS machinery.
    nx.SERVE_ROOT = out
    bundle = out/'neuroglancer'
    layers, records = [], {}
    template = dict(name='Template MTG (surface cloud)', xyz=mesh.points[::3],
                    color='#969da5', metadata=dict(representation='decimated surface vertices, every third vertex'))
    for i, layer in enumerate([template, *scene['layers']]):
        name = layer['name']
        if name == 'GFAP':
            name += ' (alignment unreliable)'
        if layer['metadata'].get('placement_status', '').startswith('unvalidated'):
            name += ' (shared display affine; unvalidated)'
        path = bundle/'data'/f'layer_{i:02d}'
        asset = nx.write_annotations(path, 'point', layer['xyz']*1000.,
                                     {'kind': ('uint8', np.zeros(len(layer['xyz']), np.uint8))},
                                     dimensions=('x','y','z'))
        layers.append(nx.annotation_layer(name, path,
                      nx.categorical_shader('kind', [layer['color']], size=2. if i==0 else 3.,
                                            alpha=.25 if i==0 else .65)))
        records[name] = dict(asset, **layer['metadata'])
    center = mesh.center
    state = nx.build_state(layers, np.asarray(center)*1000., mesh.length*1000.,
                           extra_dims={'z':[1e-6,'m']})
    state.update(position=(np.asarray(center)*1000.).tolist(), layout='3d',
                 projectionOrientation=[.3,.15,.1,.93675])
    paths = nx.write_scene(bundle, 'mtg_multimodal_3d', state,
                          dict(scene_frame=scene['frame'], display_z=scene['display_z'],
                               spacing_mm=scene['spacing_mm'], thickness_mm=scene['thickness_mm'],
                               visualization_only=True, layers=records,
                               note='Surface cloud and slice point planes; thickness is metadata only in Neuroglancer'))
    command = ('PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:src/preprocess:src/config_setup:src/scripts '
               '/cis/home/dpadova/miniconda3/envs/wsi-pipeline/bin/python '
               f'src/scripts/render_mtg_multimodal_3d.py --serve --output-dir {out}')
    (out/'NEUROGLANCER.md').write_text(
        '# Neuroglancer\n\nStart from the repository root:\n\n```bash\n'+command+
        '\n```\n\nThe command prints a local viewer link. Keep it running while viewing. '
        'The link is local to this computer; remote access requires forwarding the asset and viewer ports. '
        'No data is uploaded. The default scene uses the synthetic spacing, not the exploded spacing.\n')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=S.PROJECT/'results/analysis/mtg_multimodal_3d_visualization')
    parser.add_argument('--rotations-json', type=Path, help='Mapping stain to display-only degrees')
    parser.add_argument('--include-donor-sections', action='store_true',
                        help='Explicitly unvalidated transfer of reviewed affine to two donor sections')
    parser.add_argument('--serve', action='store_true', help='Serve the saved Neuroglancer scene without rebuilding')
    parser.add_argument('--port', type=int, default=8012, help='Local asset server port')
    parser.add_argument('--viewer-port', type=int, default=9012, help='Local Neuroglancer viewer port')
    parser.add_argument('--biological-detail', action='store_true',
                        help='Render RGB/class/layer refinement into a fresh output directory')
    parser.add_argument('--final-composition', dest='biological_detail',action='store_true',
                        help='Render coordinated placement, nine exemplars, and sampling panels')
    parser.add_argument('--prior-scene', type=Path,
                        default=S.PROJECT/'results/analysis/mtg_multimodal_3d_visualization/scene_metadata.json')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(Path.home().resolve()):
        parser.error('All outputs must stay under HOME')
    if args.serve:
        from visualization import ng_export as nx
        nx.SERVE_ROOT = out
        nx.serve([out/'neuroglancer/mtg_multimodal_3d.state.json'], port=args.port, ng_port=args.viewer_port)
        return
    # An existing nonempty output directory is never reused, even after a partial render.
    if out.exists() and any(out.iterdir()):
        parser.error(f'Refusing to overwrite existing outputs: {out}')
    rotations = json.loads(args.rotations_json.read_text()) if args.rotations_json else {}
    scene = S.build_mtg_multimodal_3d(S.load_project_config(), rotations, args.include_donor_sections)
    if args.rotations_json:
        scene['sources'].append(args.rotations_json.resolve())
    files = list(dict.fromkeys(scene['sources']))
    before = {str(p): digest(p) for p in files}
    faces = np.column_stack([np.full(len(scene['faces']),3), scene['faces']])
    mesh = pv.PolyData(scene['vertices'], faces).decimate(.85)
    out.mkdir(parents=True, exist_ok=True)
    if args.biological_detail:
        biological_outputs(scene,mesh,out,args.prior_scene)
        return
    images = []
    for view in ('overview', 'side', 'top', 'exploded'):
        images.append(render(scene, mesh, out/f'mtg_multimodal_3d_{view}.png', view, interactive=view=='overview'))
        print(f'Rendered {view}', flush=True)
    write_montage(images, out, scene)
    export_neuroglancer(scene, mesh, out)
    after = {str(p): digest(p) for p in files}
    if before != after:
        raise RuntimeError('Input assets changed during rendering')
    rows = [dict(name=l['name'], modality=l['modality'], **l['metadata']) for l in scene['layers']]
    metadata = dict(coordinate_frame=scene['frame'], display_z=scene['display_z'],
                    anatomical_axes={'+X':'Right', '-X':'Left', '+Y':'Rostral', '-Y':'Caudal',
                                     '+Z':'Dorsal', '-Z':'Ventral'},
                    slice_orientation='Coronal: in-plane X/Z; normal -Y (caudal)',
                    static_output=dict(file='mtg_multimodal_3d_montage.png', size_px=[3600,2700],
                                       shared_legend=True, explanatory_panel_footers=False,
                                       panel_order=['oblique','left lateral','coronal anterior','exploded']),
                    visualization_only=True, physical_reconstruction=False, new_registration_run=False,
                    spacing_mm=scene['spacing_mm'], thickness_mm=scene['thickness_mm'],
                    stack_origin='Each modality centred on its existing template plane; order is display-only',
                    template=dict(hemisphere='left', label='MTG (30)', vertices=len(mesh.points),
                                  faces=mesh.n_cells, representation='85% triangle-decimated isosurface'),
                    slice_representation='Deterministically subsampled point planes + 5um bounding slab wireframes; no textures',
                    slices=rows, sources_sha256=before, input_hashes_unchanged=before==after,
                    exploded=dict(QNP_spacing_mm=2., QNP_spacing_factor=100., thickness_mm=.005),
                    overview_reference_figure_available=False,
                    code_sha256={str(p):digest(p) for p in (Path(__file__), Path(S.__file__))})
    (out/'scene_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (out/'slice_display_table.json').write_text(json.dumps([
        {k:r[k] for k in ('name','modality','order','z_offset_mm','thickness_mm','rotation_degrees')}
        for r in rows], indent=2)+'\n')
    transferred = args.include_donor_sections
    report = f'''# MTG multimodal 3D visualization

Rendered the left AHRAv2 MTG label (30) as a transparent gray surface, six QNP stain
point planes (blue/purple), and {sum(l['modality']=='OMICS' for l in scene['layers'])} OMICS section point plane(s) (orange).
Only MTG anatomy is rendered. Slice tissue footprints are retained in full, including
portions outside the atlas MTG, so provisional placement mismatch stays visible.

## Existing placement reused

QNP: `load_qnp_transforms` reads stored slide-to-NeuN matrices; these are chained
through `load_plane_hypothesis` to the provisional MTG-basin pose A
(`p+000_w+000_y-027.5`). Plane origin and axes embed these coordinates in MNI world mm.
No nonlinear fit was inverted or rerun. Existing transform status is retained per stain.

OMICS: inverse of selected C5 from
`results/analysis/omics_atlas_affine_initialization/initial_affine_candidates.json`:
section = 0.6 * native_plane + (43.05, 12.1) mm. The existing input manifest embeds
this plane at MNI Y = -30 mm. Full-section exports supply provider coordinates in um.
{'Two additional donor sections inherit this SAME affine solely for display; their in-plane placement is unvalidated and is not an independently established correspondence.' if transferred else 'Only section 1194111462 has a reviewed placement; other sections are omitted rather than assigning them unsupported transforms. Thus the OMICS 5 mm interval is specified but cannot be displayed between sections.'}

## Artificial geometry and rotations

QNP centre spacing **20 µm**, thickness **5 µm**; OMICS centre spacing **5 mm**,
thickness **5 µm**. Offsets are centred on each existing plane and applied along
its normal (MNI **-Y**, not anatomical MNI Z). Slice order is a presentation convention,
not acquisition order. Bounding slab wireframes show rectangular extents, not tissue boundaries.
The exploded view alone expands QNP spacing to **2 mm (100×)** for readability;
it leaves thickness and template geometry unchanged.

Display rotations in degrees: {json.dumps(rotations or {l['name']:0 for l in scene['layers'] if l['modality']=='QNP'})}.
Rotations are applied after mapping, about each mapped footprint centroid, and recorded
with their pivots. The mentioned overview attachment was unavailable, so no visual
rotation correction was inferred. Source transforms are unchanged.

## Interpretation and limitations

This is a **visualization-only composite, not a physical 3D reconstruction or a
quantitative model**. **No new registration, affine fitting, compression, template
selection, xIV-LDDMM, or Figure A/B/C generation was run.** QNP donor H20.33.034 and
OMICS donor H20.33.001 are different individuals. Pose A is provisional and individual
QNP GFAP alignment is flagged unreliable in the existing graph and is shown only
as context; the C5 OMICS affine is a provisional visual placement.
Full footprints need not lie entirely inside the template MTG. No clipping is used
to imply anatomical agreement. QNP planes overlap at macro scale; use the exploded
view and interactive orbit to distinguish them. En face projection collapses depth.

Rendering simplifications: 85% triangle reduction of the 500 µm label isosurface;
existing 500 µm QNP particles; at most 12,000 deterministically sampled cells per OMICS
section, uniform section colors, no textures. Point sizes are screen-space marks,
not physical cell sizes or thickness. Wireframes carry the specified slab thickness.

The overview communicates the common frame; the exploded view communicates the six
QNP planes more clearly. Suitable for qualitative presentation with these captions,
not as evidence of registration accuracy. Static views use equal physical axis scale;
interactive HTML is self-contained and supports orbit/zoom. All {len(files)} input
file SHA-256 hashes were unchanged after rendering. Exact sources, transforms, display
offsets, rotations, and simplifications are in `scene_metadata.json`.
The four views are exported together in `mtg_multimodal_3d_montage.png`.
Each panel shows only its view title and anatomical orientation cartoon; one legend
serves the montage. RAS world directions map +X/-X to right/left, +Y/-Y to
rostral/caudal, +Z/-Z to dorsal/ventral. The left lateral camera is along -X and
the anterior coronal camera along +Y, with dorsal up. The camera-normal axis labels
are suppressed in the two orthographic views to prevent overlapping text.
Both stored slice embeddings are explicitly checked to be coronal (X/Z planes,
normal -Y); these checks verify the display embedding, not acquisition orientation.
Neuroglancer uses the same 3-D coordinates, a template surface cloud, and slice point
planes; 5 µm thickness is recorded in metadata only in this viewer. See `NEUROGLANCER.md`
for the local serving command. Its 3-D view supports independent layer visibility.
'''
    (out/'REPORT.md').write_text(report)
    print(f'Wrote {out}; {len(files)} input hashes unchanged', flush=True)


if __name__ == '__main__':
    main()
