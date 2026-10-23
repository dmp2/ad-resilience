"""Protect measured-gene fallback and free optimized-particle support rendering."""
from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC/'preprocess'), str(SRC/'config_setup')]

from visualization import figures_omics as O, figures_qnp as Q, scene as S, style as st


def test_omics_scale_space_keeps_fine_references_and_original_gene_positions(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    class_names = tuple(c for members in S.CLASS3.values() for c in members)
    cells = S.Points('cells','merfish_section_mm',np.array([[0,0],[1,0],[0,1],[1,1.]]),'measured','synthetic',
        categories={'class3':S.Categorical(np.array([0,1,2,0]),tuple(S.CLASS3))},
        scalars={'lognorm:CUX2':S.Scalar(np.array([0,1,2,3.]),'test')},
        meta={'donor':'synthetic','barcode':'0','section':'0'})
    refs = {}
    for b in O.BINS_MM:
        refs[b] = S.Points('reference',cells.frame,np.array([[.2,.2],[.8,.8]]),'derived','measured cells',
            scalars={'cells_per_bin':S.Scalar(np.array([4.,8.]),'count'),
                     'lognorm:CUX2':S.Scalar(np.array([.25,.75]),'test')},
            meta={'bin_mm':b,'nu_class':np.ones((2,len(class_names))), 'class_names':class_names})
    optimized = {}
    for b in (.5,1.):
        optimized[b] = S.Points('optimized',cells.frame,refs[b].xy+.123,'derived','saved optimization',
            scalars={'cells_per_bin':S.Scalar(np.array([3.,9.]),'count')},
            meta={'compression_method':'image_varifold','nu_class':np.ones((2,len(class_names))),
                  'class_names':class_names})
    monkeypatch.setattr(S,'bin_merfish',lambda cells,b,genes=():refs[b])
    monkeypatch.setattr(O,'_cell_spacing_mm',lambda _: .02)
    monkeypatch.setattr(st,'marker_size',lambda *args: 1.)
    seen=[];saved=[]
    monkeypatch.setattr(O,'_bin_squares',lambda ax,p,faces,ref:seen.append(p))
    def capture(fig,*args,**kwargs):
        saved.append((fig.get_size_inches().copy(),np.array([ax.get_position().bounds for ax in fig.axes]),
                      ' '.join(t.get_text()for ax in fig.axes for t in ax.texts)))
        plt.close(fig)
        return []
    monkeypatch.setattr(st,'save',capture)
    O.figure_a(cells,tmp_path)
    seen.clear()
    O.figure_a(cells,tmp_path,representation={'points':optimized,'caption':['saved class optimization; original genes']})
    assert seen[:2]==[refs[.1],refs[.25]]
    assert seen[2:4]==[optimized[.5],optimized[1.]]
    assert seen[4:]==list(refs.values())  # Every gene panel retains original positions and values.
    assert all('lognorm:CUX2'not in p.scalars for p in optimized.values())
    np.testing.assert_array_equal(saved[0][0],saved[1][0])
    np.testing.assert_allclose(saved[0][1],saved[1][1],atol=1e-12)
    assert 'unavailable'not in saved[1][2].lower()
    assert 'deterministic reference'in saved[1][2]
    assert 'optimized class masses'in saved[1][2]


def test_optimized_support_is_mass_weighted_points_at_saved_coordinates():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    xy=np.array([[.13,.27],[1.28,.96]])
    points=S.Points('optimized','native',xy,'derived','saved',
        scalars={'area_mm2':S.Scalar(np.array([.5,1.5]),'area'),
                 'cells_per_bin':S.Scalar(np.array([3.,9.]),'count')},
        meta={'compression_method':'image_varifold'})
    fig,axes=plt.subplots(1,2)
    q=Q._particle_squares(axes[0],points,1.,np.array([.2,.8]),st.SEQ_BLUE,Normalize(0,1),1.)
    o=O._bin_squares(axes[1],points,['blue','orange'],3.)
    for coll in (q,o):np.testing.assert_array_equal(coll.get_offsets(),xy)
    np.testing.assert_array_equal(q.get_sizes(),[6.,18.])
    np.testing.assert_array_equal(o.get_sizes(),[12.,36.])
    assert len(axes[0].patches)==len(axes[1].patches)==0
    plt.close(fig)
