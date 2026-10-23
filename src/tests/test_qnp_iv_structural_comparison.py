"""Protect coarse-only variant routing and corresponding fine-start provenance."""
from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC), str(SRC / 'preprocess'), str(SRC / 'config_setup')]

from scripts import align_qnp_multistain as align
from scripts.compare_qnp_iv_structural import particle_source
from preprocess.section_alignment import Similarity2D


def test_only_coarse_particles_use_variant_archives(tmp_path):
    native, iv = tmp_path/'native', tmp_path/'iv'
    for stain in ('NEUN','LFB','AT','ASYN','GFAP','I6'):
        stem = f'slide-{stain}'
        fine = native/f'{stem}_tissue_classes_500um.npz', 'nu_Sub'
        assert particle_source(native,stem,stain,'fine','baseline',iv) == fine
        assert particle_source(native,stem,stain,'fine','iv_structural',iv) == fine
        assert particle_source(native,stem,stain,'coarse','baseline',iv) == (
            native/f'{stem}_tissue_classes_1000um.npz', 'nu_Sub')
        assert particle_source(native,stem,stain,'coarse','iv_structural',iv) == (
            iv/f'{stain}_1000um.npz', 'nu')
    with pytest.raises(ValueError):
        particle_source(native,'slide','AT','fine','iv_modality',iv)
    with pytest.raises(ValueError):
        particle_source(native,'slide','AT','500um','baseline',iv)


def test_each_fine_fit_uses_its_own_coarse_winner(monkeypatch):
    coarse, fine = object(), object()
    P, mu = np.ones((4,2)), np.ones((4,2))
    Pf, muf = np.full((7,2),3.), np.full((7,2),4.)
    starts = [Similarity2D(11.,1.,False,(1.,2.)), Similarity2D(13.,1.,True,(3.,4.))]
    calls = []
    active = [None]

    def search(scorer, p, nu, **kwargs):
        assert scorer is coarse and p is P and nu is mu
        assert kwargs['refine_top']==4 and kwargs['scale_bounds']==(1.,1.)
        np.testing.assert_array_equal(kwargs['thetas'],np.arange(0,360,10))
        assert kwargs['flips']==[False,True] and kwargs['h']==.5 and kwargs['blur_mm']==1.
        return [(0.2,active[0])]

    def refine(scorer,p,nu,start,bounds,max_evals):
        assert scorer is fine and p is Pf and nu is muf and max_evals==200
        calls.append((start,bounds))
        # Return a visibly changed result, so the free-scale call can be distinguished.
        return .1, Similarity2D(start.theta_deg+.5,1.,start.flip,start.tau)

    monkeypatch.setattr(align,'search',search)
    monkeypatch.setattr(align,'refine',refine)
    for start in starts:
        active[0]=start
        ranked,_,best,_,_,_ = align.fit_registration_stages(coarse,fine,P,mu,Pf,muf,{})
        assert ranked[0][1] is start
        assert calls[-2] == (start,(1.,1.))
        assert calls[-1] == (best,(.85,1.15))
        assert Similarity2D.from_dict(ranked[0][1].as_dict()) == start
    assert calls[0][0] != calls[2][0]
