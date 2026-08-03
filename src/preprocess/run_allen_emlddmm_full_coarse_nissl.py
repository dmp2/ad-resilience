"""Staged, memory-bounded coarse MRI-to-symmetric-Nissl EM-LDDMM workflow."""
from __future__ import annotations

import argparse, csv, gc, hashlib, json, math, os, resource, shutil, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch

from preprocess.prepare_allen_emlddmm_inputs import accepted_loader_axes
from preprocess.run_allen_emlddmm import load_physical_rows, load_pinned_mri_image, pinned_emlddmm, sha256_file
from preprocess.visualize_allen_emlddmm_stack import (
    _coordinate_cell_edges,
    _render_saved_transform_stack_overview,
    _uniform_representative_positions,
)

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.environ.get("EMLDDMM_OUTPUT_ROOT", PROJECT / "results/allen/specimen_708424/emlddmm/full-coarse/HIST_NISSL_to_MRI_7T_WHOLE_eA1e6"))
RUN_TMP = Path(os.environ.get("EMLDDMM_RUN_TMP", "/invalid/run-tmp"))
PEAK_FILE = Path(os.environ.get("EMLDDMM_RSS_PEAK_FILE", RUN_TMP / "process_group_peak_rss_kib"))
DATASET = PROJECT / "data/derivatives/allen/specimen_708424/emlddmm_7t_symmetric"
VIEW = DATASET / "inputs/views/HIST_NISSL"
MRI = PROJECT / "data/derivatives/allen/specimen_708424/mri_7t_whole/T1_rot_space-MRI_7T_WHOLE_desc-header-corrected.nii"
MRI_PROV = PROJECT / "data/derivatives/allen/specimen_708424/mri_7t_whole/mri_provenance.json"
INITIAL_A = PROJECT / "results/qc/allen_708424_mri7t_to_symmetric_nissl_initial_similitude.txt"
CHECKPOINTS = OUTPUT / "checkpoints"
ATLAS_DIR = OUTPUT / "section_alignment_atlas_free"
REG_DIR = OUTPUT / "registration"
POST_DIR = OUTPUT / "postprocessed_baseline_factored_qc"
PIN = "864990e0619fcdfb3e22e05298291f439f1b6f3d"


def now() -> str: return datetime.now(timezone.utc).isoformat()
def peak_rss_kib() -> int:
    own = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try: group = int(PEAK_FILE.read_text().strip())
    except Exception: group = 0
    return max(own, group)
def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
def checksum(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()
def finite(name: str, x: Any) -> np.ndarray:
    a=x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    if not np.all(np.isfinite(a)): raise RuntimeError(f"{name} contains nonfinite values")
    return a
def read_samples() -> list[dict[str,str]]:
    with (VIEW/"samples.tsv").open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f,delimiter="\t"))
def load_context():
    if (PROJECT/"configs/emlddmm-upstream-commit.txt").read_text().strip()!=PIN: raise RuntimeError("pin file mismatch")
    em=pinned_emlddmm(); rows=load_physical_rows(DATASET); samples=read_samples()
    if len(rows)!=2846 or len(samples)!=2846: raise RuntimeError("expected 2,846 manifest rows")
    canvas=json.loads((DATASET/"metadata/loader_canvas_audit.json").read_text())
    axes=accepted_loader_axes(rows,canvas)
    observed=np.asarray([i for i,(r,s) in enumerate(zip(rows,samples)) if s["status"]=="present" and r["stain"]=="nissl"],dtype=np.int64)
    if observed.size!=641: raise RuntimeError(f"expected 641 Nissl observations, got {observed.size}")
    return em,rows,samples,axes,observed

def block_axis(axis: np.ndarray, factor: int) -> np.ndarray:
    n=len(axis)//factor
    return np.asarray(axis[:n*factor],dtype=np.float64).reshape(n,factor).mean(1)
def read_section(sample: dict[str,str], em=None, spatial_axes=None) -> tuple[np.ndarray,np.ndarray]:
    path=VIEW/sample["sample_id"]
    raw=tifffile.imread(path)
    if raw.dtype==np.uint8: image=raw[...,:3].astype(np.float64)/255.0
    else:
        image=raw[...,:3].astype(np.float64); image/=np.mean(np.abs(image.reshape(-1,3)),axis=0)
    image=image.transpose(2,0,1)
    if em is not None and spatial_axes is not None:
        source_y=np.arange(image.shape[1],dtype=np.float64)*200.0-(image.shape[1]-1)*100.0
        source_x=np.arange(image.shape[2],dtype=np.float64)*200.0-(image.shape[2]-1)*100.0
        query=torch.stack(torch.meshgrid(torch.as_tensor(spatial_axes[0]),torch.as_tensor(spatial_axes[1]),indexing="ij"))
        image=em.interp([source_y,source_x],torch.as_tensor(image),query,interp2d=True,padding_mode="zeros").numpy()
    support=(image[0]>0).astype(np.float32)
    return image.astype(np.float32),support
def downsample_section(image: np.ndarray,support: np.ndarray,factor: int=4):
    c,h,w=image.shape; nh,nw=h//factor,w//factor; h4,w4=nh*factor,nw*factor
    sup=support[:h4,:w4].reshape(nh,factor,nw,factor).sum((1,3))
    num=(image[:,:h4,:w4]*support[None,:h4,:w4]).reshape(c,nh,factor,nw,factor).sum((2,4))
    out=np.zeros((c,nh,nw),np.float32); positive=sup>0; out[:,positive]=num[:,positive]/sup[positive]
    return out,(sup/(factor*factor)).astype(np.float32)
def stream_stack(samples,indices,total_rows,shape=(130,182),em=None,spatial_axes=None):
    J=np.zeros((3,total_rows,*shape),np.float32); W=np.zeros((total_rows,*shape),np.float32)
    for count,index in enumerate(indices,1):
        image,support=read_section(samples[index],em,spatial_axes); image,support=downsample_section(image,support)
        if image.shape!=(3,*shape) or support.shape!=shape: raise RuntimeError(f"downsample shape error at row {index}")
        destination=count-1 if total_rows==len(indices) else index
        J[:,destination]=image; W[destination]=support
        if count%50==0: print(f"streamed {count}/{len(indices)} sections",flush=True)
    return J,W

def rigid_frame(mats: np.ndarray):
    u,_,vt=np.linalg.svd(mats[:,:2,:2].mean(0)); q=u@vt
    if np.linalg.det(q)<0: u[:,-1]*=-1; q=u@vt
    B=np.eye(3); B[:2,:2]=q; B[:2,2]=np.median(mats[:,:2,2],axis=0); return B

def residuals(path,rows,indices,mats,B):
    R=np.linalg.inv(B)[None]@mats[indices]; err=float(np.max(np.abs(B[None]@R-mats[indices])))
    tx,ty=R[:,0,2],R[:,1,2]; rot=np.degrees(np.arctan2(R[:,1,0],R[:,0,0]))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["physical_index","allen_section","serial_z_um","tx_um","ty_um","rotation_deg"])
        for i,a,b,c in zip(indices,tx,ty,rot): w.writerow([i,rows[i]["allen_section_number"],float(rows[i]["serial_z_center_mm"])*1000,a,b,c])
    return {"tx_um":[float(tx.min()),float(tx.max())],"ty_um":[float(ty.min()),float(ty.max())],"rotation_deg":[float(rot.min()),float(rot.max())],"recomposition_max_abs_error":err}

def mini_equivalence(em,rows,samples,axes,observed):
    chosen=observed[[0,len(observed)//2,-1]]; mini=RUN_TMP/"support-equivalence-mini-view"; mini.mkdir()
    with (mini/"samples.tsv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["sample_id","participant_id","species","status"])
        for i in chosen:
            s=samples[i]; w.writerow([s["sample_id"],s["participant_id"],s["species"],"present"])
            stem=Path(s["sample_id"]).stem
            for suffix in (".tif",".json"):
                src=(VIEW/(stem+suffix)).resolve(); os.symlink(os.path.relpath(src,mini),mini/(stem+suffix))
    obj=em.Image(space="HIST_SYMMETRIC_MINI",name="HIST_NISSL",fpath=str(mini),x=[axes[0][chosen],axes[1],axes[2]])
    image_error=0.0; support_error=0.0
    for j,i in enumerate(chosen):
        image,support=read_section(samples[i],em,[axes[1],axes[2]]); image_error=max(image_error,float(np.max(np.abs(obj.data[:,j]-image))))
        support_error=max(support_error,float(np.max(np.abs(obj.mask[j]-support))))
    result={"physical_indices":chosen.tolist(),"image_max_abs_error":image_error,"support_max_abs_error":support_error,"equivalent":image_error<=1e-6 and support_error==0.0}
    del obj; plt.close("all"); shutil.rmtree(mini)
    if not result["equivalent"]: raise RuntimeError(f"streamed support equivalence failed: {result}")
    return result

def validate_rigid(mats):
    hom=float(np.max(np.abs(mats[:,2]-np.array([0,0,1])))); ortho=float(np.max(np.abs(np.swapaxes(mats[:,:2,:2],1,2)@mats[:,:2,:2]-np.eye(2))))
    det=np.linalg.det(mats[:,:2,:2]); return {"homogeneous_row_max_error":hom,"orthogonality_max_error":ortho,"determinant_range":[float(det.min()),float(det.max())]}

def atlas_free():
    start=time.monotonic(); em,rows,samples,axes,observed=load_context(); ATLAS_DIR.mkdir(parents=True,exist_ok=True)
    audit=mini_equivalence(em,rows,samples,axes,observed)
    J,W=stream_stack(samples,observed,len(observed),em=em,spatial_axes=[axes[1],axes[2]]); x=[np.asarray(axes[0][observed]),block_axis(axes[1],4),block_axis(axes[2],4)]
    assert J.shape==(3,641,130,182); assert W.shape==(641,130,182); assert tuple(map(len,x))==(641,130,182)
    gaps=np.diff(x[0]); before={"stage":"atlas-free","status":"running","shapes":{"J":list(J.shape),"W":list(W.shape)},"coordinate_lengths":list(map(len,x)),"spacings_um":{"observed_serial_gaps":{"min":float(gaps.min()),"median":float(np.median(gaps)),"max":float(gaps.max())},"row":float(np.diff(x[1]).mean()),"column":float(np.diff(x[2]).mean())},"pre_downsample":[4,4],"optimizer_downI":[1,2,2],"optimizer_downJ":[1,2,2],"effective_inplane_spacing_um":1600.0,"support_equivalence":audit,"start_time":now(),"peak_process_group_rss_kib":peak_rss_kib()}
    atomic_json(CHECKPOINTS/"atlas-free.json",before); print(json.dumps(before,indent=2),flush=True)
    draw=em.draw
    try: em.draw=False; out=em.atlas_free_reconstruction(J=J,xJ=x,W=W,n_steps=10,eA2d=2e4,downI=[1,2,2],downJ=[1,2,2])
    finally: em.draw=draw; plt.close("all")
    P=finite("atlas A2d",out["A2d"]).astype(np.float64)
    if P.shape!=(641,3,3): raise RuntimeError(f"unexpected A2d {P.shape}")
    B=rigid_frame(P); full=np.repeat(B[None],2846,axis=0); full[observed]=P
    p_obs=ATLAS_DIR/"observed_A2d.npy"; p_full=ATLAS_DIR/"expanded_2846_A2d.npy"; p_idx=ATLAS_DIR/"observed_physical_indices.npy"; p_b=ATLAS_DIR/"common_bookkeeping_frame.txt"
    np.save(p_obs,P); np.save(p_full,full); np.save(p_idx,observed); np.savetxt(p_b,B)
    ranges=residuals(ATLAS_DIR/"observed_residual_transforms.tsv",rows,observed,full,B); residual_plot(ATLAS_DIR/"observed_residual_transform_summary.png",axes,observed,full,B); rigid=validate_rigid(P)
    placement=bool(np.array_equal(np.load(p_idx),observed) and np.array_equal(x[0],axes[0][observed]) and np.allclose(full[observed],P))
    if not placement or ranges["recomposition_max_abs_error"]>1e-8 or rigid["homogeneous_row_max_error"]>1e-5: raise RuntimeError("atlas matrix validation failed")
    elapsed=time.monotonic()-start
    done={**before,"status":"complete","completion_time":now(),"elapsed_seconds":elapsed,"peak_process_group_rss_kib":peak_rss_kib(),"outputs":{"observed_A2d":str(p_obs),"expanded_A2d":str(p_full),"observed_indices":str(p_idx),"bookkeeping_frame":str(p_b),"residuals":str(ATLAS_DIR/"observed_residual_transforms.tsv")},"checksums":{str(p):checksum(p) for p in (p_obs,p_full,p_idx,p_b)},"matrix_validation":{"finite":True,"observed_shape":list(P.shape),"expanded_shape":list(full.shape),"rigid":rigid,"ranges":ranges,"placement_exact":placement}}
    atomic_json(CHECKPOINTS/"atlas-free.json",done); print(json.dumps(done,indent=2),flush=True)

@dataclass
class LightImage:
    space:str; name:str; data:np.ndarray; x:list[np.ndarray]; title:str; names:list[str]
    def fnames(self): return self.names

def load_checkpoint(stage):
    p=CHECKPOINTS/f"{stage}.json"; d=json.loads(p.read_text())
    if d.get("status")!="complete": raise RuntimeError(f"{stage} checkpoint incomplete")
    for path,digest in d.get("checksums",{}).items():
        if checksum(Path(path))!=digest: raise RuntimeError(f"checkpoint checksum mismatch: {path}")
    return d

def profile_capture(code,histories):
    def fn(frame,event,arg):
        if event=="return" and frame.f_code is code and "Esave" in frame.f_locals:
            histories.append(np.asarray([[float(finite("energy",v)) for v in row] for row in frame.f_locals["Esave"]]))
        return fn
    return fn

def registration():
    start=time.monotonic(); load_checkpoint("atlas-free"); em,rows,samples,axes,observed=load_context(); REG_DIR.mkdir(parents=True,exist_ok=True)
    prov=json.loads(MRI_PROV.read_text()); native=load_pinned_mri_image(em,mri_path=MRI,provenance=prov)
    xI,I=em.downsample_image_domain(native.x,native.data,[4,4,4]); I=np.asarray(I,dtype=np.float32); xI=[np.asarray(v) for v in xI]
    if I.shape[1:]!=(237,284,254): raise RuntimeError(f"downsampled MRI shape {I.shape[1:]}")
    mri=LightImage("MRI_7T_WHOLE","7T_T1",I,xI,"volume",[str(MRI)])
    rss_before=peak_rss_kib(); del native; gc.collect(); rss_after=peak_rss_kib()
    J,W=stream_stack(samples,observed,2846,em=em,spatial_axes=[axes[1],axes[2]]); xJ=[np.asarray(axes[0]),block_axis(axes[1],4),block_axis(axes[2],4)]
    assert J.shape==(3,2846,130,182); assert W.shape==(2846,130,182)
    serial_pitch=float(abs(xJ[0][1]-xJ[0][0])); si=[float(abs(v[1]-v[0])) for v in xI]; sj=[serial_pitch,float(abs(xJ[1][1]-xJ[1][0])),float(abs(xJ[2][1]-xJ[2][0]))]
    if not np.allclose(si,[800]*3,atol=1e-3) or not np.allclose(sj,[serial_pitch,800,800],atol=1e-3): raise RuntimeError("spacing assertion failed")
    names=[Path(s["sample_id"]).stem for s in samples]; hist=LightImage("HIST_SYMMETRIC","HIST_NISSL",J,xJ,"slice_dataset",names)
    A2d=np.load(ATLAS_DIR/"expanded_2846_A2d.npy"); A=np.loadtxt(INITIAL_A)
    running={"stage":"registration","status":"running","shapes":{"I":list(I.shape),"J":list(J.shape),"W0":list(W.shape)},"coordinate_lengths":{"xI":list(map(len,xI)),"xJ":list(map(len,xJ))},"spacings_um":{"xI":si,"xJ":sj},"pre_downsample":{"MRI":[4,4,4],"histology":[1,4,4]},"optimizer_downI":[[2,2,2],[1,1,1]],"optimizer_downJ":[[1,2,2],[1,1,1]],"effective_spacings_um":{"level1_I":[1600]*3,"level2_I":[800]*3,"level1_J":[serial_pitch,1600,1600],"level2_J":sj},"rss_before_native_mri_release_kib":rss_before,"rss_after_native_mri_release_kib":rss_after,"start_time":now()}
    atomic_json(CHECKPOINTS/"registration.json",running); print(json.dumps(running,indent=2),flush=True)
    cfg=dict(I=I,xI=[xI],J=J,xJ=[xJ],W0=W,A=A,A2d=A2d,downI=[[2,2,2],[1,1,1]],downJ=[[1,2,2],[1,1,1]],dv=[2000.0],a=[4000.0],n_iter=[50,50],slice_matching=True,slice_matching_start=[49,0],v_start=[50,25],Amode=2,rigid_procrustes=True,eA=[1e6],eA2d=[1e5],ev=[1e-2],sigmaR=[1e4],order=1,local_contrast=[[]],full_outputs=False,n_draw=0,dtype=torch.float32,device="cpu")
    histories=[]; sys.setprofile(profile_capture(em.emlddmm.__code__,histories))
    try: outs=em.emlddmm_multiscale(**cfg)
    finally: sys.setprofile(None); plt.close("all")
    if len(outs)!=2 or len(histories)!=2: raise RuntimeError("missing multiscale outputs/histories")
    final=outs[-1]; em.write_transform_outputs(str(REG_DIR),final,mri,hist)
    Aout=finite("A",final["A"]); P=finite("A2d",final["A2d"]); v=finite("v",final["v"]); xv=[finite(f"xv{i}",q) for i,q in enumerate(final["xv"])]
    numerical=REG_DIR/"full_coarse_numerical_outputs.npz"; np.savez_compressed(numerical,A=Aout,A2d=P,v=v,xv0=xv[0],xv1=xv[1],xv2=xv[2],xI0=xI[0],xI1=xI[1],xI2=xI[2],xJ0=xJ[0],xJ1=xJ[1],xJ2=xJ[2],observed=observed)
    energy=[]
    for level,h in enumerate(histories,1): p=REG_DIR/f"raw_Esave_level-{level}.npy"; np.save(p,h); energy.append(p)
    elapsed=time.monotonic()-start; done={**running,"status":"complete","elapsed_seconds":elapsed,"peak_process_group_rss_kib":peak_rss_kib(),"outputs":{"numerical":str(numerical),"raw_Esave":[str(p) for p in energy],"transform_root":str(REG_DIR)},"checksums":{str(p):checksum(p) for p in [numerical,*energy]}}
    atomic_json(CHECKPOINTS/"registration.json",done); print(json.dumps(done,indent=2),flush=True)

def geometry_audit(source,nifti):
    shape=tuple(nifti.shape[:3]); affine=np.asarray(nifti.affine); points=[(i,j,k) for i in (0,shape[0]-1) for j in (0,shape[1]-1) for k in (0,shape[2]-1)]; points.append(tuple((np.asarray(shape)-1)//2)); errors=[]
    if source.data.shape[1:]!=shape: raise RuntimeError("MRI array order mismatch")
    for p in points: errors.append(float(np.max(np.abs((affine@[*p,1])[:3]*1000-np.asarray([source.x[a][p[a]] for a in range(3)])))))
    if max(errors)>1e-3: raise RuntimeError("MRI affine/loader mismatch")
    return {"points_checked":9,"max_abs_error_um":max(errors),"shape":list(shape)}

def jacobian_chunks(em,xv,v,chunk=24):
    xt=[torch.as_tensor(x,dtype=torch.float32) for x in xv]; phi=em.v_to_phii(xt,-torch.as_tensor(v,dtype=torch.float32).flip(0)).cpu().numpy(); spacing=[float(x[1]-x[0]) for x in xv]
    minimum=math.inf; maximum=-math.inf; nonpositive=0; count=0
    for start in range(0,phi.shape[1],chunk):
        lo=max(0,start-1); hi=min(phi.shape[1],start+chunk+1); block=phi[:,lo:hi]
        derivatives=[]
        for component in range(3): derivatives.append(np.gradient(block[component],*spacing,edge_order=1))
        interior=slice(start-lo,min(start+chunk,phi.shape[1])-lo)
        a,b,c=derivatives[0][0][interior],derivatives[0][1][interior],derivatives[0][2][interior]
        d,e,f=derivatives[1][0][interior],derivatives[1][1][interior],derivatives[1][2][interior]
        g,h,i=derivatives[2][0][interior],derivatives[2][1][interior],derivatives[2][2][interior]
        det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        if not np.all(np.isfinite(det)): raise RuntimeError("nonfinite Jacobian")
        minimum=min(minimum,float(det.min())); maximum=max(maximum,float(det.max())); nonpositive+=int(np.count_nonzero(det<=0)); count+=det.size
        del derivatives,det
    return phi,{"min":minimum,"max":maximum,"nonpositive":nonpositive,"count":count}

def postprocess():
    start=time.monotonic(); load_checkpoint("atlas-free"); load_checkpoint("registration"); em,rows,samples,axes,observed=load_context(); POST_DIR.mkdir(parents=True,exist_ok=True)
    z=np.load(REG_DIR/"full_coarse_numerical_outputs.npz"); A=z["A"]; P=z["A2d"]; v=z["v"]; xv=[z[f"xv{i}"] for i in range(3)]
    prov=json.loads(MRI_PROV.read_text()); source=load_pinned_mri_image(em,mri_path=MRI,provenance=prov); ni=nib.load(str(MRI)); geometry=geometry_audit(source,ni)
    B=P[np.setdiff1d(np.arange(2846),observed)[0]]; ranges=residuals(POST_DIR/"observed_residual_transforms.tsv",rows,observed,P,B); residual_plot(POST_DIR/"observed_residual_transform_summary.png",axes,observed,P,B)
    # Compact QC and support-aware full reconstruction reuse streamed 800-um histology.
    J,W=stream_stack(samples,observed,2846,em=em,spatial_axes=[axes[1],axes[2]]); xJ=[axes[0],block_axis(axes[1],4),block_axis(axes[2],4)]
    bi=np.linalg.inv(B); corners=np.array([[xJ[1][0],xJ[2][0],1],[xJ[1][0],xJ[2][-1],1],[xJ[1][-1],xJ[2][0],1],[xJ[1][-1],xJ[2][-1],1]]).T; rc=bi@corners
    yr=np.linspace(rc[0].min(),rc[0].max(),130); xr=np.linspace(rc[1].min(),rc[1].max(),182); RR,CC=np.meshgrid(yr,xr,indexing="ij")
    num=np.memmap(RUN_TMP/"reg_num",mode="w+",dtype=np.float32,shape=J.shape); sup=np.memmap(RUN_TMP/"reg_sup",mode="w+",dtype=np.float32,shape=W.shape); num[:]=0; sup[:]=0
    for idx in observed:
        qy=P[idx,0,0]*RR+P[idx,0,1]*CC+P[idx,0,2]; qx=P[idx,1,0]*RR+P[idx,1,1]*CC+P[idx,1,2]; iy=(qy-xJ[1][0])/(xJ[1][1]-xJ[1][0]); ix=(qx-xJ[2][0])/(xJ[2][1]-xJ[2][0]); sup[idx]=ndi.map_coordinates(W[idx],[iy,ix],order=1,mode="constant",cval=0,prefilter=False)
        for c in range(3): num[c,idx]=ndi.map_coordinates(J[c,idx]*W[idx],[iy,ix],order=1,mode="constant",cval=0,prefilter=False)
    phi,jac=jacobian_chunks(em,xv,v); XV=np.stack(np.meshgrid(*xv,indexing="ij")); disp=phi-XV; shape=tuple(source.data.shape[1:]); recon=np.memmap(RUN_TMP/"recon",mode="w+",dtype=np.float32,shape=(*shape,3)); frac=np.memmap(RUN_TMP/"support",mode="w+",dtype=np.float32,shape=shape); support_min=math.inf; support_max=-math.inf; support_nonempty=False; zero_violation=False
    for start1 in range(0,shape[1],8):
        stop=min(start1+8,shape[1]); X=np.stack(np.meshgrid(source.x[0],source.x[1][start1:stop],source.x[2],indexing="ij")); iv=[(X[k]-xv[k][0])/(xv[k][1]-xv[k][0]) for k in range(3)]; flow=X+np.stack([ndi.map_coordinates(disp[k],iv,order=1,mode="nearest",prefilter=False) for k in range(3)]); q=(A[:3,:3]@flow.reshape(3,-1)+A[:3,3,None]).reshape(flow.shape); ih=[(q[0]-xJ[0][0])/(xJ[0][1]-xJ[0][0]),(q[1]-yr[0])/(yr[1]-yr[0]),(q[2]-xr[0])/(xr[1]-xr[0])]; s=np.clip(ndi.map_coordinates(sup,ih,order=1,mode="constant",cval=0,prefilter=False),0,1); out=np.zeros((*s.shape,3),np.float32); positive=s>0
        for c in range(3): sampled=ndi.map_coordinates(num[c],ih,order=1,mode="constant",cval=0,prefilter=False); out[...,c][positive]=sampled[positive]/s[positive]
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(out)): raise RuntimeError("nonfinite reconstruction slab")
        support_min=min(support_min,float(s.min())); support_max=max(support_max,float(s.max())); support_nonempty=support_nonempty or bool(np.any(positive)); zero_violation=zero_violation or bool(np.any(out[~positive]!=0))
        recon[:,start1:stop]=out; frac[:,start1:stop]=s; recon.flush(); frac.flush(); print(f"postprocess slab {start1}:{stop}",flush=True)
    rh=ni.header.copy(); rh.set_data_dtype(np.float32); sh=ni.header.copy(); sh.set_data_dtype(np.float32); rp=OUTPUT/"nissl_support_weighted_reconstruction_on_mri_grid.nii"; sp=OUTPUT/"nissl_fractional_support_on_mri_grid.nii"; nib.save(nib.Nifti1Image(recon,ni.affine,header=rh),str(rp)); nib.save(nib.Nifti1Image(frac,ni.affine,header=sh),str(sp))
    compact_qc=compact_post_qc(source,recon,frac,axes,observed,A)
    if not support_nonempty or support_min<0 or support_max>1+1e-6: raise RuntimeError("invalid fractional support volume")
    if zero_violation: raise RuntimeError("nonzero reconstruction outside support")
    deformation={"velocity_max_um":float(np.linalg.norm(v,axis=1).max()),"jacobian":jac}; atomic_json(POST_DIR/"velocity_jacobian_summary.json",deformation)
    done={"stage":"postprocess","status":"review_required","elapsed_seconds":time.monotonic()-start,"peak_process_group_rss_kib":peak_rss_kib(),"geometry":geometry,"residual_ranges":ranges,"reconstruction":{"intensity":str(rp),"fractional_support":str(sp)},"deformation":deformation,"compact_qc":str(compact_qc),"production_refinement_launched":False}; atomic_json(CHECKPOINTS/"postprocess.json",done); atomic_json(OUTPUT/"full_coarse_summary.json",done); print(json.dumps(done,indent=2),flush=True)

def residual_plot(path: Path, axes, observed, mats, baseline):
    r=np.linalg.inv(baseline)[None]@mats[observed]; serial=np.asarray(axes[0])[observed]
    values=(r[:,0,2],r[:,1,2],np.degrees(np.arctan2(r[:,1,0],r[:,0,0])))
    labels=("row translation (um)","column translation (um)","rotation (degrees)")
    fig,panels=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for panel,value,label in zip(panels,values,labels): panel.plot(serial,value,lw=.7); panel.set_ylabel(label)
    panels[-1].set_xlabel("original serial coordinate (um)"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

def compact_post_qc(source,recon,fraction,axes,observed,A):
    reps=observed[np.linspace(0,len(observed)-1,9).round().astype(int)]; inv=np.linalg.inv(A)
    fig,p=plt.subplots(3,len(reps),figsize=(27,9),facecolor="white")
    for col,index in enumerate(reps):
        point=inv@np.asarray([float(axes[0][index]),0,0,1]); plane=int(np.argmin(np.abs(np.asarray(source.x[1])-point[1])))
        m=np.asarray(source.data[0,::4,plane,::4],np.float32); r=np.asarray(recon[::4,plane,::4],np.float32); s=np.asarray(fraction[::4,plane,::4],np.float32)
        lo,hi=np.percentile(m,[1,99]); md=np.clip((m-lo)/(hi-lo+1e-8),0,1); overlay=.5*np.clip(r,0,1)+.5*md[...,None]
        p[0,col].imshow(md,cmap="gray",origin="lower"); p[1,col].imshow(np.clip(r,0,1),origin="lower"); p[2,col].imshow(np.clip(overlay,0,1),origin="lower")
        if np.any(s>.01): p[1,col].contour(s,levels=[.01],colors="red",linewidths=.3)
        p[0,col].set_title(str(index)); [p[row,col].axis("off") for row in range(3)]
    p[0,0].set_ylabel("MRI"); p[1,0].set_ylabel("support-weighted Nissl"); p[2,0].set_ylabel("overlay")
    fig.tight_layout(); path=POST_DIR/"full_extent_compact_mri_nissl_qc.png"; fig.savefig(path,dpi=150); plt.close(fig)
    for level in (1,2):
        history=np.load(REG_DIR/f"raw_Esave_level-{level}.npy"); fig,ax=plt.subplots(figsize=(8,4))
        for column,label in enumerate(("E","matching","regularization")[:history.shape[1]]): ax.plot(history[:,column],label=label)
        ax.set(xlabel="iteration",title=f"Raw Esave level {level}"); ax.legend(); fig.tight_layout(); fig.savefig(POST_DIR/f"objective_history_level-{level}.png",dpi=150); plt.close(fig)
    return path


def _warp_saved_section(image, support, transform, row_um, column_um):
    """Sample one prepared section into its baseline-factored registered frame."""
    rr, cc = np.meshgrid(row_um, column_um, indexing="ij")
    source_row = (
        transform[0, 0] * rr + transform[0, 1] * cc + transform[0, 2]
    )
    source_column = (
        transform[1, 0] * rr + transform[1, 1] * cc + transform[1, 2]
    )
    iy = (source_row - row_um[0]) / (row_um[1] - row_um[0])
    ix = (source_column - column_um[0]) / (
        column_um[1] - column_um[0]
    )
    transformed_support = ndi.map_coordinates(
        support, [iy, ix], order=1, mode="constant", cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    transformed = np.zeros_like(image, dtype=np.float32)
    positive = transformed_support > 0.0
    for channel in range(3):
        numerator = ndi.map_coordinates(
            image[channel] * support,
            [iy, ix],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        transformed[channel, positive] = (
            numerator[positive] / transformed_support[positive]
        )
    return transformed, transformed_support


def _publish_saved_overview(source: Path, destination: Path) -> None:
    """Publish one completed file atomically, including across filesystems."""
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite overview output: {destination}")
    if os.stat(source.parent).st_dev == os.stat(destination.parent).st_dev:
        os.replace(source, destination)
        return
    local = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    if local.exists():
        raise RuntimeError(f"Destination-local temporary file exists: {local}")
    try:
        with source.open("rb") as incoming, local.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(local, destination)
    finally:
        if local.exists():
            local.unlink()


def _saved_atlas_free_stack_overview() -> dict[str, Any]:
    """Render saved-transform atlas-free QC without invoking a pipeline stage."""
    png = ATLAS_DIR / "atlas_free_transformed_stack_overview.png"
    report_path = ATLAS_DIR / "atlas_free_transformed_stack_overview.json"
    if png.exists() or report_path.exists():
        raise RuntimeError("Saved-transform overview output already exists")
    checkpoint_path = CHECKPOINTS / "atlas-free.json"
    registration_checkpoint = CHECKPOINTS / "registration.json"
    sources = {
        "observed_A2d": ATLAS_DIR / "observed_A2d.npy",
        "bookkeeping_frame": ATLAS_DIR / "common_bookkeeping_frame.txt",
        "observed_indices": ATLAS_DIR / "observed_physical_indices.npy",
        "atlas_free_checkpoint": checkpoint_path,
    }
    registration_before = registration_checkpoint.exists()
    before_hashes = {name: checksum(path) for name, path in sources.items()}
    checkpoint = load_checkpoint("atlas-free")
    if checkpoint.get("status") != "complete":
        raise RuntimeError("Atlas-free checkpoint is not complete")

    em, rows, samples, axes, expected_observed = load_context()
    observed = np.load(sources["observed_indices"]).astype(np.int64)
    matrices = finite("observed A2d", np.load(sources["observed_A2d"]))
    baseline = finite("bookkeeping frame", np.loadtxt(sources["bookkeeping_frame"]))
    if matrices.shape != (641, 3, 3) or observed.shape != (641,):
        raise RuntimeError("Unexpected saved atlas-free transform dimensions")
    if not np.array_equal(observed, expected_observed):
        raise RuntimeError("Saved observed indices do not match the Nissl manifest")
    if np.unique(observed).size != 641 or observed.min() < 0 or observed.max() >= 2846:
        raise RuntimeError("Saved observed indices are invalid")

    residual = np.linalg.inv(baseline)[None] @ matrices
    recomposition_error = float(
        np.max(np.abs(baseline[None] @ residual - matrices))
    )
    if recomposition_error > 1e-8:
        raise RuntimeError("Baseline-factored transforms do not recompose")
    row_translation = residual[:, 0, 2]
    column_translation = residual[:, 1, 2]
    rotation = np.degrees(
        np.arctan2(residual[:, 1, 0], residual[:, 0, 0])
    )
    serial = np.asarray(axes[0], dtype=np.float64)
    observed_serial = serial[observed]
    row_um = block_axis(axes[1], 4)
    column_um = block_axis(axes[2], 4)
    representative_positions = _uniform_representative_positions(641, 9)
    representative_lookup = {
        int(position): slot
        for slot, position in enumerate(representative_positions)
    }
    representative_original = [None] * len(representative_positions)
    representative_transformed = [None] * len(representative_positions)

    original_serial_row_num = np.zeros((3, 2846, 130), np.float32)
    transformed_serial_row_num = np.zeros((3, 2846, 130), np.float32)
    original_serial_column_num = np.zeros((3, 2846, 182), np.float32)
    transformed_serial_column_num = np.zeros((3, 2846, 182), np.float32)
    original_serial_row_den = np.zeros((2846, 130), np.float32)
    transformed_serial_row_den = np.zeros((2846, 130), np.float32)
    original_serial_column_den = np.zeros((2846, 182), np.float32)
    transformed_serial_column_den = np.zeros((2846, 182), np.float32)

    for order, physical_index in enumerate(observed):
        image, support = read_section(
            samples[physical_index], em, [axes[1], axes[2]]
        )
        image, support = downsample_section(image, support)
        if image.shape != (3, 130, 182) or support.shape != (130, 182):
            raise RuntimeError(
                f"Unexpected streamed section shape at {physical_index}"
            )
        transformed, transformed_support = _warp_saved_section(
            image, support, residual[order], row_um, column_um
        )
        weighted = image * support[None]
        transformed_weighted = transformed * transformed_support[None]
        original_serial_row_num[:, physical_index] = weighted.sum(axis=2)
        transformed_serial_row_num[:, physical_index] = (
            transformed_weighted.sum(axis=2)
        )
        original_serial_column_num[:, physical_index] = weighted.sum(axis=1)
        transformed_serial_column_num[:, physical_index] = (
            transformed_weighted.sum(axis=1)
        )
        original_serial_row_den[physical_index] = support.sum(axis=1)
        transformed_serial_row_den[physical_index] = (
            transformed_support.sum(axis=1)
        )
        original_serial_column_den[physical_index] = support.sum(axis=0)
        transformed_serial_column_den[physical_index] = (
            transformed_support.sum(axis=0)
        )
        if order in representative_lookup:
            slot = representative_lookup[order]
            representative_original[slot] = image.copy()
            representative_transformed[slot] = transformed.copy()
        if (order + 1) % 50 == 0:
            print(f"overview streamed {order + 1}/641", flush=True)

    if any(image is None for image in representative_original):
        raise RuntimeError("A representative original section was not retained")
    if any(image is None for image in representative_transformed):
        raise RuntimeError("A representative transformed section was not retained")
    expected_shapes = {
        "original_serial_row_num": (3, 2846, 130),
        "transformed_serial_row_num": (3, 2846, 130),
        "original_serial_column_num": (3, 2846, 182),
        "transformed_serial_column_num": (3, 2846, 182),
        "original_serial_row_den": (2846, 130),
        "transformed_serial_row_den": (2846, 130),
        "original_serial_column_den": (2846, 182),
        "transformed_serial_column_den": (2846, 182),
    }
    for name, expected in expected_shapes.items():
        if locals()[name].shape != expected:
            raise RuntimeError(
                f"{name} has shape {locals()[name].shape}, expected {expected}"
            )

    jump_order = np.argsort(np.abs(np.diff(row_translation)))[-3:][::-1]
    jump_records = []
    marked_orders = set()
    for left in jump_order:
        right = int(left + 1)
        marked_orders.update((int(left), right))
        jump_records.append(
            {
                "left_allen_section": int(
                    rows[observed[left]]["allen_section_number"]
                ),
                "right_allen_section": int(
                    rows[observed[right]]["allen_section_number"]
                ),
                "serial_gap_um": float(
                    observed_serial[right] - observed_serial[left]
                ),
                "absolute_row_translation_jump_um": float(
                    abs(row_translation[right] - row_translation[left])
                ),
            }
        )
    allen_to_order = {
        int(rows[index]["allen_section_number"]): order
        for order, index in enumerate(observed)
    }
    for allen_section in (1082, 1089):
        if allen_section not in allen_to_order:
            raise RuntimeError(f"Required marked Allen section {allen_section} is absent")
        marked_orders.add(allen_to_order[allen_section])
    marked_serial = {
        str(int(rows[observed[order]]["allen_section_number"])): float(
            observed_serial[order]
        )
        for order in sorted(marked_orders)
    }

    representatives = []
    titles = []
    for position in representative_positions:
        physical_index = int(observed[position])
        allen = int(rows[physical_index]["allen_section_number"])
        z_um = float(serial[physical_index])
        titles.append(f"Allen {allen}, z={z_um / 1000.0:.3f} mm")
        representatives.append(
            {
                "observed_order": int(position),
                "physical_index": physical_index,
                "allen_section": allen,
                "serial_um": z_um,
            }
        )
    projections = {
        "original_serial_row": (
            original_serial_row_num, original_serial_row_den, row_um
        ),
        "transformed_serial_row": (
            transformed_serial_row_num, transformed_serial_row_den, row_um
        ),
        "original_serial_column": (
            original_serial_column_num, original_serial_column_den, column_um
        ),
        "transformed_serial_column": (
            transformed_serial_column_num,
            transformed_serial_column_den,
            column_um,
        ),
    }
    traces = {
        "row_translation_um": row_translation,
        "column_translation_um": column_translation,
        "rotation_deg": rotation,
    }

    run_tmp = Path(os.environ["TMPDIR"]).resolve()
    approved_tmp = Path(
        "/cis/home/dpadova/.cache/ad-resilience/tmp"
    ).resolve()
    if run_tmp != approved_tmp and approved_tmp not in run_tmp.parents:
        raise RuntimeError("Overview TMPDIR is outside the approved home cache")
    private = Path(
        __import__("tempfile").mkdtemp(prefix="atlas-overview-", dir=run_tmp)
    ).resolve()
    if approved_tmp not in private.parents:
        raise RuntimeError("Resolved overview temporary directory escaped cache")
    tmp_png = private / png.name
    tmp_json = private / report_path.name
    try:
        _, projection_shapes = _render_saved_transform_stack_overview(
            representative_original=representative_original,
            representative_transformed=representative_transformed,
            representative_titles=titles,
            projections=projections,
            serial_um=serial,
            trace_serial_um=observed_serial,
            traces=traces,
            marked_serial_um=marked_serial,
            output=tmp_png,
        )
        after_hashes = {
            name: checksum(path) for name, path in sources.items()
        }
        registration_after = registration_checkpoint.exists()
        source_unchanged = all(
            before_hashes[name] == after_hashes[name]
            for name in ("observed_A2d", "bookkeeping_frame", "observed_indices")
        )
        checkpoint_unchanged = (
            before_hashes["atlas_free_checkpoint"]
            == after_hashes["atlas_free_checkpoint"]
        )
        if not source_unchanged or not checkpoint_unchanged:
            raise RuntimeError("Authoritative atlas-free inputs changed during QC")
        serial_edges = _coordinate_cell_edges(serial)
        row_edges = _coordinate_cell_edges(row_um)
        column_edges = _coordinate_cell_edges(column_um)
        report = {
            "created_at": now(),
            "outputs": {"png": str(png), "json": str(report_path)},
            "source_paths": {name: str(path) for name, path in sources.items()},
            "source_checksums_before": before_hashes,
            "source_checksums_after": after_hashes,
            "representative_sections": representatives,
            "projection_numerator_shapes": {
                key: list(expected_shapes[key])
                for key in expected_shapes if key.endswith("_num")
            },
            "projection_denominator_shapes": {
                key: list(expected_shapes[key])
                for key in expected_shapes if key.endswith("_den")
            },
            "normalized_projection_shapes": projection_shapes,
            "projection_orientation": {
                "storage": "(serial, spatial-coordinate), RGB channel first for numerators",
                "display": "projection.T",
                "horizontal_axis": "anterior-to-posterior serial position",
                "vertical_axes": ["row position", "column position"],
                "zero_support": "masked neutral background",
            },
            "physical_extents_um": {
                "serial_cell_edges": [
                    float(serial_edges[0]), float(serial_edges[-1])
                ],
                "row_cell_edges": [
                    float(row_edges[0]), float(row_edges[-1])
                ],
                "column_cell_edges": [
                    float(column_edges[0]), float(column_edges[-1])
                ],
            },
            "marked_allen_sections": sorted(
                int(value) for value in marked_serial
            ),
            "largest_adjacent_row_translation_jumps": jump_records,
            "residual_ranges": {
                "row_translation_um": [
                    float(row_translation.min()), float(row_translation.max())
                ],
                "column_translation_um": [
                    float(column_translation.min()),
                    float(column_translation.max()),
                ],
                "rotation_deg": [
                    float(rotation.min()), float(rotation.max())
                ],
            },
            "baseline_recomposition_max_abs_error": recomposition_error,
            "optimization_invoked_by_helper": False,
            "registration_invoked_by_helper": False,
            "registration_checkpoint_present_before": registration_before,
            "registration_checkpoint_present_after": registration_after,
            "source_transforms_unchanged": source_unchanged,
            "atlas_free_checkpoint_unchanged": checkpoint_unchanged,
        }
        with tmp_json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_saved_overview(tmp_png, png)
        try:
            _publish_saved_overview(tmp_json, report_path)
        except BaseException:
            if png.exists():
                png.unlink()
            raise
    finally:
        shutil.rmtree(private)
        plt.close("all")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS","8"))); torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["atlas-free","registration","postprocess"]); args=parser.parse_args()
    CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    try: {"atlas-free":atlas_free,"registration":registration,"postprocess":postprocess}[args.stage]()
    except BaseException as exc:
        atomic_json(CHECKPOINTS/f"{args.stage}.json",{"stage":args.stage,"status":"failed","failure":repr(exc),"failure_time":now(),"peak_process_group_rss_kib":peak_rss_kib(),"production_refinement_launched":False}); raise
if __name__=="__main__": main()
