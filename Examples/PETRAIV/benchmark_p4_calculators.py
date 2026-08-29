#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import at
from at import get_refpts
from pySC import SimulatedCommissioning

import improve_pyLOCO_latest as drv
from pyLOCO.pyloco import get_fit_param_block
from set_correction import set_correction
from analyze_ring import analyze_ring

CONFIGS = [
    ("A", "Linear", "Numerical"),
    ("B", "Linear", "Analytical"),
    ("C", "Analytical", "Numerical"),
    ("D", "Analytical", "Analytical"),
]
TRACKING_CONFIGS = [
    ("E", "Tracking", "Numerical"),
    ("F", "Tracking", "Analytical"),
]

def jdefault(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    raise TypeError(type(x).__name__)

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=jdefault))

def last_finite(x):
    a=np.asarray(x,dtype=float).ravel()
    a=a[np.isfinite(a)]
    return float(a[-1]) if len(a) else math.nan

def rel_norm(a,b):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel()
    d=np.linalg.norm(b)
    return float(np.linalg.norm(a-b)/(d if d else 1.0))

def read_metrics(path):
    path=Path(path)
    if not path.exists(): return []
    with path.open(newline="") as f:
        out=[]
        for r in csv.DictReader(f):
            rr={}
            for k,v in r.items():
                try: rr[k]=float(v)
                except Exception: rr[k]=v
            out.append(rr)
        return out

def sum_metric(rows, names):
    if not rows: return math.nan
    keys=set().union(*(r.keys() for r in rows))
    for n in names:
        hits=[n] if n in keys else [k for k in keys if n.lower() in k.lower()]
        if hits:
            vals=[r.get(hits[0]) for r in rows]
            vals=[float(v) for v in vals if isinstance(v,(int,float)) and np.isfinite(v)]
            if vals: return float(np.sum(vals))
    return math.nan

def machine_selection(seed_file, m):
    sc=SimulatedCommissioning.from_json(seed_file, lattice_file=str(drv.LATTICE_FILE))
    sc.lattice.ring.enable_6d(); sc.lattice.design.enable_6d()
    bpms=np.asarray(m["used_bpms_ords"],int)
    hc=np.asarray(m["hcor_inds"],int).tolist()
    vc=np.asarray(m["vcor_inds"],int).tolist()
    q=np.asarray(get_refpts(sc.lattice.ring, at.elements.Quadrupole),int)
    names=(sc.magnet_arrays["cysf"]+sc.magnet_arrays["cxysf"]+
           sc.magnet_arrays["cxsf"]+sc.magnet_arrays["cxysf2"])
    sq=np.asarray(sorted(sc.magnet_settings.magnets[n].sim_index for n in names),int)
    cav=np.asarray(get_refpts(sc.lattice.ring, at.elements.RFCavity),int)
    ids=np.sort(np.concatenate([
        get_refpts(sc.lattice.ring,"BPM_09_DW*"),
        get_refpts(sc.lattice.ring,"BPM_01_DW*")
    ])).astype(int)
    cm=[np.asarray(m["CMstep_h"],float),np.asarray(m["CMstep_v"],float)]
    return sc,bpms,[hc,vc],q,sq,cav,ids,cm

def optics_flat(d,prefix):
    def get(path):
        cur=d
        try:
            for p in path: cur=cur[p]
            return float(cur)
        except Exception: return math.nan
    out={
        prefix+"orbit_h_um":get(["rms_elements","rms_orbit_um","horizontal"]),
        prefix+"orbit_v_um":get(["rms_elements","rms_orbit_um","vertical"]),
        prefix+"beta_h_pct":get(["rms_elements","rms_beta_beating_percent","horizontal"]),
        prefix+"beta_v_pct":get(["rms_elements","rms_beta_beating_percent","vertical"]),
        prefix+"disp_h_mm":get(["rms_elements","rms_dispersion_err_mm","horizontal"]),
        prefix+"disp_v_mm":get(["rms_elements","rms_dispersion_err_mm","vertical"]),
        prefix+"emit_h_pm":get(["with_errors","emittance_pm","horizontal"]),
        prefix+"emit_v_pm":get(["with_errors","emittance_pm","vertical"]),
    }
    for key,base in [("Q","tune"),("chrom_","chromaticity")]:
        try:
            arr=np.asarray(d["with_errors"][base],float).ravel()
        except Exception:
            arr=np.array([])
        names=("x","y","z")
        for i,n in enumerate(names):
            out[prefix+key+n]=float(arr[i]) if i<len(arr) else math.nan
    return out

def write_csv(path, rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def run_cfg(label,orm,jac,seed_file,m,out_root,n1,n2):
    out=Path(out_root)/f"{label}_{orm}_ORM_{jac}_J"; out.mkdir(parents=True,exist_ok=True)
    sc,bpms,cors,q,sq,cav,ids,cm=machine_selection(seed_file,m)
    stage1_orm,stage2_orm=drv.benchmark_stage_response_matrix_calculators(orm)
    drv.STAGE1_RESPONSE_MATRIX_CALCULATOR=stage1_orm
    drv.STAGE2_RESPONSE_MATRIX_CALCULATOR=stage2_orm
    drv.QUAD_JACOBIAN_CALCULATOR=jac
    drv.SKEW_JACOBIAN_CALCULATOR=jac
    drv.STAGE1_NITER=n1; drv.STAGE2_NITER=n2
    drv.FORCE_RECOMPUTE_JACOBIANS=True
    drv.SAVE_JACOBIANS=True

    t0=time.perf_counter()
    result=drv.run_latest_pyloco_two_stage(
        model_ring=sc.lattice.design, CMstep=cm, CAVords=cav,
        quad_indices=q, skew_quad_indices=sq,
        used_cor_ords=cors, used_bpms_ords=bpms,
        measured_orm=np.asarray(m["measured_orm"],float),
        sigma_w=np.asarray(m["sigma_w"],float),
        measured_eta_x=np.asarray(m["measured_eta_x"],float),
        measured_eta_y=np.asarray(m["measured_eta_y"],float),
        output_dir=out/"pyloco",
        stage1_response_matrix_calculator=stage1_orm,
        stage2_response_matrix_calculator=stage2_orm,
    )
    fit_results,fit_dict,ring_fit,fitted_orm,c_bpms_after,details=result
    fit_wall=time.perf_counter()-t0

    fq=np.asarray(get_fit_param_block(fit_dict,"quads"),float).ravel()
    dq=fq-np.asarray([sc.lattice.design[i].K for i in q],float)
    fs=np.asarray(get_fit_param_block(fit_dict,"skew_quads"),float).ravel()
    ds=fs-np.asarray([drv.current_skew_strength(sc.lattice.design[i]) for i in sq],float)

    sc2,bpms2,_,q2,sq2,_,ids2,_=machine_selection(seed_file,m)
    set_correction(sc2,-dq,q2,individuals=True,skewness=False)
    sc2.lattice.ring.enable_6d(); sc2.lattice.design.enable_6d()
    oq=analyze_ring(sc2,elements_indices=bpms2,special_elements=ids2,useIdealRing=False,makeplot=False,return_dict=True)
    set_correction(sc2,-ds,sq2,individuals=True,skewness=True)
    sc2.lattice.ring.enable_6d(); sc2.lattice.design.enable_6d()
    osk=analyze_ring(sc2,elements_indices=bpms2,special_elements=ids2,useIdealRing=False,makeplot=False,return_dict=True)

    m1=read_metrics(out/"pyloco"/"stage_1_normal"/"iteration_metrics.csv")
    m2=read_metrics(out/"pyloco"/"stage_2_coupling"/"iteration_metrics.csv")
    row={
        "label":label,"orm_calculator":orm,
        "stage1_orm_calculator":stage1_orm,"stage2_orm_calculator":stage2_orm,
        "jacobian_calculator":jac,
        "stage1_final_chi2":last_finite(details["stage1"]["chi2_history"]),
        "stage2_final_chi2":last_finite(details["stage2"]["chi2_history"]),
        "fit_wall_s":fit_wall,
        "stage1_jacobian_time_s":sum_metric(m1,["jacobian_time_s","jacobian_time"]),
        "stage2_jacobian_time_s":sum_metric(m2,["jacobian_time_s","jacobian_time"]),
        "stage1_model_orm_time_s":sum_metric(m1,["model_orm_time_s","model_orm_time"]),
        "stage2_model_orm_time_s":sum_metric(m2,["model_orm_time_s","model_orm_time"]),
        "delta_q_rms":float(np.sqrt(np.mean(dq**2))),
        "delta_q_maxabs":float(np.max(np.abs(dq))),
        "delta_skew_rms":float(np.sqrt(np.mean(ds**2))),
        "delta_skew_maxabs":float(np.max(np.abs(ds))),
    }
    row.update(optics_flat(oq,"after_quad_"))
    row.update(optics_flat(osk,"after_skew_"))
    np.savez_compressed(out/"benchmark_arrays.npz",delta_q=dq,delta_skew=ds,fitted_orm=np.asarray(fitted_orm))
    save_json(out/"benchmark_summary.json",row)
    return row,dq,ds

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed-file",required=True,type=Path)
    p.add_argument("--measured",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--include-tracking",action="store_true")
    p.add_argument("--stage1-iter",type=int,default=4)
    p.add_argument("--stage2-iter",type=int,default=4)
    p.add_argument("--only",nargs="*")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)

    with np.load(a.measured,allow_pickle=True) as z:
        m={k:z[k] for k in z.files}

    cfgs=CONFIGS+(TRACKING_CONFIGS if a.include_tracking else [])
    if a.only: cfgs=[c for c in cfgs if c[0] in set(a.only)]

    rows=[]; arr={}
    for label,orm,jac in cfgs:
        print(f"\n### {label}: ORM={orm}, Jacobian={jac}")
        r,dq,ds=run_cfg(label,orm,jac,a.seed_file,m,a.output,a.stage1_iter,a.stage2_iter)
        rows.append(r); arr[label]=(dq,ds)
        write_csv(a.output/"calculator_benchmark_raw.csv",rows)

    ref="A" if "A" in arr else rows[0]["label"]
    rr=next(r for r in rows if r["label"]==ref)
    comp=[]
    for r in rows:
        x=dict(r); lab=r["label"]
        x["reference_label"]=ref
        x["speedup_fit_vs_ref"]=rr["fit_wall_s"]/r["fit_wall_s"]
        x["delta_q_relative_norm_error_vs_ref"]=rel_norm(arr[lab][0],arr[ref][0])
        x["delta_skew_relative_norm_error_vs_ref"]=rel_norm(arr[lab][1],arr[ref][1])
        comp.append(x)
    write_csv(a.output/"calculator_benchmark_comparison.csv",comp)
    save_json(a.output/"calculator_benchmark_comparison.json",comp)

    print("\nCfg ORM         Jacobian     Fit[h]  Speedup  dQ rel.err   dSkew rel.err")
    for r in comp:
        print(f"{r['label']:<3} {r['orm_calculator']:<11} {r['jacobian_calculator']:<12} "
              f"{r['fit_wall_s']/3600:7.3f} {r['speedup_fit_vs_ref']:8.3f} "
              f"{r['delta_q_relative_norm_error_vs_ref']:11.3e} "
              f"{r['delta_skew_relative_norm_error_vs_ref']:13.3e}")

if __name__=="__main__":
    main()
