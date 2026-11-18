#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local throughput benchmark runner for amber-cupotential.

Compares Stock Amber24 pmemd.cuda_SPFP against Phase 1 (nonbonded), Stage 2A
(+cupo_tuples bonded), and Phase 2 Whole-Step (Prolate & Ewald) on a user-
supplied prmtop/rst7 pair and writes a local JSON + Markdown table.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional


def check_gpu_idle(gpu_id: str, min_free_mib: int = 1500) -> None:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    util_s, free_s = [x.strip() for x in out.splitlines()[0].split(",")]
    if int(free_s) < min_free_mib:
        raise RuntimeError(f"GPU {gpu_id} free VRAM {free_s} MiB < {min_free_mib} MiB")
    if int(util_s) > 15:
        time.sleep(1.0)


def make_mdin(
    title: str,
    nstlim: int,
    ntpr: int,
    cut: float,
    dt: float = 0.002,
    ntp: int = 0,
    barostat: int = 2,
    mcbarint: int = 100,
    ntt: int = 3,
    gamma_ln: float = 1.0,
    ischeme: int = 0,
    ithermostat: int = 0,
    therm_par: float = 0.0,
) -> str:
    ntb_eff = 2 if ntp > 0 else 1
    lines = [
        title,
        " &cntrl",
        "  imin=0, ntx=5, irest=1, ntxo=2, ioutfm=1,",
        "  ntc=2, ntf=2, tol=0.00001,",
        f"  nstlim={nstlim}, ntpr={ntpr}, ntwx=0, ntwr=500000,",
        f"  dt={dt}, cut={cut},",
        f"  ntb={ntb_eff}, ntp={ntp}, temp0=300.0, ig=12345,",
    ]
    if ischeme == 1:
        lines.append(f"  ntt={ntt}, ischeme=1, ithermostat={ithermostat}, therm_par={therm_par},")
    elif ntt == 3:
        lines.append(f"  ntt={ntt}, gamma_ln={gamma_ln},")
    if ntp > 0:
        lines.append(f"  pres0=1.0, barostat={barostat}, mcbarint={mcbarint},")
    lines.extend([" &end", " &ewald", "  dsum_tol=0.00001,", " &end", ""])
    return "\n".join(lines)


def run_case(
    pmemd_bin: str,
    run_dir: pathlib.Path,
    parm: pathlib.Path,
    crd: pathlib.Path,
    mdin_text: str,
    gpu_id: str,
    env_overrides: Dict[str, str],
) -> dict:
    check_gpu_idle(gpu_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mdin").write_text(mdin_text, encoding="utf-8")

    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for k, v in env_overrides.items():
        env[k] = str(v)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [pmemd_bin, "-O", "-i", "mdin", "-p", str(parm), "-c", str(crd), "-o", "mdout", "-r", "restrt", "-inf", "mdinfo"],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{run_dir.name} failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    mdout = (run_dir / "mdout").read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"Final Performance Info:\s*\n.*?\n"
        r"\|\s*Average timings for last\s+(\d+)\s+steps:\s*\n"
        r"\|\s*Elapsed\(s\)\s*=\s*([\d.]+)\s+Per Step\(ms\)\s*=\s*([\d.]+)\s*\n"
        r"\|\s*ns/day\s*=\s*([\d.]+)\s+seconds/ns\s*=\s*([\d.]+)\s*\n.*?\n"
        r"\|\s*Average timings for all steps:\s*\n"
        r"\|\s*Elapsed\(s\)\s*=\s*([\d.]+)\s+Per Step\(ms\)\s*=\s*([\d.]+)\s*\n"
        r"\|\s*ns/day\s*=\s*([\d.]+)\s+seconds/ns\s*=\s*([\d.]+)",
        mdout,
        re.S,
    )
    if not m:
        raise RuntimeError(f"Could not parse Final Performance Info in {run_dir / 'mdout'}")
    return {
        "wall_s": round(wall_s, 3),
        "last_steps": int(m.group(1)),
        "last_ms_step": float(m.group(3)),
        "last_ns_day": float(m.group(4)),
        "all_ns_day": float(m.group(8)),
    }


def main() -> int:
    default_pmemd = (
        os.path.join(os.environ["AMBERHOME"], "bin", "pmemd.cuda_SPFP")
        if os.environ.get("AMBERHOME")
        else "pmemd.cuda_SPFP"
    )
    parser = argparse.ArgumentParser(description="Benchmark Amber24 pmemd.cuda_SPFP + cuPotential locally.")
    parser.add_argument("--pmemd", default=default_pmemd, help="Path to pmemd.cuda_SPFP")
    parser.add_argument("--parm", required=True, help="Amber prmtop file")
    parser.add_argument("--crd", required=True, help="Amber equilibrated restart/crd file")
    parser.add_argument("--hmr-parm", help="Optional HMR prmtop for 4fs Middle benchmark")
    parser.add_argument("--nstlim", type=int, default=50000, help="MD steps per benchmark (default: 50000)")
    parser.add_argument("--cut", type=float, default=8.0, help="Cutoff in Angstroms (default: 8.0)")
    parser.add_argument("--gpu", default="0", help="CUDA GPU index (default: 0)")
    parser.add_argument("--work-dir", default="./bench_runs", help="Directory for temporary run outputs")
    parser.add_argument("--out-json", default="benchmark_results.json", help="Output JSON summary path")
    args = parser.parse_args()

    pmemd_bin = str(pathlib.Path(args.pmemd).resolve()) if os.path.sep in args.pmemd else args.pmemd
    parm = pathlib.Path(args.parm).resolve()
    crd = pathlib.Path(args.crd).resolve()
    hmr_parm = pathlib.Path(args.hmr_parm).resolve() if args.hmr_parm else None
    work_dir = pathlib.Path(args.work_dir).resolve()
    ntpr = max(1, args.nstlim // 2)

    protocols = [
        ("nvt_langevin_2fs", parm, make_mdin("lg_2fs", args.nstlim, ntpr, args.cut, dt=0.002, ntt=3, gamma_ln=1.0)),
        ("nvt_middle_2fs", parm, make_mdin("mid_2fs", args.nstlim, ntpr, args.cut, dt=0.002, ntt=3, ischeme=1, ithermostat=1, therm_par=1.0)),
        ("npt_mc100_2fs", parm, make_mdin("npt_2fs", args.nstlim, ntpr, args.cut, dt=0.002, ntp=1, barostat=2, mcbarint=100, ntt=3, gamma_ln=1.0)),
    ]
    if hmr_parm is not None:
        protocols.append(
            ("nvt_hmr_middle_4fs", hmr_parm, make_mdin("hmr_4fs", args.nstlim, ntpr, args.cut, dt=0.004, ntt=3, ischeme=1, ithermostat=1, therm_par=1.0))
        )

    modes = [
        ("stock", {"AMBER_USE_CUPO": "0"}),
        ("phase1_nb", {"AMBER_USE_CUPO": "1", "AMBER_CUPO_SPLITTING": "prolate", "AMBER_CUPO_BONDED": "0", "AMBER_CUPO_WHOLE_STEP": "0"}),
        ("stage2a_bonded", {"AMBER_USE_CUPO": "1", "AMBER_CUPO_SPLITTING": "prolate", "AMBER_CUPO_BONDED": "1", "AMBER_CUPO_WHOLE_STEP": "0"}),
        ("phase2_prolate", {"AMBER_USE_CUPO": "1", "AMBER_CUPO_SPLITTING": "prolate", "AMBER_CUPO_BONDED": "1", "AMBER_CUPO_WHOLE_STEP": "1", "AMBER_CUPO_CHUNK": "8"}),
        ("phase2_ewald", {"AMBER_USE_CUPO": "1", "AMBER_CUPO_SPLITTING": "ewald", "AMBER_CUPO_BONDED": "1", "AMBER_CUPO_WHOLE_STEP": "1", "AMBER_CUPO_CHUNK": "8"}),
    ]

    summary = {}
    print(f"| Protocol | Stock | Phase 1 (NB) | Stage 2A (+Bonded) | Phase 2 Prolate (Speedup) | Phase 2 Ewald (Speedup) |")
    print(f"| --- | ---: | ---: | ---: | ---: | ---: |")
    for proto_name, proto_parm, mdin_text in protocols:
        row = {}
        for mode_name, env in modes:
            row[mode_name] = run_case(
                pmemd_bin, work_dir / f"{proto_name}_{mode_name}", proto_parm, crd, mdin_text, args.gpu, env
            )
        summary[proto_name] = row
        s_ns = row["stock"]["last_ns_day"]
        p1_ns = row["phase1_nb"]["last_ns_day"]
        s2a_ns = row["stage2a_bonded"]["last_ns_day"]
        p2p_ns = row["phase2_prolate"]["last_ns_day"]
        p2e_ns = row["phase2_ewald"]["last_ns_day"]
        print(
            f"| {proto_name} | {s_ns:.2f} | {p1_ns:.2f} | {s2a_ns:.2f} | "
            f"{p2p_ns:.2f} ({p2p_ns / s_ns:.2f}x) | {p2e_ns:.2f} ({p2e_ns / s_ns:.2f}x) |"
        )

    out_path = pathlib.Path(args.out_json).resolve()
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
