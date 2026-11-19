#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Self-contained integration verification suite for amber-cupotential.

Verifies Stage 2A (bonded tuple force accuracy vs Phase 1 and Stock), Stage 2B
(whole-step GPU integration NVE conservation, SETTLE/SHAKE H-bond constraints,
chunk=8 vs chunk=1 vs Eager CUDA Graph bitwise equivalence), NVT/NPT ensembles,
and automatic fallbacks (AMBER_USE_CUPO=0, Berendsen barostat=1).

When --parm/--crd are omitted, synthesizes a 2,114-atom periodic n-butane +
700 TIP3P water test system in --work-dir (zero external dataset files needed).
"""

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import netcdf_file


def check_gpu_idle(gpu_id: str, min_free_mib: int = 1500) -> Tuple[int, int]:
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
    util, free_mib = int(util_s), int(free_s)
    if free_mib < min_free_mib:
        raise RuntimeError(
            f"GPU {gpu_id} has insufficient free VRAM ({free_mib} MiB < {min_free_mib} MiB)"
        )
    if util > 15:
        time.sleep(1.0)
    return util, free_mib


def run_pmemd(
    pmemd_bin: str,
    run_dir: pathlib.Path,
    parm: pathlib.Path,
    crd: pathlib.Path,
    mdin_text: str,
    gpu_id: str,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[pathlib.Path, str, str]:
    check_gpu_idle(gpu_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mdin").write_text(mdin_text, encoding="utf-8")

    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if env_overrides:
        for k, v in env_overrides.items():
            env[k] = str(v)

    proc = subprocess.run(
        [
            pmemd_bin,
            "-O",
            "-i",
            "mdin",
            "-p",
            str(parm),
            "-c",
            str(crd),
            "-o",
            "mdout",
            "-r",
            "restrt",
            "-x",
            "mdcrd",
            "-frc",
            "mdfrc",
            "-inf",
            "mdinfo",
        ],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    mdout = (run_dir / "mdout").read_text(encoding="utf-8", errors="replace") if (run_dir / "mdout").exists() else ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"pmemd failed in {run_dir} (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nMDOUT tail:\n{mdout[-2000:]}"
        )
    return run_dir, mdout, proc.stderr


def make_mdin(
    title: str,
    imin: int = 0,
    irest: int = 1,
    ntx: int = 5,
    maxcyc: int = 200,
    nstlim: int = 100,
    ntpr: int = 10,
    ntwx: int = 0,
    ntwr: int = 500000,
    ntwf: int = 0,
    cut: float = 8.0,
    dt: float = 0.002,
    ntb: int = 1,
    ntp: int = 0,
    barostat: int = 2,
    mcbarint: int = 20,
    ntt: int = 3,
    gamma_ln: float = 1.0,
    tempi: float = 0.0,
    temp0: float = 300.0,
    ischeme: int = 0,
    ithermostat: int = 0,
    therm_par: float = 0.0,
    ntc: int = 2,
    ntf: int = 2,
) -> str:
    lines = [title, " &cntrl", f"  imin={imin},"]
    if imin == 1:
        lines.extend([
            f"  maxcyc={maxcyc}, ncyc=50, ntmin=1,",
            f"  ntc={ntc}, ntf={ntf}, ntb={ntb}, cut={cut}, ntpr={ntpr},",
        ])
    else:
        ntb_eff = 2 if ntp > 0 else ntb
        lines.extend([
            f"  ntx={ntx}, irest={irest}, ntxo=2, ioutfm=1,",
            f"  ntc={ntc}, ntf={ntf}, tol=0.00001,",
            f"  nstlim={nstlim}, ntpr={ntpr}, ntwx={ntwx}, ntwr={ntwr},",
            f"  dt={dt}, cut={cut}, ntb={ntb_eff}, ntp={ntp}, ig=12345,",
        ])
        if ntwf != 0:
            lines.append(f"  ntwf={ntwf},")
        if irest == 0 and ntx == 1:
            lines.append(f"  tempi={tempi},")
        if ischeme == 1:
            lines.append(
                f"  ntt={ntt}, temp0={temp0}, ischeme=1, ithermostat={ithermostat}, therm_par={therm_par},"
            )
        elif ntt == 3:
            lines.append(f"  ntt={ntt}, temp0={temp0}, gamma_ln={gamma_ln},")
        else:
            lines.append(f"  ntt={ntt},")
        if ntp > 0:
            lines.append(f"  pres0=1.0, barostat={barostat}, mcbarint={mcbarint}, taup=1.0,")
    lines.extend([" &end", " &ewald", "  dsum_tol=0.00001,", " &end", ""])
    return "\n".join(lines)


def parse_steps(mdout_text: str) -> Dict[int, dict]:
    body = mdout_text.split("A V E R A G E S")[0]
    block_pat = re.compile(
        r"NSTEP\s*=\s*(\d+)\s+TIME\(PS\)\s*=\s*([\d.-]+)\s+TEMP\(K\)\s*=\s*([\d.-]+)\s+PRESS\s*=\s*([\d.-]+)\s*\n"
        r"([\s\S]*?)(?=\n\s*-----|\n\s*A V E R A G E S|$)"
    )
    kv_pat = re.compile(
        r"(Etot|EKtot|EPtot|BOND|ANGLE|DIHED|1-4 NB|1-4 EEL|VDWAALS|EELEC|EHBOND|RESTRAINT|UB|IMP|CMAP|VOLUME|Density)"
        r"\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
    )
    rename = {"1-4 NB": "14_NB", "1-4 EEL": "14_EEL"}
    records = {}
    for m in block_pat.finditer(body):
        nstep = int(m.group(1))
        rec = {
            "nstep": nstep,
            "TIME": float(m.group(2)),
            "TEMP": float(m.group(3)),
            "PRESS": float(m.group(4)),
            "VOLUME": None,
            "Density": None,
        }
        for k, v in kv_pat.findall(m.group(5)):
            rec[rename.get(k, k)] = float(v)
        records[nstep] = rec
    return records


def read_netcdf_restart(path: pathlib.Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    with netcdf_file(str(path), "r", mmap=False) as nf:
        coords = np.array(nf.variables["coordinates"][:], dtype=np.float64)
        vels = np.array(nf.variables["velocities"][:], dtype=np.float64) if "velocities" in nf.variables else None
        box = np.array(nf.variables["cell_lengths"][:], dtype=np.float64) if "cell_lengths" in nf.variables else None
    return coords, vels, box


def read_netcdf_forces_first(path: pathlib.Path) -> np.ndarray:
    with netcdf_file(str(path), "r", mmap=False) as nf:
        frc = np.array(nf.variables["forces"][0], dtype=np.float64)
    return frc


def parse_prmtop_topology(
    parm_path: pathlib.Path,
) -> Tuple[int, List[Tuple[int, int, float]], int, List[Tuple[int, int, int, int]]]:
    text = parm_path.read_text(encoding="utf-8").splitlines()
    sections: Dict[str, List[str]] = {}
    cur = None
    for line in text:
        if line.startswith("%FLAG"):
            cur = line.split()[1]
            sections[cur] = []
        elif line.startswith("%FORMAT") or cur is None:
            continue
        else:
            if cur in ("ATOM_NAME", "RESIDUE_LABEL", "AMBER_ATOM_TYPE"):
                for idx in range(0, len(line), 4):
                    sections[cur].append(line[idx : idx + 4].strip())
            else:
                sections[cur].extend(line.split())
    natoms = int(sections["POINTERS"][0])
    nres = int(sections["POINTERS"][11])
    masses = [float(x) for x in sections["MASS"][:natoms]]
    req = [float(x) for x in sections["BOND_EQUIL_VALUE"]]
    b_raw = [int(x) for x in sections["BONDS_INC_HYDROGEN"]]
    bonds = []
    for k in range(0, len(b_raw), 3):
        i = abs(b_raw[k]) // 3
        j = abs(b_raw[k + 1]) // 3
        itype = b_raw[k + 2] - 1
        if masses[i] > 0.0 and masses[j] > 0.0:
            bonds.append((i, j, req[itype]))
    ncmaps = int(sections["CMAP_COUNT"][0]) if "CMAP_COUNT" in sections else 0
    ep_waters = []
    res_labels = sections.get("RESIDUE_LABEL", [])[:nres]
    res_ptrs = [int(x) - 1 for x in sections.get("RESIDUE_POINTER", [])[:nres]] + [natoms]
    for r in range(min(nres, len(res_labels))):
        if res_labels[r] in ("WAT", "HOH"):
            a0, a1 = res_ptrs[r], res_ptrs[r + 1]
            if a1 - a0 == 4 and masses[a0 + 3] == 0.0:
                ep_waters.append((a0, a0 + 1, a0 + 2, a0 + 3))
    return natoms, bonds, ncmaps, ep_waters


def max_h_bond_deviation(coords: np.ndarray, box_lengths: np.ndarray, h_bonds) -> Tuple[float, float]:
    max_dev = 0.0
    devs = []
    for i, j, r0 in h_bonds:
        d = coords[j] - coords[i]
        d -= box_lengths * np.round(d / box_lengths)
        dist = float(np.linalg.norm(d))
        dev = abs(dist - r0)
        devs.append(dev)
        if dev > max_dev:
            max_dev = dev
    return max_dev, float(np.mean(devs)) if devs else 0.0


def max_ep_water_deviation(
    coords: np.ndarray, box_lengths: np.ndarray, ep_waters: List[Tuple[int, int, int, int]], ref_ep_dist: float
) -> Tuple[float, float]:
    max_dist_dev = 0.0
    max_bis_dev = 0.0
    for o, h1, h2, ep in ep_waters:
        v1 = coords[h1] - coords[o]
        v2 = coords[h2] - coords[o]
        ve = coords[ep] - coords[o]
        v1 -= box_lengths * np.round(v1 / box_lengths)
        v2 -= box_lengths * np.round(v2 / box_lengths)
        ve -= box_lengths * np.round(ve / box_lengths)
        dist_dev = abs(float(np.linalg.norm(ve)) - ref_ep_dist)
        u = v1 / np.linalg.norm(v1)
        v = v2 / np.linalg.norm(v2)
        bis = u + v
        ep_ideal = ref_ep_dist * (bis / np.linalg.norm(bis))
        bis_dev = float(np.linalg.norm(ve - ep_ideal))
        max_dist_dev = max(max_dist_dev, dist_dev)
        max_bis_dev = max(max_bis_dev, bis_dev)
    return max_dist_dev, max_bis_dev


def fmt_section_i8(vals: List[int]) -> str:
    if not vals:
        return "\n"
    out = []
    for i in range(0, len(vals), 10):
        out.append("".join(f"{v:8d}" for v in vals[i : i + 10]))
    return "\n".join(out) + "\n"


def fmt_section_e16(vals: List[float]) -> str:
    if not vals:
        return "\n"
    out = []
    for i in range(0, len(vals), 5):
        out.append("".join(f"{v:16.8E}" for v in vals[i : i + 5]))
    return "\n".join(out) + "\n"


def fmt_section_a4(vals: List[str]) -> str:
    if not vals:
        return "\n"
    out = []
    for i in range(0, len(vals), 20):
        out.append("".join(f"{v:<4s}"[:4] for v in vals[i : i + 20]))
    return "\n".join(out) + "\n"


def synthesize_butane_water_system(out_dir: pathlib.Path, pmemd_bin: str, gpu_id: str) -> Tuple[pathlib.Path, pathlib.Path]:
    """Build a valid Amber7 prmtop + equilibrated NetCDF restart for n-butane + 700 TIP3P waters."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parm_path = out_dir / "butane_tip3p.parm7"
    init_crd_path = out_dir / "butane_tip3p_init.rst7"

    amber_q = 18.2223
    box_l = 30.0
    n_wat = 700

    # Butane (14 atoms): 0:C1, 1..3:H1x, 4:C2, 5..6:H2x, 7:C3, 8..9:H3x, 10:C4, 11..13:H4x
    but_names = ["C1", "H11", "H12", "H13", "C2", "H21", "H22", "C3", "H31", "H32", "C4", "H41", "H42", "H43"]
    but_types = ["CT", "HC", "HC", "HC", "CT", "HC", "HC", "CT", "HC", "HC", "CT", "HC", "HC", "HC"]
    but_iac = [1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2]
    but_mass = [12.01, 1.008, 1.008, 1.008, 12.01, 1.008, 1.008, 12.01, 1.008, 1.008, 12.01, 1.008, 1.008, 1.008]
    # Exact net-zero charges for C4H10
    but_qe = [-0.18, 0.06, 0.06, 0.06, -0.12, 0.06, 0.06, -0.12, 0.06, 0.06, -0.18, 0.06, 0.06, 0.06]
    but_tree = ["M", "E", "E", "E", "M", "E", "E", "M", "E", "E", "E", "E", "E", "E"]

    # Build staggered trans n-butane coordinates centered near (15, 15, 15)
    cc = 1.526
    ch = 1.090
    th = math.radians(109.5)
    c1 = np.array([-1.5 * cc * math.sin(th / 2), -0.5 * cc * math.cos(th / 2), 0.0])
    c2 = np.array([-0.5 * cc * math.sin(th / 2), +0.5 * cc * math.cos(th / 2), 0.0])
    c3 = np.array([+0.5 * cc * math.sin(th / 2), -0.5 * cc * math.cos(th / 2), 0.0])
    c4 = np.array([+1.5 * cc * math.sin(th / 2), +0.5 * cc * math.cos(th / 2), 0.0])

    def unit(v):
        return v / np.linalg.norm(v)

    def place_methyl(c_pos, neighbor_c, dir_hint):
        e1 = unit(c_pos - neighbor_c)
        perp1 = unit(dir_hint - np.dot(dir_hint, e1) * e1)
        perp2 = np.cross(e1, perp1)
        ca = math.cos(math.radians(180.0 - 109.5))
        sa = math.sin(math.radians(180.0 - 109.5))
        hs = []
        for k in range(3):
            phi = k * (2.0 * math.pi / 3.0)
            d = ca * e1 + sa * (math.cos(phi) * perp1 + math.sin(phi) * perp2)
            hs.append(c_pos + ch * d)
        return hs

    def place_ch2(c_pos, n1, n2):
        bis = unit(unit(c_pos - n1) + unit(c_pos - n2))
        perp = unit(np.cross(n2 - c_pos, n1 - c_pos))
        ca = math.cos(math.radians(54.75))
        sa = math.sin(math.radians(54.75))
        return [c_pos + ch * (ca * bis + sa * perp), c_pos + ch * (ca * bis - sa * perp)]

    h1 = place_methyl(c1, c2, c3 - c2)
    h2 = place_ch2(c2, c1, c3)
    h3 = place_ch2(c3, c2, c4)
    h4 = place_methyl(c4, c3, c2 - c3)
    but_xyz = np.array([c1, *h1, c2, *h2, c3, *h3, c4, *h4], dtype=np.float64) + np.array([15.0, 15.0, 15.0])

    # Butane bond topology (0-based within molecule)
    heavy_bonds = [(0, 4, 1), (4, 7, 1), (7, 10, 1)]  # CT-CT -> btype 1
    h_bonds_but = [
        (0, 1, 2), (0, 2, 2), (0, 3, 2),
        (4, 5, 2), (4, 6, 2),
        (7, 8, 2), (7, 9, 2),
        (10, 11, 2), (10, 12, 2), (10, 13, 2),
    ]
    all_but_bonds = [(i, j) for i, j, _ in heavy_bonds + h_bonds_but]

    # Bond types: 1:CT-CT, 2:CT-HC, 3:OW-HW, 4:HW-HW
    bond_rk = [310.0, 340.0, 553.0, 553.0]
    bond_req = [1.526, 1.090, 0.9572, 1.5136]

    # Angle types: 1:CT-CT-CT, 2:CT-CT-HC, 3:HC-CT-HC
    ang_tk = [40.0, 50.0, 35.0]
    ang_teq = [math.radians(109.5), math.radians(109.5), math.radians(109.5)]

    # Generate all butane angles from bond adjacency
    adj = {i: [] for i in range(14)}
    for u, v in all_but_bonds:
        adj[u].append(v)
        adj[v].append(u)
    heavy_angles = []
    h_angles = []
    for j in range(14):
        nbrs = sorted(adj[j])
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                i, k = nbrs[a_idx], nbrs[b_idx]
                nh = (but_types[i] == "HC") + (but_types[j] == "HC") + (but_types[k] == "HC")
                if nh == 0:
                    heavy_angles.append((i, j, k, 1))
                else:
                    atype = 3 if (but_types[i] == "HC" and but_types[k] == "HC") else 2
                    h_angles.append((i, j, k, atype))

    # Dihedral types:
    # 1..3: C1-C2-C3-C4 3-term RY-like Ryckaert/GAFF cosine expansion (n=3, 2, 1)
    # 4: X-CT-CT-X single-term 3-fold (for H-containing dihedrals)
    dihed_pk = [0.18, 0.25, 0.20, 0.155556]
    dihed_pn = [3.0, 2.0, 1.0, 3.0]
    dihed_ph = [0.0, math.pi, 0.0, 0.0]

    heavy_diheds = [
        (0, 4, 7, 10, 1, True),   # skip 1-4 (-k)
        (0, 4, 7, 10, 2, True),   # skip 1-4 (-k)
        (0, 4, 7, 10, 3, False),  # include 1-4 (+k)
    ]
    h_diheds = []
    seen_14 = {(0, 10)}
    for j, k in [(0, 4), (4, 7), (7, 10)]:
        for i in sorted(adj[j]):
            if i == k:
                continue
            for l in sorted(adj[k]):
                if l == j or i == l:
                    continue
                if but_types[i] != "HC" and but_types[l] != "HC":
                    continue
                pair = (min(i, l), max(i, l))
                skip_14 = pair in seen_14
                seen_14.add(pair)
                h_diheds.append((i, j, k, l, 4, skip_14))

    # Amber pointer encoding helper: atom -> 3*atom, k negative when 1-4 is ignored
    def enc_bond(lst):
        out = []
        for i, j, t in lst:
            out.extend([3 * i, 3 * j, t])
        return out

    def enc_ang(lst):
        out = []
        for i, j, k, t in lst:
            out.extend([3 * i, 3 * j, 3 * k, t])
        return out

    def enc_dih(lst):
        out = []
        for i, j, k, l, t, skip_14 in lst:
            k_code = -(3 * k) if skip_14 else (3 * k)
            out.extend([3 * i, 3 * j, k_code, 3 * l, t])
        return out

    # Append 700 waters (2100 atoms, atom offset 14..)
    natoms = 14 + 3 * n_wat
    atom_names = list(but_names)
    charges = [q * amber_q for q in but_qe]
    masses = list(but_mass)
    iac = list(but_iac)
    isym = list(but_types)
    tree = list(but_tree)
    res_labels = ["BUT"]
    res_ptrs = [1]

    wat_h_bonds = []
    r_oh = 0.9572
    half_ang = math.radians(104.52 * 0.5)
    h1_rel = np.array([r_oh * math.sin(half_ang), r_oh * math.cos(half_ang), 0.0])
    h2_rel = np.array([-r_oh * math.sin(half_ang), r_oh * math.cos(half_ang), 0.0])

    # Place 700 waters on a 9x9x9 grid skipping 27 central points around (15, 15, 15)
    grid_pts = []
    spacing = box_l / 9.0
    for gx in range(9):
        for gy in range(9):
            for gz in range(9):
                pos = np.array([(gx + 0.5) * spacing, (gy + 0.5) * spacing, (gz + 0.5) * spacing])
                if np.max(np.abs(pos - 15.0)) < 4.5:
                    continue
                grid_pts.append(pos)
    assert len(grid_pts) >= n_wat
    wat_xyz = []

    for w in range(n_wat):
        o_idx = 14 + 3 * w
        h1_idx = o_idx + 1
        h2_idx = o_idx + 2
        atom_names.extend(["O", "H1", "H2"])
        charges.extend([-0.834 * amber_q, 0.417 * amber_q, 0.417 * amber_q])
        masses.extend([16.00, 1.008, 1.008])
        iac.extend([3, 4, 4])
        isym.extend(["OW", "HW", "HW"])
        tree.extend(["M", "E", "E"])
        res_labels.append("WAT")
        res_ptrs.append(o_idx + 1)
        wat_h_bonds.append((o_idx, h1_idx, 3))
        wat_h_bonds.append((o_idx, h2_idx, 3))
        wat_h_bonds.append((h1_idx, h2_idx, 4))
        o_pos = grid_pts[w]
        wat_xyz.extend([o_pos, o_pos + h1_rel, o_pos + h2_rel])

    all_xyz = np.vstack([but_xyz, np.array(wat_xyz, dtype=np.float64)])

    # Nonbonded LJ matrix (4x4, Lorentz-Berthelot)
    rstar = [1.9080, 1.4870, 1.7683, 0.0]
    eps = [0.1094, 0.0157, 0.1520, 0.0]
    sigma = [2.0 * r / (2.0 ** (1.0 / 6.0)) if r > 0.0 else 0.0 for r in rstar]
    ntypes = 4
    ico = [0] * (ntypes * ntypes)
    cn1 = []
    cn2 = []
    pair_idx = 0
    for a in range(ntypes):
        for b in range(a + 1):
            pair_idx += 1
            ico[a * ntypes + b] = pair_idx
            ico[b * ntypes + a] = pair_idx
            e_ab = math.sqrt(eps[a] * eps[b])
            if e_ab > 0.0:
                s_ab = 0.5 * (sigma[a] + sigma[b])
                cn1.append(4.0 * e_ab * (s_ab ** 12))
                cn2.append(4.0 * e_ab * (s_ab ** 6))
            else:
                cn1.append(0.0)
                cn2.append(0.0)

    # Exclusions (all 1-2, 1-3, 1-4 pairs with j > i; Amber requires numex[i] >= 1 with 0 placeholder)
    excl_sets: List[set] = [set() for _ in range(natoms)]
    for i, j in all_but_bonds:
        excl_sets[min(i, j)].add(max(i, j))
    for i, j, k, _ in heavy_angles + h_angles:
        excl_sets[min(i, k)].add(max(i, k))
    for i, j in seen_14:
        excl_sets[min(i, j)].add(max(i, j))
    for w in range(n_wat):
        o = 14 + 3 * w
        excl_sets[o].update([o + 1, o + 2])
        excl_sets[o + 1].add(o + 2)

    numex = []
    natex = []
    for i in range(natoms):
        s = sorted(excl_sets[i])
        if not s:
            numex.append(1)
            natex.append(0)
        else:
            numex.append(len(s))
            natex.extend(x + 1 for x in s)  # 1-based!

    all_h_bonds = h_bonds_but + wat_h_bonds
    pointers = [0] * 31
    pointers[0] = natoms
    pointers[1] = ntypes
    pointers[2] = len(all_h_bonds)
    pointers[3] = len(heavy_bonds)
    pointers[4] = len(h_angles)
    pointers[5] = len(heavy_angles)
    pointers[6] = len(h_diheds)
    pointers[7] = len(heavy_diheds)
    pointers[10] = len(natex)
    pointers[11] = len(res_labels)
    pointers[12] = len(heavy_bonds)
    pointers[13] = len(heavy_angles)
    pointers[14] = len(heavy_diheds)
    pointers[15] = len(bond_rk)
    pointers[16] = len(ang_tk)
    pointers[17] = len(dihed_pk)
    pointers[18] = ntypes
    pointers[19] = 1
    pointers[27] = 1   # IFBOX=1 orthogonal
    pointers[28] = 14  # NMXRS

    parm_parts = [
        "%VERSION  VERSION_STAMP = V0001.000  DATE = 11/18/26  00:00:00\n",
        "%FLAG TITLE\n%FORMAT(20a4)\nn-butane + 700 TIP3P water (synthetic verification)\n",
        "%FLAG POINTERS\n%FORMAT(10I8)\n" + fmt_section_i8(pointers),
        "%FLAG ATOM_NAME\n%FORMAT(20a4)\n" + fmt_section_a4(atom_names),
        "%FLAG CHARGE\n%FORMAT(5E16.8)\n" + fmt_section_e16(charges),
        "%FLAG MASS\n%FORMAT(5E16.8)\n" + fmt_section_e16(masses),
        "%FLAG ATOM_TYPE_INDEX\n%FORMAT(10I8)\n" + fmt_section_i8(iac),
        "%FLAG NUMBER_EXCLUDED_ATOMS\n%FORMAT(10I8)\n" + fmt_section_i8(numex),
        "%FLAG NONBONDED_PARM_INDEX\n%FORMAT(10I8)\n" + fmt_section_i8(ico),
        "%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\n" + fmt_section_a4(res_labels),
        "%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n" + fmt_section_i8(res_ptrs),
        "%FLAG BOND_FORCE_CONSTANT\n%FORMAT(5E16.8)\n" + fmt_section_e16(bond_rk),
        "%FLAG BOND_EQUIL_VALUE\n%FORMAT(5E16.8)\n" + fmt_section_e16(bond_req),
        "%FLAG ANGLE_FORCE_CONSTANT\n%FORMAT(5E16.8)\n" + fmt_section_e16(ang_tk),
        "%FLAG ANGLE_EQUIL_VALUE\n%FORMAT(5E16.8)\n" + fmt_section_e16(ang_teq),
        "%FLAG DIHEDRAL_FORCE_CONSTANT\n%FORMAT(5E16.8)\n" + fmt_section_e16(dihed_pk),
        "%FLAG DIHEDRAL_PERIODICITY\n%FORMAT(5E16.8)\n" + fmt_section_e16(dihed_pn),
        "%FLAG DIHEDRAL_PHASE\n%FORMAT(5E16.8)\n" + fmt_section_e16(dihed_ph),
        "%FLAG SOLTY\n%FORMAT(5E16.8)\n" + fmt_section_e16([0.0] * ntypes),
        "%FLAG LENNARD_JONES_ACOEF\n%FORMAT(5E16.8)\n" + fmt_section_e16(cn1),
        "%FLAG LENNARD_JONES_BCOEF\n%FORMAT(5E16.8)\n" + fmt_section_e16(cn2),
        "%FLAG BONDS_INC_HYDROGEN\n%FORMAT(10I8)\n" + fmt_section_i8(enc_bond(all_h_bonds)),
        "%FLAG BONDS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n" + fmt_section_i8(enc_bond(heavy_bonds)),
        "%FLAG ANGLES_INC_HYDROGEN\n%FORMAT(10I8)\n" + fmt_section_i8(enc_ang(h_angles)),
        "%FLAG ANGLES_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n" + fmt_section_i8(enc_ang(heavy_angles)),
        "%FLAG DIHEDRALS_INC_HYDROGEN\n%FORMAT(10I8)\n" + fmt_section_i8(enc_dih(h_diheds)),
        "%FLAG DIHEDRALS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n" + fmt_section_i8(enc_dih(heavy_diheds)),
        "%FLAG EXCLUDED_ATOMS_LIST\n%FORMAT(10I8)\n" + fmt_section_i8(natex),
        "%FLAG HBOND_ACOEF\n%FORMAT(5E16.8)\n" + fmt_section_e16([0.0]),
        "%FLAG HBOND_BCOEF\n%FORMAT(5E16.8)\n" + fmt_section_e16([0.0]),
        "%FLAG HBCUT\n%FORMAT(5E16.8)\n" + fmt_section_e16([0.0]),
        "%FLAG AMBER_ATOM_TYPE\n%FORMAT(20a4)\n" + fmt_section_a4(isym),
        "%FLAG TREE_CHAIN_CLASSIFICATION\n%FORMAT(20a4)\n" + fmt_section_a4(tree),
        "%FLAG JOIN_ARRAY\n%FORMAT(10I8)\n" + fmt_section_i8([0] * natoms),
        "%FLAG IROTAT\n%FORMAT(10I8)\n" + fmt_section_i8([0] * natoms),
        "%FLAG SOLVENT_POINTERS\n%FORMAT(10I8)\n" + fmt_section_i8([1, 1 + n_wat, 2]),
        "%FLAG ATOMS_PER_MOLECULE\n%FORMAT(10I8)\n" + fmt_section_i8([14] + [3] * n_wat),
        "%FLAG BOX_DIMENSIONS\n%FORMAT(5E16.8)\n" + fmt_section_e16([90.0, box_l, box_l, box_l]),
    ]
    parm_path.write_text("".join(parm_parts), encoding="utf-8")

    flat = all_xyz.reshape(-1)
    crd_lines = [
        "synthetic butane + 700 TIP3P water",
        f"{natoms:6d}",
    ]
    for i in range(0, len(flat), 6):
        crd_lines.append("".join(f"{x:12.7f}" for x in flat[i : i + 6]))
    crd_lines.append("".join(f"{x:12.7f}" for x in [box_l, box_l, box_l, 90.0, 90.0, 90.0]))
    init_crd_path.write_text("\n".join(crd_lines) + "\n", encoding="utf-8")

    # Equilibrate cleanly with stock pmemd (minimization + 200-step 300 K NVT warmup)
    run_pmemd(
        pmemd_bin,
        out_dir / "min",
        parm_path,
        init_crd_path,
        make_mdin("synth_min", imin=1, maxcyc=200, ntpr=100, cut=8.0),
        gpu_id,
        {"AMBER_USE_CUPO": "0"},
    )
    run_pmemd(
        pmemd_bin,
        out_dir / "eq",
        parm_path,
        out_dir / "min" / "restrt",
        make_mdin(
            "synth_eq",
            imin=0,
            irest=0,
            ntx=1,
            tempi=300.0,
            nstlim=1000,
            ntpr=500,
            cut=8.0,
            dt=0.002,
            ntt=3,
            gamma_ln=5.0,
        ),
        gpu_id,
        {"AMBER_USE_CUPO": "0"},
    )
    eq_rst = out_dir / "butane_tip3p_eq.rst7"
    shutil.copy2(out_dir / "eq" / "restrt", eq_rst)
    return parm_path, eq_rst


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify amber-cupotential integration end-to-end.")
    default_pmemd = (
        os.path.join(os.environ["AMBERHOME"], "bin", "pmemd.cuda_SPFP")
        if os.environ.get("AMBERHOME")
        else "pmemd.cuda_SPFP"
    )
    parser.add_argument("--pmemd", default=default_pmemd, help="Path to pmemd.cuda_SPFP")
    parser.add_argument("--parm", help="Amber prmtop file (optional; synthesizes test system if omitted)")
    parser.add_argument("--crd", help="Amber coordinate/restart file matching --parm")
    parser.add_argument("--hmr-parm", help="Optional HMR prmtop for 4fs Middle test")
    parser.add_argument("--cut", type=float, default=8.0, help="Cutoff in Angstroms (default: 8.0)")
    parser.add_argument("--gpu", default="0", help="CUDA GPU index (default: 0)")
    parser.add_argument(
        "--work-dir",
        default="./verify_runs",
        help="Working directory for temporary outputs (default: ./verify_runs)",
    )
    args = parser.parse_args()

    if bool(args.parm) != bool(args.crd):
        parser.error("--parm and --crd must be supplied together")

    pmemd_bin = str(pathlib.Path(args.pmemd).resolve()) if os.path.sep in args.pmemd else args.pmemd
    work_dir = pathlib.Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.parm:
        parm = pathlib.Path(args.parm).resolve()
        crd = pathlib.Path(args.crd).resolve()
    else:
        print("Synthesizing self-contained 2,114-atom n-butane + 700 TIP3P water system...")
        parm, crd = synthesize_butane_water_system(work_dir / "synth_system", pmemd_bin, args.gpu)

    natoms, h_bonds, ncmaps, ep_waters = parse_prmtop_topology(parm)
    hmr_parm = pathlib.Path(args.hmr_parm).resolve() if args.hmr_parm else None
    ref_ep_dist = None
    if ep_waters:
        c0, _, b0 = read_netcdf_restart(crd)
        dists = []
        for o, _, _, ep in ep_waters:
            ve = c0[ep] - c0[o]
            if b0 is not None:
                ve -= b0 * np.round(ve / b0)
            dists.append(float(np.linalg.norm(ve)))
        ref_ep_dist = float(np.mean(dists))
    results = {"natoms": natoms, "h_bonds": len(h_bonds), "cmaps": ncmaps, "ep_waters": len(ep_waters), "checks": {}}

    # Note: intentionally do NOT override CUPO_LIB_PATH / CUPO_CUDA13_LIB_DIR so
    # runtime auto-discovery inside cupo_nonbond.cu is verified as well.
    common_prolate = {"AMBER_USE_CUPO": "1", "AMBER_CUPO_SPLITTING": "prolate"}

    print("[1/5] Checking Stage 2A single-step force & energy accuracy...")
    sp_mdin = make_mdin("sp_force", nstlim=1, ntpr=100, ntwf=1, cut=args.cut, dt=0.000001, ntt=0)
    d_stk, m_stk, _ = run_pmemd(pmemd_bin, work_dir / "sp_stock", parm, crd, sp_mdin, args.gpu, {"AMBER_USE_CUPO": "0"})
    d_p1, m_p1, _ = run_pmemd(
        pmemd_bin, work_dir / "sp_p1", parm, crd, sp_mdin, args.gpu,
        {**common_prolate, "AMBER_CUPO_BONDED": "0", "AMBER_CUPO_WHOLE_STEP": "0"},
    )
    d_2a, m_2a, _ = run_pmemd(
        pmemd_bin, work_dir / "sp_2a", parm, crd, sp_mdin, args.gpu,
        {**common_prolate, "AMBER_CUPO_BONDED": "1", "AMBER_CUPO_WHOLE_STEP": "0"},
    )

    f_stk = read_netcdf_forces_first(d_stk / "mdfrc")
    f_p1 = read_netcdf_forces_first(d_p1 / "mdfrc")
    f_2a = read_netcdf_forces_first(d_2a / "mdfrc")

    df_2a_p1 = f_2a - f_p1
    rmse_2a_p1 = float(np.sqrt(np.mean(df_2a_p1 ** 2)))
    rel_2a_p1 = float(np.linalg.norm(df_2a_p1) / np.linalg.norm(f_p1))
    df_2a_stk = f_2a - f_stk
    rmse_2a_stk = float(np.sqrt(np.mean(df_2a_stk ** 2)))
    rel_2a_stk = float(np.linalg.norm(df_2a_stk) / np.linalg.norm(f_stk))

    ene_mdin = make_mdin("sp_ene", nstlim=1, ntpr=1, ntwx=0, ntwf=0, cut=args.cut, dt=0.002, ntt=0)
    _, m_ene_stk, _ = run_pmemd(pmemd_bin, work_dir / "ene_stock", parm, crd, ene_mdin, args.gpu, {"AMBER_USE_CUPO": "0"})
    _, m_ene_2a, _ = run_pmemd(
        pmemd_bin, work_dir / "ene_2a", parm, crd, ene_mdin, args.gpu,
        {**common_prolate, "AMBER_CUPO_BONDED": "1", "AMBER_CUPO_WHOLE_STEP": "0"},
    )
    e_stk = parse_steps(m_ene_stk)[1]
    e_2a = parse_steps(m_ene_2a)[1]
    term_keys = ["BOND", "ANGLE", "DIHED", "14_NB", "14_EEL"]
    if ncmaps > 0 and "CMAP" in e_stk and "CMAP" in e_2a:
        term_keys.append("CMAP")
    bonded_diffs = {k: abs(e_2a[k] - e_stk[k]) for k in term_keys}
    results["checks"]["stage2a_force_energy"] = {
        "rmse_2a_vs_p1": rmse_2a_p1,
        "relL2_2a_vs_p1": rel_2a_p1,
        "rmse_2a_vs_stock": rmse_2a_stk,
        "relL2_2a_vs_stock": rel_2a_stk,
        "bonded_energy_diffs": bonded_diffs,
    }
    assert rmse_2a_p1 < 1e-3, f"Stage 2A vs P1 RMSE too high: {rmse_2a_p1}"
    assert rel_2a_p1 < 2e-4, f"Stage 2A vs P1 relL2 too high: {rel_2a_p1}"
    assert rel_2a_stk < 2e-3, f"Stage 2A vs Stock relL2 too high: {rel_2a_stk}"
    for k, dv in bonded_diffs.items():
        assert dv < 5e-3, f"Energy term {k} diff too high: {dv}"
    print(f"  PASS: 2A vs P1 RMSE={rmse_2a_p1:.3e} relL2={rel_2a_p1:.3e}, 2A vs Stock relL2={rel_2a_stk:.3e}")

    print("[2/5] Checking Stage 2B whole-step NVE conservation & SETTLE/SHAKE constraints...")
    nve_mdin = make_mdin("nve_100", nstlim=100, ntpr=20, cut=args.cut, dt=0.001, ntt=0)
    d_nve, m_nve, _ = run_pmemd(
        pmemd_bin, work_dir / "nve_ws1", parm, crd, nve_mdin, args.gpu,
        {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1", "AMBER_CUPO_CHUNK": "8"},
    )
    recs = parse_steps(m_nve)
    step_keys = sorted(recs.keys())
    assert len(step_keys) >= 2, f"Expected >=2 printed steps in NVE output, got {step_keys}"
    d_etot = recs[step_keys[-1]]["Etot"] - recs[step_keys[0]]["Etot"]
    drift_per_atom = abs(d_etot) / natoms
    c_nve, _, box_nve = read_netcdf_restart(d_nve / "restrt")
    max_hb_dev, mean_hb_dev = max_h_bond_deviation(c_nve, box_nve, h_bonds)
    nve_res = {
        "dEtot": d_etot,
        "abs_dEtot_per_atom": drift_per_atom,
        "max_hbond_dev_A": max_hb_dev,
        "mean_hbond_dev_A": mean_hb_dev,
    }
    if ep_waters and ref_ep_dist is not None:
        max_ep_d, max_ep_b = max_ep_water_deviation(c_nve, box_nve, ep_waters, ref_ep_dist)
        nve_res["max_ep_dist_dev_A"] = max_ep_d
        nve_res["max_ep_bisector_dev_A"] = max_ep_b
        assert max_ep_d < 1e-6, f"EP distance violation too large: {max_ep_d}"
        assert max_ep_b < 1e-6, f"EP bisector violation too large: {max_ep_b}"
    results["checks"]["stage2b_nve"] = nve_res
    assert drift_per_atom < 2e-3, f"NVE energy drift too large: {drift_per_atom}"
    assert max_hb_dev < 1e-5, f"H-bond constraint violation too large: {max_hb_dev}"
    print(f"  PASS: NVE dEtot={d_etot:+.4f} kcal/mol ({drift_per_atom:.2e}/atom), max H-bond dev={max_hb_dev:.2e} A")

    print("[3/5] Checking CUDA Graph chunk=8 vs chunk=1 vs Eager bitwise equivalence...")
    graph_cases = {}
    for label, gmdin in [
        ("langevin_2fs", make_mdin("g_lg", nstlim=100, ntpr=50, cut=args.cut, dt=0.002, ntt=3, gamma_ln=1.0)),
        ("middle_2fs", make_mdin("g_mid", nstlim=100, ntpr=50, cut=args.cut, dt=0.002, ntt=3, ischeme=1, ithermostat=1, therm_par=1.0)),
    ]:
        da, _, _ = run_pmemd(
            pmemd_bin, work_dir / f"{label}_c8", parm, crd, gmdin, args.gpu,
            {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1", "AMBER_CUPO_CHUNK": "8", "AMBER_CUPO_GRAPH": "1"},
        )
        db, _, _ = run_pmemd(
            pmemd_bin, work_dir / f"{label}_c1", parm, crd, gmdin, args.gpu,
            {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1", "AMBER_CUPO_CHUNK": "1", "AMBER_CUPO_GRAPH": "1"},
        )
        dc, _, _ = run_pmemd(
            pmemd_bin, work_dir / f"{label}_eager", parm, crd, gmdin, args.gpu,
            {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1", "AMBER_CUPO_CHUNK": "1", "AMBER_CUPO_GRAPH": "0"},
        )
        ca, va, _ = read_netcdf_restart(da / "restrt")
        cb, vb, _ = read_netcdf_restart(db / "restrt")
        cc, vc, _ = read_netcdf_restart(dc / "restrt")
        c_diff_18 = float(np.max(np.abs(ca - cb)))
        v_diff_18 = float(np.max(np.abs(va - vb)))
        c_diff_e1 = float(np.max(np.abs(cb - cc)))
        v_diff_e1 = float(np.max(np.abs(vb - vc)))
        graph_cases[label] = {
            "max_crd_c8_vs_c1": c_diff_18,
            "max_vel_c8_vs_c1": v_diff_18,
            "max_crd_c1_vs_eager": c_diff_e1,
            "max_vel_c1_vs_eager": v_diff_e1,
        }
        assert c_diff_18 == 0.0 and v_diff_18 == 0.0, f"{label} chunk=8 vs chunk=1 not bitwise identical!"
        assert c_diff_e1 == 0.0 and v_diff_e1 == 0.0, f"{label} chunk=1 vs Eager not bitwise identical!"
        print(f"  PASS {label}: chunk=8 vs chunk=1 vs Eager bitwise identical (max diff = 0.0)")
    results["checks"]["graph_equivalence"] = graph_cases

    print("[4/5] Checking NVT Langevin/Middle and NPT MC ensembles (200 steps)...")
    ens_results = {}
    d_l, m_l, _ = run_pmemd(
        pmemd_bin, work_dir / "ens_langevin", parm, crd,
        make_mdin("ens_lg", nstlim=200, ntpr=100, cut=args.cut, dt=0.002, ntt=3, gamma_ln=1.0),
        args.gpu, {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1"},
    )
    ens_results["nvt_langevin_2fs"] = parse_steps(m_l)[200]["TEMP"]

    d_m, m_m, _ = run_pmemd(
        pmemd_bin, work_dir / "ens_middle", parm, crd,
        make_mdin("ens_mid", nstlim=200, ntpr=100, cut=args.cut, dt=0.002, ntt=3, ischeme=1, ithermostat=1, therm_par=1.0),
        args.gpu, {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1"},
    )
    ens_results["nvt_middle_2fs"] = parse_steps(m_m)[200]["TEMP"]

    if hmr_parm is not None:
        _, m_h, _ = run_pmemd(
            pmemd_bin, work_dir / "ens_hmr_4fs", hmr_parm, crd,
            make_mdin("ens_hmr", nstlim=200, ntpr=100, cut=args.cut, dt=0.004, ntt=3, ischeme=1, ithermostat=1, therm_par=1.0),
            args.gpu, {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1"},
        )
        ens_results["nvt_hmr_middle_4fs"] = parse_steps(m_h)[200]["TEMP"]

    d_npt, m_npt, _ = run_pmemd(
        pmemd_bin, work_dir / "ens_npt_mc", parm, crd,
        make_mdin("ens_npt", nstlim=200, ntpr=100, ntwx=100, cut=args.cut, dt=0.002, ntp=1, barostat=2, mcbarint=20, ntt=3, gamma_ln=1.0),
        args.gpu, {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1"},
    )
    rec_npt = parse_steps(m_npt)[200]
    ens_results["npt_mc20_2fs_density"] = rec_npt["Density"]
    for k, val in ens_results.items():
        assert val is not None and math.isfinite(val), f"Non-finite ensemble metric for {k}: {val}"
    print("  PASS:", ens_results)
    results["checks"]["ensembles"] = ens_results

    print("[5/5] Checking automatic fallbacks (AMBER_USE_CUPO=0 and barostat=1)...")
    run_pmemd(
        pmemd_bin, work_dir / "fb_disabled", parm, crd,
        make_mdin("fb_off", nstlim=20, ntpr=20, cut=args.cut, dt=0.002, ntt=3, gamma_ln=1.0),
        args.gpu, {"AMBER_USE_CUPO": "0"},
    )
    run_pmemd(
        pmemd_bin, work_dir / "fb_berendsen", parm, crd,
        make_mdin("fb_ber", nstlim=20, ntpr=20, cut=args.cut, dt=0.002, ntp=1, barostat=1, ntt=3, gamma_ln=1.0),
        args.gpu, {**common_prolate, "AMBER_CUPO_WHOLE_STEP": "1"},
    )
    results["checks"]["fallbacks"] = {"cupo_0": "PASS", "barostat_1": "PASS"}
    print("  PASS: both fallbacks completed cleanly.")

    out_json = work_dir / "verify_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nALL INTEGRATION VERIFICATION CHECKS PASSED. Summary saved to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
