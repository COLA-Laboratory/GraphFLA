"""Modal benchmark harness for GraphFLA landscape construction.

Runs the construction benchmark on Modal (4 vCPU / 8 GiB, single-threaded) over
8 datasets (4 datatypes x 2 scales), 5 reps each, plus the pytest suite to
guarantee no behaviour change.

  graphfla source  -> mounted from THIS worktree (the variant under test)
  7 local datasets -> mounted from the fixed main checkout (absolute path)
  ordinal-large    -> read from the existing "HPO" Modal volume

Usage (from any worktree root):
  modal run bench/bench_modal.py --action peek
  modal run bench/bench_modal.py --action bench --label baseline
  modal run bench/bench_modal.py --action bench --label baseline --only GB1_protein_large --reps 3
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import modal

# --- paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]          # worktree root (variant)
MAIN = Path("/Users/arwen/Documents/GitHub/GraphFLA")  # stable data source
DATA_BIO = MAIN / "data" / "BioSequence"
DATA_WREOS = MAIN / "data" / "Materials" / "WReOs"

N_REPS_DEFAULT = 5

# --- dataset registry ------------------------------------------------------
# container paths: local CSVs flattened to /root/data/<file>; ordinal-large on
# the HPO volume mounted at /hpo. ordinal-large columns are filled after peek().
DATASETS = [
    dict(name="GB1_protein_large", type="protein",
         path="/root/data/Wu2016_GB1.csv", xcols="sequences", fcol="fitness"),
    dict(name="TrpB3I_protein_small", type="protein",
         path="/root/data/Johnston2024_TrpB3I.csv", xcols="sequences", fcol="fitness"),
    dict(name="Papkou_dna_large", type="dna",
         path="/root/data/Papkou2023_DHFR_RAW.csv", xcols="seq", fcol="fitness"),
    dict(name="Westmann_dna_small", type="dna",
         path="/root/data/Westmann2024.csv", xcols="sequences", fcol="fitness"),
    dict(name="CR9114h1_boolean_large", type="boolean",
         path="/root/data/Phillips2021_CR9114_h1.csv", xcols="sequences", fcol="fitness"),
    dict(name="CR6261h1_boolean_small", type="boolean",
         path="/root/data/Phillips2021_CR6261_h1.csv", xcols="sequences", fcol="fitness"),
    dict(name="WReOs_ordinal_small", type="ordinal",
         path="/root/data/WReOs_simplex.csv", xcols="W,Re", fcol="TOPSIS_Ci"),
    dict(name="HPO_ordinal_large", type="ordinal",
         path="/hpo/results_cv_super/44136.csv",
         xcols="learning_rate,subsample,max_depth,min_child_weight,n_estimators",
         fcol="mean_r2"),
]

# --- image -----------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4", "pandas==2.2.2", "scipy==1.13.1",
        "scikit-learn==1.5.1", "python-igraph==0.11.6", "tqdm==4.66.4",
        "joblib==1.4.2", "psutil==5.9.8", "pytest==8.2.2",
    )
    .env({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "PYTHONPATH": "/root/src",
    })
    .add_local_dir(str(REPO / "graphfla"), "/root/src/graphfla", copy=False)
    .add_local_dir(str(REPO / "tests"), "/root/src/tests", copy=False)
    .add_local_dir(str(REPO / "bench"), "/root/src/bench", copy=False)
    .add_local_file(str(DATA_BIO / "Wu2016_GB1.csv"), "/root/data/Wu2016_GB1.csv", copy=False)
    .add_local_file(str(DATA_BIO / "Johnston2024_TrpB3I.csv"), "/root/data/Johnston2024_TrpB3I.csv", copy=False)
    .add_local_file(str(DATA_BIO / "Papkou2023_DHFR_RAW.csv"), "/root/data/Papkou2023_DHFR_RAW.csv", copy=False)
    .add_local_file(str(DATA_BIO / "Westmann2024.csv"), "/root/data/Westmann2024.csv", copy=False)
    .add_local_file(str(DATA_BIO / "Phillips2021_CR9114_h1.csv"), "/root/data/Phillips2021_CR9114_h1.csv", copy=False)
    .add_local_file(str(DATA_BIO / "Phillips2021_CR6261_h1.csv"), "/root/data/Phillips2021_CR6261_h1.csv", copy=False)
    .add_local_file(str(DATA_WREOS / "simplex.csv"), "/root/data/WReOs_simplex.csv", copy=False)
)

app = modal.App("graphfla-bench")
hpo_vol = modal.Volume.from_name("HPO")


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(
        mean=round(statistics.mean(vals), 4),
        std=round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        min=round(min(vals), 4),
        max=round(max(vals), 4),
        vals=[round(v, 4) for v in vals],
    )


@app.function(image=image, volumes={"/hpo": hpo_vol}, cpu=4.0, memory=8192,
              timeout=3600)
def peek():
    """Inspect ordinal-large (HPO/results_cv_super) candidate schemas."""
    import os
    import pandas as pd
    base = "/hpo/results_cv_super"
    out = ["FILES:" + ",".join(sorted(os.listdir(base))[:60])]
    for fn in ["44136.csv", "44134.csv", "44132.csv", "44133.csv",
               "44137.csv", "44138.csv"]:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            out.append(f"{fn}: MISSING")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            out.append(f"{fn}: ERR {e}")
            continue
        nun = {c: int(df[c].nunique()) for c in df.columns}
        dt = {c: str(t) for c, t in df.dtypes.items()}
        out.append(f"{fn}: nrows={len(df)} ncols={len(df.columns)}")
        out.append(f"   cols={list(df.columns)}")
        out.append(f"   dtypes={dt}")
        out.append(f"   nunique={nun}")
    return "\n".join(out)


@app.function(image=image, volumes={"/hpo": hpo_vol}, cpu=4.0, memory=8192,
              timeout=3600)
def run_all(label: str = "baseline", only: str = "", reps: int = N_REPS_DEFAULT):
    """Run all (or one) dataset(s), reps each, then the pytest suite."""
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/src"

    datasets = DATASETS
    if only:
        wanted = set(only.split(","))
        datasets = [d for d in DATASETS if d["name"] in wanted]

    results = []
    for ds in datasets:
        per_rep = []
        for rep in range(reps):
            cmd = [
                sys.executable, "/root/src/bench/_bench_core.py",
                "--name", ds["name"], "--type", ds["type"],
                "--path", ds["path"], "--xcols", ds["xcols"],
                "--fcol", ds["fcol"], "--rep", str(rep),
            ]
            if ds.get("data_types"):
                cmd += ["--data-types", json.dumps(ds["data_types"])]
            if ds.get("strategy"):
                cmd += ["--strategy", ds["strategy"]]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  env=env, cwd="/root/src", timeout=1200)
            rec = None
            for line in proc.stdout.splitlines():
                if line.startswith("REPJSON "):
                    rec = json.loads(line[len("REPJSON "):])
            if rec is None:
                results.append(dict(name=ds["name"], error="no REPJSON",
                                    stderr=proc.stderr[-2000:],
                                    stdout=proc.stdout[-1000:]))
                per_rep = []
                break
            per_rep.append(rec)

        if not per_rep:
            continue

        module_keys = set()
        for r in per_rep:
            module_keys.update(r["modules"].keys())
        modules_agg = {
            k: _agg([r["modules"].get(k) for r in per_rep])
            for k in sorted(module_keys)
        }
        results.append(dict(
            name=ds["name"], type=ds["type"],
            n_configs=per_rep[-1]["n_configs"], n_edges=per_rep[-1]["n_edges"],
            total_s=_agg([r["total_s"] for r in per_rep]),
            peak_rss_mb=_agg([r["peak_rss_mb"] for r in per_rep]),
            construct_delta_mb=_agg([r["construct_delta_mb"] for r in per_rep]),
            rss_postbuild_mb=_agg([r["rss_postbuild_mb"] for r in per_rep]),
            modules=modules_agg,
        ))

    # behaviour guard: full test suite must pass
    tproc = subprocess.run(
        [sys.executable, "-m", "pytest", "/root/src/tests", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd="/root/src", timeout=1200,
    )
    tests = dict(returncode=tproc.returncode,
                 summary=tproc.stdout.strip().splitlines()[-1]
                 if tproc.stdout.strip() else "",
                 tail=tproc.stdout.strip()[-1500:])

    return dict(label=label, reps=reps, datasets=results, tests=tests)


@app.local_entrypoint()
def main(action: str = "bench", label: str = "baseline", only: str = "",
         reps: int = N_REPS_DEFAULT):
    if action == "peek":
        print(peek.remote())
        return
    out = run_all.remote(label=label, only=only, reps=reps)
    print("=== BENCH_JSON_START ===")
    print(json.dumps(out, indent=2))
    print("=== BENCH_JSON_END ===")
