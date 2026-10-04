"""Bounded full-build comparison against an explicit naive neighbor kernel.

Only the benchmark child's edge builder is replaced; source is not edited.
Both methods execute the same input preparation, igraph allocation, node/edge
attributes, topology filtering, plateau handling and local-optima detection.
The naive method checks each unordered pair once in Python, writes symmetric
Hamming distances to a float64 N-by-N array, then extracts distance-one edges.
It represents this particular simple implementation, not all pairwise methods.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'docs/content/assets/benchmarks/construction.json'


def naive_edges(**kw):
    import numpy as np
    from graphfla._neighbors import EdgeResult
    assert kw['n_edit'] == 1 and kw['epsilon'] == 0 and kw['maximize']
    rows = kw['configs_array'].tolist()
    n = len(rows)
    # float64 matches a conventional general-purpose dense distance matrix.
    distances = np.zeros((n, n), dtype=np.float64)
    for i, left in enumerate(rows):
        for j in range(i + 1, n):
            value = sum(a != b for a, b in zip(left, rows[j]))
            distances[i, j] = distances[j, i] = value
    edges, deltas = [], []
    fitness = kw['fitness']
    for i in range(n):
        for j in np.flatnonzero(distances[i, i + 1:] == 1) + i + 1:
            delta = float(fitness[j] - fitness[i])
            assert delta != 0
            edges.append((i, int(j)) if delta > 0 else (int(j), i))
            deltas.append(abs(delta))
    return EdgeResult(edges, deltas, [])


def worker(bits, method):
    import gc
    import resource
    import numpy as np
    import igraph
    import pandas as pd
    from graphfla.landscape import BooleanLandscape
    from graphfla.landscape import _build
    n = 2 ** bits
    X = ((np.arange(n)[:, None] >> np.arange(bits)) & 1).astype(np.uint8)
    f = 1 + np.random.default_rng(20261004).permutation(n) / n
    # Resolve one-time library setup on a tiny input, identically in both runs.
    BooleanLandscape().build_from_data(X[:16], f[:16], verbose=False)
    if method == 'naive':
        _build.build_edges = naive_edges
    gc.collect()
    start = time.perf_counter()
    landscape = BooleanLandscape().build_from_data(X, f, verbose=False)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak *= 1024
    # Canonicalize graph output only after timing and RSS capture.
    g = landscape.graph
    canonical = {
        'vertices': {name: g.vs[name] for name in sorted(g.vs.attributes())},
        'edges': sorted((int(a), int(b), float(d)) for (a,b),d in zip(g.get_edgelist(),g.es['delta_fit'])),
        'shape': landscape.shape,
        'optima': landscape.n_lo,
    }
    assert landscape.n_configs == n
    assert landscape.n_edges == n * bits // 2
    result = dict(bits=bits, configurations=n, method=method, seconds=elapsed,
                  peak_rss_bytes=int(peak), shape=landscape.shape,
                  input_sha256=hashlib.sha256(X.tobytes()+f.tobytes()).hexdigest(),
                  graph_sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True,default=lambda value: value.item()).encode()).hexdigest(),
                  versions={'numpy':np.__version__,'pandas':pd.__version__,'igraph':igraph.__version__})
    print(json.dumps(result))


def run(sizes, repeats, output):
    from tools._resource_guard import run_guarded
    if any(b not in (8,10,12) for b in sizes) or not 1 <= repeats <= 3:
        raise ValueError('Bounded workloads: 8, 10 or 12 bits; at most three processes per method.')
    env = {**os.environ, **dict.fromkeys(['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'],'1'), 'PYTHONHASHSEED':'0'}
    report = dict(date='2026-10-04',python=platform.python_version(),platform=platform.platform(),
                  machine=platform.machine(),processor=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol={'scope':'full build_from_data; only the naive neighbor kernel is substituted',
                            'distance_matrix':'float64, N × N, symmetric; each unordered pair evaluated once',
                            'seed':20261004,'processes_per_method':repeats,'timed_builds_per_process':1,
                            'setup':'same imports and 16-row warmup; input generation outside timing',
                            'memory':'peak process RSS, including Python, dependencies, inputs and build',
                            'timeout_seconds':45,'memory_limit_bytes':768*1024**2},runs=[],summary=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for bits in sizes:
        runs=[]
        for repeat in range(repeats):
            # Alternate order to reduce systematic thermal/order effects.
            for method in (('graphfla','naive') if repeat%2==0 else ('naive','graphfla')):
                done=run_guarded([sys.executable,str(Path(__file__).resolve()),'--worker','--bits',str(bits),'--method',method],cwd=ROOT,timeout=45,memory_bytes=768*1024**2,env=env)
                result=json.loads(done.stdout)
                result['monitored_tree_peak_rss_bytes']=done.monitored_peak_rss_bytes
                runs.append(result)
                report['runs'].append(result)
                print(f'{bits} bits {method}: {result["seconds"]:.4f}s, {result["peak_rss_bytes"]/1024**2:.1f} MiB',flush=True)
        assert len({r['graph_sha256'] for r in runs})==1, 'Graph outputs differ'
        assert len({r['input_sha256'] for r in runs})==1, 'Inputs differ'
        row={'configurations':2**bits,'bits':bits,'edges':runs[0]['shape'][1],'dense_matrix_bytes':(2**bits)**2*8,'outputs_equal':True}
        for method in ('graphfla','naive'):
            selected=[r for r in runs if r['method']==method]
            row[method]={key:statistics.median(r[key] for r in selected) for key in ('seconds','peak_rss_bytes')}
            row[method]['seconds_range']=[min(r['seconds'] for r in selected),max(r['seconds'] for r in selected)]
        row['speedup']=row['naive']['seconds']/row['graphfla']['seconds']
        row['peak_rss_reduction']=1-row['graphfla']['peak_rss_bytes']/row['naive']['peak_rss_bytes']
        report['summary'].append(row)
        output.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--bits',type=int,default=8)
    parser.add_argument('--method',choices=['graphfla','naive'],default='graphfla')
    parser.add_argument('--sizes',type=int,nargs='+',default=[8,10,12])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.worker:
        worker(args.bits,args.method)
    else:
        run(args.sizes,args.repeats,args.output)
