# nmon

Terminal-first monitoring for NMoE training runs.

Design goals:
- Single-run-first (fast follow: multi-run compare).
- Rendering via `lipgloss` widgets (nvitop-style tables + compact charts).
- Reads the existing NMoE metrics/artifacts:
  - DuckDB: `/data/metrics/{run_id}/rank_0.duckdb` (rank-0-only for v1)
  - SQLite: `/data/experiments.db` (read-only snapshot)

## Run (container-first)

```bash
cd /workspace/nmoe

docker build -f docker/Dockerfile.nmon -t nmon:dev .

# Default: read from cluster PVC via k8s exec
docker run --rm -it --net=host \
  -v ~/.kube:/root/.kube:ro \
  -e KUBECONFIG=/root/.kube/config \
  nmon:dev

# Local data (if /data is mounted locally)
docker run --rm -it \
  -v /data:/data:ro \
  nmon:dev --k8s=false
```

## Run (opt-in local)

```bash
cd /workspace/nmoe/tools/nmon
go run ./cmd/nmon
```

## Run (opt-in local + k8s remote data)

Runs `nmon` locally, but reads `/data/metrics` and `/data/experiments.db` from a cluster pod via the Kubernetes API (exec).

```bash
cd /workspace/nmoe/tools/nmon

go run ./cmd/nmon --k8s \
  --k8s-namespace=default \
  --k8s-selector='app=nmoe,stage=debug' \
  --metrics-dir=/data/metrics \
  --experiments-db=/data/experiments.db
```

Notes:
- If neither `--k8s-pod` nor `--k8s-selector` is set, `nmon` picks the first Running pod in the namespace.
- The target container must have `python3` with the `duckdb` module available (for metrics queries). SQLite queries use Python's `sqlite3`.
