// Server-only metrics readers backed by DuckDB reading parquet files.
//
// nmoe writes per-step parquet files (rank 0 only):
//   /data/metrics/{run_id}/step_XXXXXXXX.parquet
//
// NVIZ queries them via DuckDB's read_parquet glob for real-time access
// (no file locking issues unlike attached duckdb files).
import { readdirSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { DuckDBInstance } from '@duckdb/node-api'

export type Row = { run: string; tag: string; step: number; ts_ms: number; value: number }
export type RunSummary = { run: string; last_ts: number; last_step: number }

function sqlLit(s: string): string {
  return `'${s.replaceAll("'", "''")}'`
}

export function metricsDir(): string {
  const dir = process.env.NVIZ_METRICS_DIR
  if (!dir) throw new Error('NVIZ_METRICS_DIR must be set')
  return dir
}

function safeRunDir(metricsRoot: string, run: string): string {
  // `run` is user-supplied via query params. Keep it strictly as a directory name.
  if (!run) throw new Error("Missing 'run'")
  if (!/^[A-Za-z0-9._-]+$/.test(run)) throw new Error('Invalid run')
  const root = resolve(metricsRoot)
  const target = resolve(join(root, run))
  if (target !== root && !target.startsWith(root + '/')) {
    throw new Error('Invalid run')
  }
  return target
}

function listRunDirs(dir: string): string[] {
  try {
    return readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
  } catch {
    return []
  }
}

function hasParquetFiles(dir: string, run: string): boolean {
  try {
    const runDir = safeRunDir(dir, run)
    const files = readdirSync(runDir, { withFileTypes: true })
    return files.some((e) => e.isFile() && e.name.startsWith('step_') && e.name.endsWith('.parquet'))
  } catch {
    return false
  }
}

function newestParquetMtime(dir: string, run: string): number {
  try {
    const runDir = safeRunDir(dir, run)
    const files = readdirSync(runDir, { withFileTypes: true })
      .filter((e) => e.isFile() && e.name.startsWith('step_') && e.name.endsWith('.parquet'))
    let newest = 0
    for (const f of files) {
      try {
        const st = statSync(join(runDir, f.name))
        if (st.mtimeMs > newest) newest = st.mtimeMs
      } catch {}
    }
    return newest
  } catch {
    return 0
  }
}

function parquetGlob(dir: string, run: string): string {
  const runDir = safeRunDir(dir, run)
  return join(runDir, 'step_*.parquet')
}

async function withParquetView<T>(run: string, fn: (q: (sql: string) => Promise<any[]>) => Promise<T>): Promise<T> {
  const dir = metricsDir()
  if (!hasParquetFiles(dir, run)) return await fn(async () => [])

  const inst = await DuckDBInstance.create(':memory:')
  const conn = await inst.connect()
  try {
    const glob = parquetGlob(dir, run)
    await conn.run(`CREATE VIEW metrics AS SELECT * FROM read_parquet(${sqlLit(glob)}, union_by_name=true, filename=false)`)

    const q = async (sql: string): Promise<any[]> => {
      const reader = await conn.runAndReadAll(sql)
      const cols = reader.columnNames()
      const rows = reader.getRows()
      return rows.map((r: any[]) => Object.fromEntries(cols.map((c, i) => [c, r[i]])))
    }

    return await fn(q)
  } finally {
    try { (conn as any).close?.() } catch {}
    try { (inst as any).close?.() } catch {}
  }
}

let allRunsCache: { at_ms: number; runs: RunSummary[] } | null = null

export async function allRuns(): Promise<RunSummary[]> {
  const now = Date.now()
  if (allRunsCache && (now - allRunsCache.at_ms) < 2000) {
    return allRunsCache.runs
  }

  const dir = metricsDir()
  const runs = listRunDirs(dir)
  const out: RunSummary[] = []
  for (const run of runs) {
    if (!hasParquetFiles(dir, run)) continue
    const inst = await DuckDBInstance.create(':memory:')
    const conn = await inst.connect()
    try {
      const glob = parquetGlob(dir, run)
      await conn.run(`CREATE VIEW metrics AS SELECT * FROM read_parquet(${sqlLit(glob)}, union_by_name=true, filename=false)`)
      const reader = await conn.runAndReadAll(`SELECT max(ts_ms) AS last_ts, max(step) AS last_step FROM metrics`)
      const rows = reader.getRows()
      const last_ts = Number(rows?.[0]?.[0] ?? 0)
      const last_step = Number(rows?.[0]?.[1] ?? 0)
      out.push({ run, last_ts, last_step })
    } finally {
      try { (conn as any).close?.() } catch {}
      try { (inst as any).close?.() } catch {}
    }
  }
  out.sort((a, b) => b.last_ts - a.last_ts || b.last_step - a.last_step || a.run.localeCompare(b.run))
  allRunsCache = { at_ms: now, runs: out }
  return out
}

export async function latestForTags(run: string, tags: string[]): Promise<Record<string, number>> {
  if (tags.length === 0) return {}
  return await withParquetView(run, async (q) => {
    const inList = tags.map(sqlLit).join(',')
    const rows = await q(`
      SELECT tag, value
      FROM metrics
      WHERE run = ${sqlLit(run)} AND tag IN (${inList})
      QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1
    `)
    const out: Record<string, number> = {}
    for (const r of rows as any[]) out[String(r.tag)] = Number(r.value)
    return out
  })
}

export async function latestForPrefixes(run: string, prefixes: string[]): Promise<Record<string, number>> {
  if (prefixes.length === 0) return {}
  return await withParquetView(run, async (q) => {
    const where = prefixes.map((p) => `tag LIKE ${sqlLit(p + '%')}`).join(' OR ')
    const rows = await q(`
      SELECT tag, value
      FROM metrics
      WHERE run = ${sqlLit(run)} AND (${where})
      QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1
    `)
    const out: Record<string, number> = {}
    for (const r of rows as any[]) out[String(r.tag)] = Number(r.value)
    return out
  })
}

export async function scalarsSampled(run: string, tag: string, buckets: number): Promise<Array<{ step: number; ts_ms: number; value: number }>> {
  const n = Math.max(8, Math.min(buckets, 20000))
  return await withParquetView(run, async (q) => {
    const rows = await q(`
      WITH dedup AS (
        SELECT step, ts_ms, value
        FROM metrics
        WHERE run = ${sqlLit(run)} AND tag = ${sqlLit(tag)}
        QUALIFY row_number() OVER (PARTITION BY step ORDER BY ts_ms DESC) = 1
      ),
      bucketed AS (
        SELECT step, ts_ms, value,
               ntile(${n}) OVER (ORDER BY step) AS b
        FROM dedup
      )
      SELECT step, ts_ms, value
      FROM bucketed
      QUALIFY
        row_number() OVER (PARTITION BY b ORDER BY step ASC) = 1
        OR row_number() OVER (PARTITION BY b ORDER BY step DESC) = 1
      ORDER BY step ASC
    `)
    return (rows as any[]).map((r) => ({ step: Number(r.step), ts_ms: Number(r.ts_ms), value: Number(r.value) }))
  })
}
