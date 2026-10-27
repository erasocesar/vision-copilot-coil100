-- Base operativa de la demo (en Azure: PostgreSQL Flexible Server).
-- Todas las tablas de línea llevan plant_id y line_id: el aislamiento se aplica en cada consulta.

CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    ts TEXT NOT NULL, shift INTEGER NOT NULL,
    image_id TEXT NOT NULL,
    obj_id_true INTEGER, angle_true INTEGER,
    obj_id_pred INTEGER, class_prob REAL, ood_distance REAL,
    angle_pred REAL, angle_std REAL, angle_mode TEXT, pick_score REAL,
    action TEXT NOT NULL, reasons TEXT NOT NULL DEFAULT '[]',
    model_version TEXT NOT NULL, camera_blur REAL
);
CREATE INDEX IF NOT EXISTS ix_insp ON inspections(plant_id, line_id, ts);

CREATE TABLE IF NOT EXISTS camera_health (
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL, ts TEXT NOT NULL,
    blur REAL, brightness REAL, dropped_frames INTEGER, temperature_c REAL
);
CREATE INDEX IF NOT EXISTS ix_cam ON camera_health(plant_id, line_id, ts);

CREATE TABLE IF NOT EXISTS robot_commands (
    id INTEGER PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL, ts TEXT NOT NULL,
    inspection_id INTEGER, action TEXT NOT NULL, obj_id INTEGER, angle REAL, angle_mode TEXT,
    source TEXT NOT NULL DEFAULT 'policy', approval_id TEXT
);

CREATE TABLE IF NOT EXISTS grasp_events (
    id INTEGER PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL, ts TEXT NOT NULL,
    inspection_id INTEGER, success INTEGER NOT NULL, failure_code TEXT
);

CREATE TABLE IF NOT EXISTS review_queue (
    case_id TEXT PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    inspection_id INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', created_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    case_id TEXT PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    obj_id INTEGER NOT NULL, angle INTEGER NOT NULL, labeled_by TEXT NOT NULL, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrain_dataset (
    image_id TEXT PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    obj_id INTEGER NOT NULL, angle INTEGER NOT NULL, source TEXT NOT NULL, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    title TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
    body TEXT NOT NULL, related TEXT NOT NULL DEFAULT '[]', created_by TEXT NOT NULL, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    op TEXT NOT NULL, params_json TEXT NOT NULL, justification TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    requested_by TEXT NOT NULL, requested_role TEXT NOT NULL,
    approved_by TEXT, approved_role TEXT, ts_requested TEXT NOT NULL, ts_decided TEXT, result_json TEXT
);

CREATE TABLE IF NOT EXISTS line_state (
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    paused INTEGER NOT NULL DEFAULT 0, min_pick_score REAL NOT NULL, model_version TEXT NOT NULL,
    PRIMARY KEY (plant_id, line_id)
);

CREATE TABLE IF NOT EXISTS sku_candidates (
    candidate_id TEXT PRIMARY KEY,
    plant_id TEXT NOT NULL, line_id TEXT NOT NULL,
    name TEXT NOT NULL, image_ids TEXT NOT NULL, angles TEXT NOT NULL,
    n_views INTEGER NOT NULL, max_gap_deg INTEGER NOT NULL,
    status TEXT NOT NULL, metrics_json TEXT, created_by TEXT NOT NULL, ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY, stage TEXT NOT NULL, metrics_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL, session_id TEXT, user_id TEXT, role TEXT, plant_id TEXT, line_id TEXT,
    tool TEXT NOT NULL, args_sha256 TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL, session_id TEXT, model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL, cached_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, usd REAL NOT NULL
);
