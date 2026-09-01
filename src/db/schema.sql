CREATE TABLE IF NOT EXISTS accounts (
  id                  INTEGER PRIMARY KEY,
  provider            TEXT    NOT NULL,
  provider_account_id TEXT    NOT NULL,
  trusted             INTEGER NOT NULL DEFAULT 0,
  auth_mode           TEXT    NOT NULL DEFAULT 'oauth',
  email               TEXT,
  label               TEXT,
  plan_type           TEXT,
  access_token        TEXT    NOT NULL,
  refresh_token       TEXT    NOT NULL,
  http_referer        TEXT,
  access_expires_at   INTEGER,
  last_refresh_at     INTEGER,
  status              TEXT    NOT NULL DEFAULT 'active',
  cooldown_until      INTEGER,
  disabled_reason     TEXT,
  created_at          INTEGER NOT NULL DEFAULT (unixepoch()),
  updated_at          INTEGER NOT NULL DEFAULT (unixepoch()),
  UNIQUE (provider, provider_account_id)
);

CREATE TABLE IF NOT EXISTS api_tokens (
  id             INTEGER PRIMARY KEY,
  user           TEXT    NOT NULL,
  token_hash     BLOB    NOT NULL UNIQUE,
  token_prefix   TEXT    NOT NULL,
  request_limit  INTEGER,
  token_limit    INTEGER,
  window_seconds INTEGER NOT NULL DEFAULT 3600,
  slowdown_ms    INTEGER NOT NULL DEFAULT 0,
  prefer_trusted INTEGER NOT NULL DEFAULT 0,
  created_at     INTEGER NOT NULL DEFAULT (unixepoch()),
  revoked_at     INTEGER
);

CREATE TABLE IF NOT EXISTS api_meter (
  id            INTEGER PRIMARY KEY,
  token_id      INTEGER NOT NULL REFERENCES api_tokens(id),
  ts_ms         INTEGER NOT NULL,
  input_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_api_meter_token_ts ON api_meter(token_id, ts_ms);

CREATE TABLE IF NOT EXISTS usage_log (
  id                INTEGER PRIMARY KEY,
  ts                INTEGER NOT NULL DEFAULT (unixepoch()),
  token_id          INTEGER REFERENCES api_tokens(id),
  user              TEXT    NOT NULL,
  account_id        INTEGER REFERENCES accounts(id),
  dialect           TEXT    NOT NULL,
  requested_model   TEXT    NOT NULL,
  upstream_model    TEXT    NOT NULL,
  effort            TEXT    NOT NULL DEFAULT '',
  service_tier      TEXT    NOT NULL DEFAULT '',
  input_tokens      INTEGER NOT NULL DEFAULT 0,
  output_tokens     INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens  INTEGER NOT NULL DEFAULT 0,
  cost_usd          REAL    NOT NULL DEFAULT 0,
  status            INTEGER NOT NULL,
  error_kind        TEXT,
  duration_ms       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_usage_ts         ON usage_log(ts);
CREATE INDEX IF NOT EXISTS idx_usage_user_ts    ON usage_log(user, ts);
CREATE INDEX IF NOT EXISTS idx_usage_account_ts ON usage_log(account_id, ts);
