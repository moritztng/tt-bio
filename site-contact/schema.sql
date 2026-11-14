-- One row per accepted submission. The only place the submitter's IP is kept.
CREATE TABLE IF NOT EXISTS messages (
  ref     TEXT PRIMARY KEY,
  at      INTEGER NOT NULL,  -- unix ms
  name    TEXT NOT NULL,
  email   TEXT NOT NULL,
  org     TEXT NOT NULL,
  message TEXT NOT NULL,
  ip      TEXT NOT NULL,
  mail    TEXT,              -- "ok" or the failure, written once the leg settles
  notify  TEXT
);
CREATE INDEX IF NOT EXISTS messages_ip_at ON messages (ip, at);
CREATE INDEX IF NOT EXISTS messages_at ON messages (at);
