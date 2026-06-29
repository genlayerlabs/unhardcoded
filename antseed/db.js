// node-postgres connection config from DATABASE_URL — accepting the libpq
// keyword/value conninfo form, not just a URL.
//
// The shared `llm-router-db` secret is composed by infra as a libpq kv conninfo
// (`host=... port=... dbname=... user=... password='...' sslmode=require`), NOT
// a postgresql:// URL — deliberately, because the RDS password contains
// URL-delimiter characters (@ : / ? # [ ]) that a URL DSN would need
// percent-encoded. psycopg (router/ingress) accepts the kv form; node-postgres
// does NOT — handed a kv string it finds no host and falls back to a bogus one
// (`getaddrinfo ENOTFOUND base`), so the sidecar never reaches the store. We
// therefore parse the kv conninfo into explicit pg fields here. A real
// postgres:// URL passes straight through, so both forms work.
"use strict";

// libpq conninfo: space-separated key=value; a value may be single-quoted to
// hold spaces/specials, with \' and \\ escapes honoured inside the quotes.
function parseKvConninfo(s) {
  const out = {};
  const n = s.length;
  let i = 0;
  while (i < n) {
    while (i < n && /\s/.test(s[i])) i++;                 // skip inter-pair space
    if (i >= n) break;
    let key = "";
    while (i < n && s[i] !== "=" && !/\s/.test(s[i])) key += s[i++];
    while (i < n && /\s/.test(s[i])) i++;
    if (s[i] !== "=") continue;                           // malformed pair, skip
    i++;                                                  // consume '='
    while (i < n && /\s/.test(s[i])) i++;
    let val = "";
    if (s[i] === "'") {                                   // single-quoted value
      i++;
      while (i < n && s[i] !== "'") {
        if (s[i] === "\\" && i + 1 < n) i++;              // escape: take next char
        val += s[i++];
      }
      i++;                                                // consume closing quote
    } else {
      while (i < n && !/\s/.test(s[i])) val += s[i++];
    }
    if (key) out[key] = val;
  }
  return out;
}

// libpq sslmode -> node-postgres `ssl`. `require` means encrypt but do NOT
// verify the CA (what psycopg does here); `verify-*` means verify.
function sslFor(sslmode) {
  switch ((sslmode || "").toLowerCase()) {
    case "disable": return false;
    case "allow":
    case "prefer":
    case "require": return { rejectUnauthorized: false };
    case "verify-ca":
    case "verify-full": return { rejectUnauthorized: true };
    default: return undefined;                            // unset -> let pg decide
  }
}

// DATABASE_URL -> a pg Client/Pool config. A postgres:// URL is passed through
// as a connectionString; a libpq kv conninfo is parsed into explicit fields.
function pgConfig(databaseUrl) {
  const s = (databaseUrl || "").trim();
  if (!s) return {};
  if (/^postgres(ql)?:\/\//i.test(s)) return { connectionString: s };
  const kv = parseKvConninfo(s);
  const cfg = {};
  if (kv.host) cfg.host = kv.host;
  else if (kv.hostaddr) cfg.host = kv.hostaddr;
  if (kv.port) cfg.port = Number(kv.port);
  if (kv.dbname) cfg.database = kv.dbname;
  if (kv.user) cfg.user = kv.user;
  if (kv.password !== undefined) cfg.password = kv.password;
  const ssl = sslFor(kv.sslmode);
  if (ssl !== undefined) cfg.ssl = ssl;
  return cfg;
}

module.exports = { pgConfig, parseKvConninfo };
