// Guards pgConfig() against the prod outage where the antseed sidecar logged
// `getaddrinfo ENOTFOUND base`: infra composes DATABASE_URL as a libpq kv
// conninfo (because the RDS password has URL-delimiter chars), psycopg accepts
// it, but node-postgres only parses postgres:// URLs and fell back to a bogus
// host. compose/dev never caught it because compose feeds a postgres:// URL —
// the two environments hand the sidecar DIFFERENT formats. This test exercises
// BOTH so the kv (prod) path can't silently break again.
//
// Run: node --test antseed/db.test.js
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { pgConfig, parseKvConninfo } = require("./db.js");

// The exact shape infra's external-secrets composes, with a password full of
// URL-delimiter chars (@ : / ? # [ ]) single-quoted — the reason kv was chosen.
const PROD_KV =
  "host=llm-router-dev.abc123.us-east-1.rds.amazonaws.com port=5432 " +
  "dbname=hoststore user=hoststore password='p@s:s/w?r#d[1]' sslmode=require";

test("kv conninfo -> explicit pg fields (the prod form that broke)", () => {
  const cfg = pgConfig(PROD_KV);
  assert.equal(cfg.host, "llm-router-dev.abc123.us-east-1.rds.amazonaws.com");
  assert.equal(cfg.port, 5432);
  assert.equal(cfg.database, "hoststore");
  assert.equal(cfg.user, "hoststore");
  assert.equal(cfg.password, "p@s:s/w?r#d[1]");        // special chars intact
  assert.deepEqual(cfg.ssl, { rejectUnauthorized: false }); // sslmode=require
  // The regression: it must NOT be handed back as a connectionString — that is
  // exactly what made node-postgres fall back to host "base".
  assert.equal(cfg.connectionString, undefined);
});

test("postgres:// URL passes straight through (the compose/dev form)", () => {
  const url = "postgresql://hoststore:hoststore@postgres:5432/hoststore";
  assert.deepEqual(pgConfig(url), { connectionString: url });
});

test("empty / missing DATABASE_URL -> empty config (no crash)", () => {
  assert.deepEqual(pgConfig(""), {});
  assert.deepEqual(pgConfig(undefined), {});
});

test("sslmode variants map to the right ssl", () => {
  assert.equal(pgConfig("host=h sslmode=disable").ssl, false);
  assert.deepEqual(pgConfig("host=h sslmode=verify-full").ssl, { rejectUnauthorized: true });
  assert.equal(pgConfig("host=h").ssl, undefined);     // unset -> let pg decide
});

test("parseKvConninfo handles unquoted values and quote escapes", () => {
  assert.deepEqual(parseKvConninfo("host=h port=5432 user=u"),
    { host: "h", port: "5432", user: "u" });
  // a single-quoted value may contain an escaped quote
  assert.equal(parseKvConninfo("password='a\\'b'").password, "a'b");
});
