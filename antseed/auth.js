// Shared-secret checks and bind addresses for the sidecar's HTTP listeners.
// Dependency-free so auth.test.js runs outside the sidecar image (control.js
// pulls in `pg`, which only exists inside it).
'use strict';
const { timingSafeEqual } = require('crypto');

// Constant-time: a `!==` compare leaks the token prefix through response timing
// to anyone who can reach the port. Length mismatch is not secret (it is fixed
// by the deployment), so it short-circuits.
function tokenMatches(presented, token) {
  if (typeof presented !== 'string' || typeof token !== 'string' || !token) return false;
  const a = Buffer.from(presented), b = Buffer.from(token);
  return a.length === b.length && timingSafeEqual(a, b);
}

// Pod-local by default: in k8s the router shares the pod and dials 127.0.0.1.
// Compose runs the sidecar as its own service and opts in to 0.0.0.0 explicitly.
function listenHost(env, name) {
  return (env && env[name]) || '127.0.0.1';
}

module.exports = { tokenMatches, listenHost };
