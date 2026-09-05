#!/usr/bin/env node
// Assert every shipped sidecar module can resolve its local imports.
//
// Run INSIDE the built antseed image (CI mounts this file in):
//     docker run --rm -v "$PWD/scripts/check_sidecar_modules.js:/tmp/c.js" IMG node /tmp/c.js
//
// Why it has to run in the image rather than the repo: the repo is always
// self-consistent, so the Python and Node suites pass whether or not a file
// reaches the image. control.js gained `require('./ids.js')` while
// Dockerfile.antseed still named its COPY files individually; the build
// succeeded, every test stayed green, and the sidecar's control server died at
// import in production ("Cannot find module './ids.js'") with :8379 never
// binding and every wallet endpoint returning 502. The only place that
// divergence is observable is the built artifact.
//
// Resolution-only: never executes a module, so a checker run cannot start the
// control server or touch a wallet.
const fs = require("fs");
const path = require("path");

const DIR = process.argv[2] || "/usr/local/lib/antseed";
const LOCAL_IMPORT = /(?:require\(\s*|from\s+)['"](\.\/[^'"]+)['"]/g;

const shipped = fs.readdirSync(DIR)
    .filter((f) => (f.endsWith(".js") || f.endsWith(".mjs")) && !f.endsWith(".test.js"));

if (shipped.length === 0) {
    console.error(`no shipped modules found in ${DIR} — wrong path, or the COPY is broken`);
    process.exit(1);
}

const problems = [];
for (const name of shipped) {
    const file = path.join(DIR, name);
    const src = fs.readFileSync(file, "utf8");
    for (const m of src.matchAll(LOCAL_IMPORT)) {
        const target = path.join(DIR, m[1]);
        // .mjs imports carry the extension; require() may omit it. Accept either.
        const candidates = path.extname(target) ? [target] : [target + ".js", target + ".mjs", target];
        if (!candidates.some((c) => fs.existsSync(c))) {
            problems.push(`${name} imports ${m[1]} — not present in the image`);
        }
    }
}

if (problems.length) {
    console.error("sidecar image is missing modules its own code requires:");
    for (const p of problems) console.error("  " + p);
    process.exit(1);
}
console.log(`ok — ${shipped.length} shipped modules, all local imports resolve`);
