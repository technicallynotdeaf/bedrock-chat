/**
 * Postinstall patch: redirect xstate imports in @aws-amplify/ui to xstate4.
 *
 * @aws-amplify/ui was built against xstate v4 and uses APIs (e.g. `actions`)
 * that were removed in v5.  This project needs xstate v5 for its own state
 * machines, so both versions are installed side-by-side:
 *   - xstate  → v5 (used by the app's own code)
 *   - xstate4 → v4 alias (used by @aws-amplify/ui)
 *
 * The `fixAmplifyXstate` Rollup plugin in vite.config.ts handles this
 * redirection for Vite's own build pipeline, but it cannot intercept the
 * workbox-build Rollup pass (which has its own isolated nodeResolve instance).
 * Patching the source files directly is the only approach that works across
 * ALL bundlers / build contexts.
 *
 * This script is run automatically via `postinstall` in package.json.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const NODE_MODULES = path.join(__dirname, '..', 'node_modules');

let patchedCount = 0;

// Walk the entire node_modules tree and patch every .mjs file under any
// @aws-amplify/ui/dist/esm/ directory, regardless of nesting depth.
// npm may hoist @aws-amplify/ui to the root or nest it inside other packages
// depending on peer-dep resolution; we cover both cases.
function walk(dir) {
  if (!fs.existsSync(dir)) return;
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
  catch { return; }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const fullPath = path.join(dir, entry.name);

    if (entry.name === 'esm' && dir.endsWith(path.join('@aws-amplify', 'ui', 'dist'))) {
      // We are inside an @aws-amplify/ui/dist/esm — patch every .mjs here
      patchMjsFiles(fullPath);
    } else {
      walk(fullPath);
    }
  }
}

function patchMjsFiles(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      patchMjsFiles(fullPath);
    } else if (entry.isFile() && entry.name.endsWith('.mjs')) {
      const original = fs.readFileSync(fullPath, 'utf8');
      // Replace bare `'xstate'` specifier; leave `'xstate4'` untouched (idempotent)
      const patched = original.replace(/from 'xstate'(?!')/g, "from 'xstate4'");
      if (patched !== original) {
        fs.writeFileSync(fullPath, patched, 'utf8');
        patchedCount++;
        console.log(`[fix-amplify-xstate] Patched: ${path.relative(process.cwd(), fullPath)}`);
      }
    }
  }
}

walk(NODE_MODULES);

if (patchedCount > 0) {
  console.log(`[fix-amplify-xstate] Done — patched ${patchedCount} file(s).`);
} else {
  console.log('[fix-amplify-xstate] No files needed patching.');
}
