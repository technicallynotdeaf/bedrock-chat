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

const TARGET_DIR = path.join(
  __dirname,
  '..',
  'node_modules',
  '@aws-amplify',
  'ui',
  'dist',
  'esm'
);

if (!fs.existsSync(TARGET_DIR)) {
  console.log('[fix-amplify-xstate] @aws-amplify/ui ESM dir not found — skipping.');
  process.exit(0);
}

let patchedCount = 0;

function patchDir(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      patchDir(fullPath);
    } else if (entry.isFile() && entry.name.endsWith('.mjs')) {
      const original = fs.readFileSync(fullPath, 'utf8');
      // Replace bare `'xstate'` specifier but not `'xstate4'` (idempotent)
      const patched = original.replace(/from 'xstate'(?!')/g, "from 'xstate4'");
      if (patched !== original) {
        fs.writeFileSync(fullPath, patched, 'utf8');
        patchedCount++;
        console.log(`[fix-amplify-xstate] Patched: ${path.relative(process.cwd(), fullPath)}`);
      }
    }
  }
}

patchDir(TARGET_DIR);

if (patchedCount > 0) {
  console.log(`[fix-amplify-xstate] Done — patched ${patchedCount} file(s).`);
} else {
  console.log('[fix-amplify-xstate] No files needed patching.');
}
