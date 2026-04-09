import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';

// @aws-amplify/ui-react-core uses xstate v4 (`{ actions }` export), while the
// project's streaming state machine needs xstate v5 (`setup` / `assign` API).
// `xstate4` is an npm alias ("xstate4": "npm:xstate@4.38.3") installed as a
// direct dependency so it is always present after `npm ci`.
//
// Without intervention, the Amplify packages resolve to their own nested
// node_modules/xstate (v4) while the postinstall-patched files import from
// `xstate4` — creating TWO separate xstate v4 module instances in the bundle.
// When two copies exist, the Authenticator's xstate interpreter and state
// machine use different class identities, which silently breaks the auth state
// machine (sign-in and sign-out do nothing).
//
// Fix: resolve the `xstate4` alias to its absolute entry point once, then
// redirect every bare `xstate` import from any `@aws-amplify` (or nested
// `@xstate/react`) package to that single file path.  This guarantees ONE
// xstate v4 instance in the final bundle.
//
// IMPORTANT: This plugin must be in the top-level `plugins` array (not inside
// `build.rollupOptions.plugins`) so that it is active in ALL Rollup/Vite build
// environments, including the secondary build that vite-plugin-pwa runs for the
// service-worker injection bundle.
// Resolve the xstate4 ESM entry to a single absolute path.  All Amplify
// xstate imports will be rewritten to this path so only ONE copy ends up in
// the bundle.  Using the ESM entry (`es/index.js`) is critical — the CJS
// entry would be treated as a separate module by Vite.
const xstate4Esm = path.resolve(
  __dirname,
  'node_modules',
  'xstate4',
  'es',
  'index.js',
);
const fixAmplifyXstate = {
  name: 'fix-amplify-xstate',
  enforce: 'pre' as const,
  resolveId(source: string, importer: string | undefined) {
    if (source === 'xstate' && importer) {
      // Redirect xstate imports from any @aws-amplify package or any package
      // nested under @aws-amplify (e.g. @xstate/react inside ui-react-core)
      if (importer.includes(path.join('node_modules', '@aws-amplify'))) {
        return xstate4Esm;
      }
    }
    // Catch xstate4 bare specifier (from postinstall-patched files)
    if (source === 'xstate4') {
      return xstate4Esm;
    }
  },
};

// https://vitejs.dev/config/
export default defineConfig({
  resolve: { alias: { './runtimeConfig': './runtimeConfig.browser' } },
  plugins: [
    fixAmplifyXstate,
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      devOptions: {
        enabled: true,
      },
      injectRegister: 'auto',
      workbox: {
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
      },
      manifest: {
        name: 'AA Bedrock',
        short_name: 'AA Bedrock',
        description: 'AWS-native chatbot using Bedrock',
        start_url: '/index.html',
        display: 'standalone',
        theme_color: '#6C3F99',
        icons: [
          {
            src: '/images/bedrock_icon_72.png',
            sizes: '72x72',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_96.png',
            sizes: '96x96',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_128.png',
            sizes: '128x128',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_144.png',
            sizes: '144x144',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_152.png',
            sizes: '152x152',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_192.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_384.png',
            sizes: '384x384',
            type: 'image/png',
          },
          {
            src: '/images/bedrock_icon_512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
          {
            src: '/images/bedrock_icon_512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
        ],
      },
    }),
  ],
  server: { host: true },
});
