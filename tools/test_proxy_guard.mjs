/* Unit test for the proxy's cross-site guard (tools/test_proxy_guard.mjs).
 * Stubs fetch so no backend is needed; asserts which callers may burn GPU time.
 *   node tools/test_proxy_guard.mjs
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.env.BACKEND_URL = 'https://backend.test';
process.env.API_TOKEN = 'test-token';

let upstreamCalls = 0;
globalThis.fetch = async () => {
  upstreamCalls += 1;
  return new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'Content-Type': 'application/json' } });
};

const require = createRequire(import.meta.url);
const { handler } = require(path.join(ROOT, 'netlify', 'functions', 'proxy.js'));

const event = (op, headers = {}) => ({
  queryStringParameters: { op }, headers, body: op === 'generate' ? '{"prompt":"x"}' : null,
});

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail });

const sameOrigin = await handler(event('generate', { origin: 'https://landscape-art-demo.netlify.app', 'sec-fetch-site': 'same-origin' }));
check('same-origin browser POST is allowed', sameOrigin.statusCode === 200, `got ${sameOrigin.statusCode}`);

const previewDeploy = await handler(event('generate', { origin: 'https://6aa42c9606b9dd4dffb76a54--landscape-art-demo.netlify.app' }));
check('netlify deploy-preview origin is allowed', previewDeploy.statusCode === 200, `got ${previewDeploy.statusCode}`);

const localhost = await handler(event('generate', { origin: 'http://localhost:5500' }));
check('localhost origin is allowed', localhost.statusCode === 200, `got ${localhost.statusCode}`);

const evil = await handler(event('generate', { origin: 'https://evil.example.com', 'sec-fetch-site': 'cross-site' }));
check('cross-site origin is blocked', evil.statusCode === 403, `got ${evil.statusCode}`);

const evilReferer = await handler(event('generate', { referer: 'https://evil.example.com/steal.html' }));
check('cross-site referer is blocked', evilReferer.statusCode === 403, `got ${evilReferer.statusCode}`);

const scripted = await handler(event('generate', {}));
check('script with no origin header still works (non-strict default)', scripted.statusCode === 200, `got ${scripted.statusCode}`);

const readOp = await handler(event('job', { origin: 'https://evil.example.com', 'sec-fetch-site': 'cross-site' }));
check('read-only op is never origin-gated', readOp.statusCode === 200, `got ${readOp.statusCode}`);

const allowed = await handler(event('generate', { origin: 'https://my.other.site' }));
check('unlisted origin is blocked by default', allowed.statusCode === 403, `got ${allowed.statusCode}`);
process.env.ALLOWED_ORIGINS = 'https://my.other.site';
delete require.cache[require.resolve(path.join(ROOT, 'netlify', 'functions', 'proxy.js'))];
const { handler: handler2 } = require(path.join(ROOT, 'netlify', 'functions', 'proxy.js'));
const allowed2 = await handler2(event('generate', { origin: 'https://my.other.site' }));
check('ALLOWED_ORIGINS admits an extra origin', allowed2.statusCode === 200, `got ${allowed2.statusCode}`);

// strict mode: headerless scripted callers are rejected too
process.env.PROXY_STRICT = '1';
delete require.cache[require.resolve(path.join(ROOT, 'netlify', 'functions', 'proxy.js'))];
const { handler: handler3 } = require(path.join(ROOT, 'netlify', 'functions', 'proxy.js'));
const strict = await handler3(event('generate', {}));
check('PROXY_STRICT=1 rejects headerless callers', strict.statusCode === 403, `got ${strict.statusCode}`);

let failed = 0;
for (const { name, ok, detail } of results) {
  if (!ok) failed += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `  [${detail}]`}`);
}
console.log(`\n${results.length - failed}/${results.length} checks passed (upstream calls: ${upstreamCalls})`);
process.exit(failed ? 1 : 0);
