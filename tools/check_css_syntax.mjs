/* CSS syntax gate: parse every stylesheet with css-tree (strict) and fail on
   any parse error.  Catches the class of bug the regex-based frontend check
   cannot see (unbalanced braces, a bad `linear()` list, a stray colon).
   Run: node tools/check_css_syntax.mjs                                     */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
/* css-tree ships with jsdom; the smoke test already relies on the same temp
   install, so look there as a fallback instead of adding a dependency just for
   a syntax gate. */
function loadCssTree() {
  const candidates = ['css-tree'];
  if (process.env.TEMP) candidates.push(join(process.env.TEMP, 'lsart-jsdom', 'node_modules', 'css-tree'));
  for (const id of candidates) {
    try { return require(id); } catch { /* try next */ }
  }
  return null;
}

const csstree = loadCssTree();
if (!csstree) {
  console.log('SKIP: css-tree not installed (npm i -g css-tree, or install jsdom into %TEMP%\\lsart-jsdom)');
  process.exit(0);
}

const SITE = 'site';
const files = ['styles.css', 'extra.css']
  .concat(readdirSync(join(SITE, 'css')).filter((f) => f.endsWith('.css')).map((f) => join('css', f)));

let errors = 0;
for (const rel of files) {
  const path = join(SITE, rel);
  const css = readFileSync(path, 'utf8');
  const problems = [];
  csstree.parse(css, {
    positions: true,
    onParseError: (e) => problems.push(`${e.line}:${e.column} ${e.message}`),
  });
  if (problems.length) {
    errors += problems.length;
    console.log(`FAIL ${rel}`);
    problems.slice(0, 8).forEach((p) => console.log(`   ${p}`));
  } else {
    console.log(`ok   ${rel}`);
  }
}
console.log(errors ? `CSS SYNTAX FAILED (${errors} errors)` : 'CSS syntax OK');
process.exit(errors ? 1 : 0);
