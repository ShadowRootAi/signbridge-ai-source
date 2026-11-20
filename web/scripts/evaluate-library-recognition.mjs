import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const [targetUrl, outputPath, ...options] = process.argv.slice(2);
assert(targetUrl && outputPath,
  'Usage: node scripts/evaluate-library-recognition.mjs <target> <output> [policy.json] [--manifest list.json --compare <target> --pages 2 --screenshot-dir <folder>]');
const output = resolve(outputPath);
const policyOverridePath = options.length && !options[0].startsWith('--') ? options.shift() : null;
assert(options.length % 2 === 0, 'Options require a value');
const flags = new Map();
for (let index = 0; index < options.length; index += 2) {
  const [name, value] = options.slice(index, index + 2);
  assert(['--manifest', '--compare', '--pages', '--screenshot-dir'].includes(name) && !flags.has(name), 'Unknown or repeated option ' + name);
  flags.set(name, value);
}
const pages = Number(flags.get('--pages') ?? 1);
assert(Number.isInteger(pages) && pages >= 1 && pages <= 3, '--pages must be 1, 2 or 3');
const screenshotDir = flags.has('--screenshot-dir') ? resolve(flags.get('--screenshot-dir')) : null;
if (screenshotDir) {
  assert(flags.has('--manifest') && !flags.has('--compare'), 'Screenshots require a single target and explicit manifest');
  await mkdir(screenshotDir, { recursive: true });
}
const policyOverride = policyOverridePath ? await readFile(resolve(policyOverridePath), 'utf8') : null;
const normalize = value => value.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase('el-GR').replace(/ς/g, 'σ').trim();
const outcomes = ['correct', 'wrong', 'abstained', 'rejected', 'false_accept'];
const summarize = rows => Object.fromEntries(outcomes.map(outcome => [outcome, rows.filter(row => row.outcome === outcome).length]));
const isGloss = gloss => typeof gloss === 'string' && gloss.trim().length > 0;

async function selectClips() {
  const manifestPath = flags.get('--manifest');
  if (manifestPath) {
    const manifestFile = resolve(manifestPath);
    const manifestBytes = await readFile(manifestFile);
    const manifest = JSON.parse(manifestBytes.toString('utf8'));
    const clips = Array.isArray(manifest) ? manifest : manifest.clips ?? manifest.samples;
    assert(Array.isArray(clips) && clips.length > 0, 'Manifest must list one or more clips');
    const seen = new Set();
    for (const clip of clips) {
      assert(typeof clip.id === 'string' && clip.id.trim() && !seen.has(clip.id), 'Missing or repeated clip ID');
      assert(isGloss(clip.gloss) || clip.gloss === null, 'Require explicit gloss or null for ' + clip.id);
      assert(typeof clip.group === 'string' && clip.group.trim(), 'Missing group for ' + clip.id);
      assert(typeof clip.path === 'string' && clip.path.trim(), 'Missing path for ' + clip.id);
      if (clip.screenshot) assert(/^[a-z0-9][a-z0-9-]*$/.test(clip.id), 'Unsafe screenshot id ' + clip.id);
      seen.add(clip.id);
      clip.file = resolve(dirname(manifestFile), clip.path);
      if (clip.sha256) {
        const digest = createHash('sha256').update(await readFile(clip.file)).digest('hex');
        assert.equal(digest, clip.sha256, 'Manifest SHA-256 mismatch for ' + clip.id);
      }
    }
    return { clips, scope: 'Explicit local evaluation manifest through Chrome and the app file path',
      manifestScope: typeof manifest.scope === 'string' ? manifest.scope : undefined,
      manifestSha256: createHash('sha256').update(manifestBytes).digest('hex') };
  }
  const library = resolve(project, 'web/public/library');
  const { entries } = JSON.parse(await readFile(resolve(library, 'catalog.json'), 'utf8'));
  const isolated = entries.filter(entry => entry.contentKind !== 'phrase');
  const annotated = isolated.filter(entry => isGloss(entry.annotationLabel));
  assert(annotated.length > 0, 'No isolated clips with explicit recognition annotations');
  const clips = annotated.map(({ id, phrase, annotationLabel }) => ({
    id, phrase, gloss: annotationLabel, file: resolve(library, id + '.mp4'),
  }));
  return { clips, scope: 'Explicitly gloss-annotated isolated GSL MP4 files through Chrome and the app file path',
    excludedUnmapped: isolated.length - annotated.length, excludedPhrases: entries.length - isolated.length };
}

async function observeRecognition(page) {
  await page.addInitScript(() => {
    const BrowserWorker = window.Worker;
    window.Worker = class extends BrowserWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (Array.isArray(data?.result?.topCandidates)) window.signBridgeAuditDecision = data.result;
          if (typeof data?.error === 'string') window.signBridgeAuditError = data.error;
        });
      }
    };
  });
}

async function auditTarget(browser, target, selection, override) {
  const origin = target.origin;
  const policyUrl = origin + '/__signbridge/openhands/policy.json';
  const response = await fetch(policyUrl);
  assert.equal(response.status, 200, 'OpenHands policy from ' + origin);
  const policy = override ? JSON.parse(override) : await response.json();
  const supported = new Set(policy.supportedLabels);
  const { clips, ...selectionInfo } = selection;
  const report = { origin, decisionGate: policy.decisionGate ?? null, agreementGate: policy.agreementGate ?? null, ...selectionInfo,
    pages: Math.min(pages, clips.length), rows: new Array(clips.length) };
  const context = await browser.newContext(screenshotDir ? { viewport: { width: 1600, height: 900 } } : {});
  try {
    let policyRequests = 0;
    if (override) {
      await context.route(policyUrl, route => {
        policyRequests++;
        return route.fulfill({ contentType: 'application/json', body: override });
      });
    }
    await Promise.all(Array.from({ length: report.pages }, async (_, workerIndex) => {
      const page = await context.newPage();
      await observeRecognition(page);
      await page.goto(origin + '/' + target.search + '#/try');
      const video = page.locator('#recognition-video');
      const result = page.locator('#recognition-result');
      const error = page.locator('.recognition-error');
      for (let index = workerIndex; index < clips.length; index += report.pages) {
        const { file, ...clip } = clips[index];
        const previous = await video.evaluate(element => element.currentSrc);
        await page.locator('#video-file').setInputFiles(file);
        await page.waitForFunction(before => {
          const video = document.querySelector('#recognition-video');
          return video?.currentSrc.startsWith('blob:') && video.currentSrc !== before && video.readyState >= 2;
        }, previous);
        await page.evaluate(() => { window.signBridgeAuditDecision = null; window.signBridgeAuditError = null; });
        const start = performance.now();
        await page.getByRole('button', { name: 'Ανάλυση βίντεο', exact: true }).click();
        await result.or(error).waitFor({ timeout: 70_000 });
        const elapsedMs = Math.round(performance.now() - start);
        assert.equal(await error.count(), 0, `${clip.id}: ${await page.evaluate(() => window.signBridgeAuditError) ??
          (await error.allTextContents()).join('\n')}`);
        const text = (await result.innerText()).trim();
        const accepted = await result.getAttribute('data-accepted') === 'true';
        const diagnostic = await page.evaluate(() => window.signBridgeAuditDecision);
        assert(diagnostic, 'Missing recognizer diagnostic for ' + clip.id);
        assert.equal(diagnostic.accepted, accepted, 'Worker/UI acceptance mismatch for ' + clip.id);
        const inScope = supported.has(clip.gloss);
        if (typeof clip.on_roster === 'boolean') assert.equal(inScope, clip.on_roster, 'Roster mismatch for ' + clip.id);
        const expected = policy.displayLabels?.[clip.gloss] ?? clip.gloss;
        const outcome = accepted
          ? (inScope ? normalize(expected) === normalize(text) ? 'correct' : 'wrong' : 'false_accept')
          : (inScope ? 'abstained' : 'rejected');
        const cold = index === workerIndex;
        report.rows[index] = { ...clip, inScope, text, outcome, elapsedMs, cold, diagnostic };
        if (screenshotDir && clip.screenshot) {
          const screenshot = `${clip.id}.png`;
          await page.screenshot({ path: resolve(screenshotDir, screenshot), animations: 'disabled' });
          report.rows[index].screenshot = screenshot;
        }
        console.log(JSON.stringify({ origin, id: clip.id, gloss: clip.gloss, group: clip.group,
          outcome, elapsedMs, acceptedBy: diagnostic.acceptedBy, rejectionCode: diagnostic.rejectionCode,
          candidate: diagnostic.topCandidates[0] }));
      }
    }));
    if (override) assert(policyRequests >= 2, 'Expected candidate policy in both the app and model worker');
  } finally {
    await context.close();
  }
  const versions = new Set(report.rows.map(row => row.diagnostic.modelVersion));
  assert.equal(versions.size, 1, 'Mixed model versions for ' + origin);
  report.modelVersion = [...versions][0];
  report.summary = summarize(report.rows);
  const groups = new Set(report.rows.map(row => row.group).filter(Boolean));
  if (groups.size) report.byGroup = Object.fromEntries([...groups].map(group => [
    group, summarize(report.rows.filter(row => row.group === group)),
  ]));
  return report;
}

const selection = await selectClips();
const browser = await chromium.launch({ channel: 'chrome', headless: true });
let report;
try {
  const original = await auditTarget(browser, new URL(targetUrl), selection, policyOverride);
  const candidateUrl = flags.get('--compare');
  if (!candidateUrl) {
    report = { browser: browser.version(), ...original };
  } else {
    const candidate = await auditTarget(browser, new URL(candidateUrl), selection, null);
    const matched = original.rows.map((row, index) => {
      const next = candidate.rows[index];
      assert.equal(row.id, next.id, 'Matched audit clip mismatch');
      assert.equal(row.inScope, next.inScope, 'Compared roster mismatch for ' + row.id);
      return { id: row.id, group: row.group, original: row.outcome, candidate: next.outcome,
        acceptedBy: next.diagnostic.acceptedBy };
    });
    report = { browser: browser.version(), original, candidate, matched };
  }
} finally {
  await browser.close();
}
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, original: report.original?.summary ?? report.summary, candidate: report.candidate?.summary }));
