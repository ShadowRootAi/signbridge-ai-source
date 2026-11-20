import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const origin = new URL(process.argv[2] ?? 'http://127.0.0.1:5191').origin;
const output = resolve(process.argv[3] ?? resolve(project, 'reports/app-release/expanded-library-qa.json'));
const report = { passed: false, origin, catalogCount: 0, videos: [], playbackCount: 0, filtering: [], mobile: [], cameraCalls: 0,
  remoteRequests: [], writeRequests: [], pageErrors: [] };

const unaccented = text => text.normalize('NFD').replace(/\p{M}/gu, '');
const lookupKey = text => unaccented(text).toLocaleLowerCase('el-GR').replace(/ς/g, 'σ')
  .trim().replace(/[.!?;;·…]+$/u, '').trim().replace(/\s+/g, ' ');
const displayPhrase = phrase => phrase.charAt(0).toLocaleUpperCase('el-GR') + phrase.slice(1);
const searchableTerms = entry => [entry.phrase, ...entry.aliases];

function playbackSample(entries) {
  if (entries.length <= 24) return entries;
  const legacy = entries.filter(entry => ['geia', 'efcharisto', 'entaxei', 'poio-provlima'].includes(entry.id));
  const latest = ['word', 'phrase'].flatMap(kind => entries.findLast(entry => entry.contentKind === kind) ?? []);
  const pinned = [...new Set([...legacy, ...latest])];
  const others = entries.filter(entry => !pinned.includes(entry));
  const isolated = others.filter(entry => typeof entry.annotationLabel !== 'string');
  const candidates = isolated.length >= 8 ? isolated : others;
  return [...pinned, ...Array.from({ length: 8 }, (_, index) => candidates[Math.round(index * (candidates.length - 1) / 7)])];
}

let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.browser = browser.version();
  const context = await browser.newContext({ viewport: { width: 1440, height: 960 }, reducedMotion: 'reduce' });
  await context.addInitScript(() => {
    window.libraryQaCameraCalls = 0;
    const devices = navigator.mediaDevices;
    if (devices?.getUserMedia) {
      devices.getUserMedia = () => {
        window.libraryQaCameraCalls++;
        return Promise.reject(new DOMException('Library QA blocks camera access', 'NotAllowedError'));
      };
    }
  });
  context.on('request', request => {
    if (!/^https?:/.test(request.url())) return;
    if (new URL(request.url()).origin !== origin) report.remoteRequests.push(request.url());
    if (!['GET', 'HEAD'].includes(request.method())) report.writeRequests.push(request.url());
  });

  const catalogUrl = `${origin}/library/catalog.json`;
  const response = await context.request.get(catalogUrl);
  assert.equal(response.status(), 200);
  assert.equal(response.url(), catalogUrl);
  const catalog = await response.json();
  const staged = JSON.parse(await readFile(resolve(process.argv[5] ?? resolve(project, 'web/public/library/catalog.json')), 'utf8'));
  assert.deepEqual(catalog, staged, 'The hosted catalog differs from the reviewed source catalog');
  const { entries } = catalog;
  assert(Array.isArray(entries) && entries.length > 3, 'Expected an expanded public library');
  report.catalogCount = entries.length;
  assert.equal(new Set(entries.map(entry => entry.id)).size, entries.length, 'Catalog IDs must be unique');
  for (const id of ['geia', 'efcharisto', 'entaxei']) assert(entries.some(entry => entry.id === id), `Missing previous entry ${id}`);

  const owners = new Map();
  for (const entry of entries) {
    assert.equal(entry.rights?.redistributionConfirmed, true, `Redistribution is unconfirmed for ${entry.id}`);
    assert(['dataset-annotated', 'expert-reviewed'].includes(entry.verification?.status), `Missing annotation evidence for ${entry.id}`);
    assert.equal(entry.license?.id, 'CC-BY-4.0', `Unexpected license for ${entry.id}`);
    assert(entry.attribution?.trim() && entry.sourceUrl?.startsWith('https://'), `Missing provenance for ${entry.id}`);
    const asset = new URL(entry.videoUrl, origin);
    assert(asset.origin === origin && asset.pathname.startsWith('/library/'), `Unexpected video URL for ${entry.id}`);
    for (const term of searchableTerms(entry)) {
      const key = lookupKey(term);
      assert(key && (!owners.has(key) || owners.get(key) === entry.id), `Ambiguous exact lookup for ${term}`);
      owners.set(key, entry.id);
    }
  }

  async function verifyAsset(entry) {
    const assetUrl = new URL(entry.videoUrl, origin).href;
    const videoResponse = await context.request.get(assetUrl);
    try {
      assert.equal(videoResponse.status(), 200, entry.id);
      assert.equal(videoResponse.url(), assetUrl, entry.id);
      assert.match(videoResponse.headers()['content-type'] ?? '', /^video\/mp4(?:;|$)/i, entry.id);
      const bytes = await videoResponse.body();
      const sha256 = createHash('sha256').update(bytes).digest('hex');
      assert.equal(sha256, entry.sha256, `Hosted video differs from the reviewed asset for ${entry.id}`);
      return { id: entry.id, phrase: entry.phrase, bytes: bytes.length, sha256 };
    } finally {
      await videoResponse.dispose();
    }
  }

  const batchSize = 8;
  for (let start = 0; start < entries.length; start += batchSize) {
    const checks = await Promise.allSettled(entries.slice(start, start + batchSize).map(verifyAsset));
    for (const check of checks) {
      if (check.status === 'rejected') throw check.reason;
      report.videos.push(check.value);
    }
  }
  const videosById = new Map(report.videos.map(video => [video.id, video]));

  const page = await context.newPage();
  page.on('pageerror', error => report.pageErrors.push(error.message));
  const start = performance.now();
  const navigation = await page.goto(`${origin}/#/library`);
  assert.equal(navigation?.status(), 200);
  await page.getByRole('heading', { name: 'Βιβλιοθήκη νοημάτων', level: 1 }).waitFor();
  await page.waitForFunction(count => document.querySelectorAll('[data-library-entry]').length === count, entries.length);
  report.catalogReadyMs = Math.round(performance.now() - start);

  const searchbox = page.getByRole('searchbox', { name: 'Ελληνική λέξη ή φράση' });
  const submit = page.getByRole('button', { name: 'Αναζήτηση', exact: true });
  const player = page.locator('#library-video');
  const selected = page.locator('[data-library-entry][aria-pressed="true"]');
  const result = page.locator('#library-player-heading');

  async function search(term, keyboard = false) {
    await searchbox.fill(term);
    if (keyboard) await searchbox.press('Enter');
    else await submit.click();
  }

  async function verifyMatch(entry) {
    const expected = new URL(entry.videoUrl, origin).href;
    await page.waitForFunction(url => {
      const video = document.querySelector('#library-video');
      return video?.currentSrc === url && video.readyState >= 1 && video.videoWidth > 0;
    }, expected);
    assert.equal(await result.innerText(), displayPhrase(entry.phrase));
    assert.equal(await selected.getAttribute('data-library-entry'), entry.id);
    assert.equal(await player.getAttribute('aria-label'), `Απόδοση της φράσης ${entry.phrase} στην Ελληνική Νοηματική`);
  }

  // Exercise Chrome playback and visible attribution for the legacy and expanded sources.
  for (const [index, entry] of playbackSample(entries).entries()) {
    const variant = ` ${unaccented(entry.phrase).toLocaleUpperCase('el-GR').replace(/\s+/g, '  ')} `;
    await search(variant, index % 2 === 0);
    await verifyMatch(entry);
    assert.equal(await result.evaluate(heading => document.activeElement === heading), true, entry.id);
    await page.getByRole('button', { name: 'Αναπαραγωγή από την αρχή', exact: true }).click();
    await page.waitForFunction(() => {
      const video = document.querySelector('#library-video');
      return video?.readyState >= 2 && video.currentTime > 0;
    });
    const duration = await player.evaluate(video => video.duration);
    assert(Number.isFinite(duration) && duration > 0, `Unplayable video for ${entry.id}`);

    const source = page.locator('details.library-source');
    if (!await source.evaluate(element => element.open)) {
      await source.getByText('Πηγή και άδεια', { exact: true }).click();
    }
    assert.equal(await source.getByRole('link', { name: 'Αρχική πηγή' }).getAttribute('href'), entry.sourceUrl);
    const license = source.getByRole('link', { name: entry.license.name });
    assert.equal(await license.isVisible(), true);
    assert.equal(await license.getAttribute('href'), entry.license.url);
    assert((await source.innerText()).includes(entry.attribution), `Missing visible creator credit for ${entry.id}`);
    videosById.get(entry.id).duration = duration;
    report.playbackCount++;
  }

  async function verifyLiveFiltering(width) {
    if (entries.length <= 24) return;
    const partialKey = entry => lookupKey(lookupKey(entry.phrase).slice(0, 3));
    const entry = entries.findLast(candidate => {
      const prefix = partialKey(candidate);
      return prefix.length === 3 && lookupKey(candidate.phrase).length > 3 && !owners.has(prefix);
    });
    assert(entry, 'Expected a partial query for the large library');
    const prefix = partialKey(entry);
    const expected = entries.filter(candidate => searchableTerms(candidate).some(term => lookupKey(term).includes(prefix)));
    assert(expected.length > 0 && expected.length < entries.length, 'The partial query should narrow the full list');
    await searchbox.fill(prefix.toLocaleUpperCase('el-GR'));
    const list = page.locator('[data-library-entry]');
    await page.waitForFunction(count => document.querySelectorAll('[data-library-entry]').length === count, expected.length);
    assert.deepEqual(await list.evaluateAll(items => items.map(item => item.dataset.libraryEntry)), expected.map(item => item.id));
    assert.equal(await page.locator('.library-list-heading span').innerText(), `${expected.length} από ${entries.length}`);
    await list.nth(expected.indexOf(entry)).click();
    await verifyMatch(entry);
    assert.equal(await list.count(), entries.length, 'Selecting a suggestion should restore the full list');
    report.filtering.push({ width, prefix, matches: expected.length, selected: entry.id });
  }

  await verifyLiveFiltering(1440);

  const punctuationEntry = entries.find(entry => entry.id === 'poio-provlima');
  assert(punctuationEntry, 'Expected the approved problem question in the library');
  const punctuationQuery = 'ΠΟΙΟ ΕΙΝΑΙ ΤΟ ΠΡΟΒΛΗΜΑ;';
  await search(punctuationQuery, true);
  await verifyMatch(punctuationEntry);
  report.terminalPunctuation = { query: punctuationQuery, id: punctuationEntry.id };
  if (process.argv[4]) await page.screenshot({ path: resolve(process.argv[4]) });

  const missing = `${entries[0].phrase} ζζζ άγνωστο μήνυμα`;
  assert(!owners.has(lookupKey(missing)));
  await search(missing, true);
  await page.getByRole('heading', { name: 'Δεν βρέθηκε απόδοση', exact: true }).waitFor();
  assert((await page.locator('.library-placeholder').innerText()).includes(missing));
  assert.equal(await player.count(), 0, 'A nonmatching phrase must clear the previous video');
  assert.equal(await selected.count(), 0);
  report.exactMiss = missing;

  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await verifyLiveFiltering(width);
    await page.locator('[data-library-entry]').last().click();
    await verifyMatch(entries.at(-1));
    assert.equal(await result.evaluate(heading => document.activeElement === heading), true);
    const geometry = await page.evaluate(() => {
      const heading = document.querySelector('#library-player-heading').getBoundingClientRect();
      return { noHorizontalOverflow: document.documentElement.scrollWidth <= innerWidth + 1,
        resultInViewport: heading.top >= 0 && heading.bottom <= innerHeight };
    });
    assert(geometry.noHorizontalOverflow && geometry.resultInViewport, `Library at ${width}px`);
    report.mobile.push({ width, ...geometry });
  }

  const expectedExamples = entries.filter(entry => entry.contentKind !== 'phrase' && typeof entry.annotationLabel === 'string'
    && lookupKey(entry.annotationLabel) && lookupKey(entry.annotationLabel) === lookupKey(entry.phrase)).slice(0, 6).map(entry => entry.phrase);
  await page.goto(`${origin}/#/try`);
  await page.waitForFunction(count => document.querySelectorAll('.recognition-examples .phrase-chip').length === count, expectedExamples.length);
  const examples = await page.locator('.recognition-examples .phrase-chip').allTextContents();
  assert.deepEqual(examples.map(text => text.trim()), expectedExamples, 'Camera examples must carry exact recognition labels');
  report.recognitionExamples = examples.map(text => text.trim());

  report.cameraCalls = await page.evaluate(() => window.libraryQaCameraCalls);
  assert.equal(report.cameraCalls, 0, 'Library must not request a camera or microphone');
  assert.deepEqual(report.remoteRequests, []);
  assert.deepEqual(report.writeRequests, []);
  assert.deepEqual(report.pageErrors, []);
  report.passed = true;
  console.log(`Greek-to-GSL library passed at ${origin}: ${entries.length} verified MP4 files, ${report.playbackCount} Chrome playback checks`);
} catch (error) {
  report.failure = error.stack ?? String(error);
  throw error;
} finally {
  await browser?.close();
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(output);
}
