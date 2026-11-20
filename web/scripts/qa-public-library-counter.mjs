import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const dist = resolve(root, 'publication-public-library-200/dist');
const live = Boolean(process.argv[2]);
const origin = new URL(process.argv[2] ?? 'http://127.0.0.1:5212').origin;
const output = resolve(root, `reports/pretrained-gsl/public-library-counter-${live ? 'live' : 'local'}.json`);
const badge = '.library-list-heading > span';
const finalLabel = '3.000+';
const clips = ['elementary-glossa-erg-b2-172', 'elementary-mp-b-1604', 'elementary-anth-cd-1315'];
const report = { passed: false, source: live ? origin : 'publication-public-library-200/dist', clips: [] };
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.mp4': 'video/mp4',
  '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.png': 'image/png', '.webp': 'image/webp' };
const formatter = new Intl.NumberFormat('el-GR');
const lookup = text => text.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase('el-GR').replace(/ς/g, 'σ')
  .trim().replace(/[.!?;;·…]+$/u, '').trim().replace(/\s+/g, ' ');

let browser;
try {
  const catalog = JSON.parse(await readFile(resolve(dist, 'library/catalog.json'), 'utf8'));
  const { entries } = catalog;
  assert.equal(entries.length, 3003);
  assert.equal(new Set(entries.map(entry => entry.id)).size, entries.length);
  report.catalogCount = entries.length;
  const catalogLabel = `${formatter.format(entries.length)} βίντεο`;

  const headerList = (await readFile(resolve(dist, '_headers'), 'utf8')).split(/\r?\n/);
  const globalHeaders = Object.fromEntries(headerList.slice(1).filter(line => line.startsWith('  '))
    .slice(0, headerList.findIndex((line, index) => index > 0 && !line.startsWith('  ')) - 1)
    .map(line => { const colon = line.indexOf(':'); return [line.slice(2, colon).toLowerCase(), line.slice(colon + 1).trim()]; }));

  browser = await chromium.launch({ channel: 'chrome', headless: true });
  report.browser = browser.version();
  const pageErrors = [];

  async function loadLibrary(reducedMotion, width) {
    const context = await browser.newContext({ reducedMotion, viewport: { width, height: 900 }, serviceWorkers: 'block' });
    const page = await context.newPage();
    page.on('pageerror', error => pageErrors.push(error.message));
    if (!live) await page.route(`${origin}/**`, async route => {
      const request = route.request();
      const path = decodeURIComponent(new URL(request.url()).pathname);
      const file = resolve(dist, path === '/' ? 'index.html' : '.' + path);
      assert(file.startsWith(dist + sep), 'Request escaped the built assets');
      const headers = { ...globalHeaders, 'content-type': mime[extname(file)] ?? 'application/octet-stream' };
      let body;
      try { body = await readFile(file); }
      catch (error) {
        if (error.code !== 'ENOENT') throw error;
        await route.fulfill({ status: 404 });
        return;
      }
      const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers().range ?? '');
      if (range) {
        const start = Number(range[1]);
        const end = range[2] ? Math.min(Number(range[2]), body.length - 1) : body.length - 1;
        assert(start <= end);
        headers['content-range'] = `bytes ${start}-${end}/${body.length}`;
        headers['accept-ranges'] = 'bytes';
        await route.fulfill({ status: 206, headers, body: body.subarray(start, end + 1) });
      } else await route.fulfill({ status: 200, headers, body });
    });
    await page.addInitScript(selector => {
      const readings = [];
      Object.defineProperty(window, 'libraryCounterReadings', { value: readings });
      new MutationObserver(() => {
        const element = document.querySelector(selector);
        if (!element || readings.at(-1)?.text === element.textContent) return;
        const heading = element.parentElement.querySelector('h3').getBoundingClientRect();
        readings.push({ text: element.textContent, accessible: element.parentElement.querySelector('[role=status]')?.textContent,
          headingLeft: heading.left, headingTop: heading.top, time: performance.now() });
      }).observe(document, { childList: true, characterData: true, subtree: true });
    }, badge);

    const catalogResponse = page.waitForResponse(`${origin}/library/catalog.json`);
    assert.equal((await page.goto(`${origin}/#/library`, { waitUntil: 'domcontentloaded' }))?.status(), 200);
    assert.deepEqual(await (await catalogResponse).json(), catalog, 'Catalog differs from the built release');
    await page.getByRole('heading', { name: 'Βιβλιοθήκη νοημάτων', level: 1 }).waitFor();
    await page.waitForFunction(({ selector, text }) => document.querySelector(selector)?.textContent === text,
      { selector: badge, text: finalLabel });
    assert.equal(await page.locator(badge).getAttribute('aria-hidden'), 'true');
    assert.equal(await page.locator(badge).getAttribute('title'), catalogLabel);
    assert.equal(await page.locator('.library-list-heading').getByRole('status').textContent(), catalogLabel);
    assert.equal(await page.locator('[data-library-entry]').count(), entries.length);
    const geometry = await page.locator('.library-list-heading').evaluate(row => {
      const bounds = row.getBoundingClientRect();
      const heading = row.querySelector('h3').getBoundingClientRect();
      const counter = row.querySelector(':scope > span').getBoundingClientRect();
      return { noPageOverflow: document.documentElement.scrollWidth <= innerWidth + 1,
        inHeading: counter.left >= heading.right && counter.right <= bounds.right + 1
          && counter.top >= bounds.top && counter.bottom <= bounds.bottom + 1 };
    });
    assert(geometry.noPageOverflow && geometry.inHeading, `Counter placement failed at width ${width}`);
    return { context, page, geometry, readings: await page.evaluate(() => window.libraryCounterReadings) };
  }

  const desktop = await loadLibrary('no-preference', 1440);
  assert(desktop.readings.length > 2, 'The visual counter did not animate');
  assert.equal(desktop.readings.at(-1).text, finalLabel);
  assert(desktop.readings.every(reading => reading.accessible === catalogLabel), 'The accessible count changed on animation frames');
  const numbers = desktop.readings.map(reading => Number(reading.text.replace(/[.+]/g, '')));
  assert(numbers.at(-1) === 3000 && numbers.every((count, index) => index === 0 || count >= numbers[index - 1]));
  const first = desktop.readings[0];
  assert(desktop.readings.every(reading => reading.headingLeft === first.headingLeft && reading.headingTop === first.headingTop));
  report.animation = { framesObserved: numbers.length, startsAt: numbers[0], endsAt: finalLabel,
    elapsedMs: Math.round(desktop.readings.at(-1).time - first.time), staticAccessibleText: catalogLabel, ...desktop.geometry };

  const mobile = await loadLibrary('reduce', 390);
  assert.deepEqual(mobile.readings.map(reading => reading.text), [finalLabel]);
  assert.equal(mobile.readings[0].accessible, catalogLabel);
  report.reducedMotion = { width: 390, firstPaint: finalLabel, intermediateValues: 0, ...mobile.geometry };
  await mobile.context.close();

  const { page, context } = desktop;
  const search = page.getByRole('searchbox', { name: 'Ελληνική λέξη ή φράση' });
  const fragment = 'ΜΗΝ';
  const matching = entries.filter(entry => [entry.phrase, ...entry.aliases].some(term => lookup(term).includes(lookup(fragment))));
  assert(matching.length && matching.length < entries.length);
  await search.fill(fragment);
  assert.equal(await page.locator(badge).innerText(), `${matching.length} από ${entries.length}`);
  assert.equal(await page.locator('.library-list-heading').getByRole('status').textContent(),
    `${formatter.format(matching.length)} από ${formatter.format(entries.length)} βίντεο`);
  assert.deepEqual(await page.locator('[data-library-entry]').evaluateAll(items => items.map(item => item.dataset.libraryEntry)),
    matching.map(entry => entry.id));
  report.filteredCount = { query: fragment, matches: matching.length };

  for (const id of clips) {
    const entry = entries.find(item => item.id === id);
    assert(entry?.rights?.redistributionConfirmed && entry.verification.status === 'dataset-annotated');
    assert.equal(entry.license.id, 'CC-BY-4.0');
    const localBytes = await readFile(resolve(dist, '.' + entry.videoUrl));
    const observedBytes = live ? await (await context.request.get(`${origin}${entry.videoUrl}`)).body() : localBytes;
    assert.equal(createHash('sha256').update(observedBytes).digest('hex'), entry.sha256);
    await search.fill(entry.phrase);
    await search.press('Enter');
    await page.waitForFunction(src => {
      const player = document.querySelector('#library-video');
      return player?.currentSrc === src && player.readyState >= 1 && player.videoWidth > 0;
    }, `${origin}${entry.videoUrl}`);
    assert.equal(await page.locator('[data-library-entry][aria-pressed=true]').getAttribute('data-library-entry'), id);
    await page.getByRole('button', { name: 'Αναπαραγωγή από την αρχή', exact: true }).click();
    await page.waitForFunction(() => {
      const player = document.querySelector('#library-video');
      return player?.readyState >= 2 && player.currentTime > 0;
    });
    await page.locator('.library-source summary').click();
    assert.equal(await page.getByRole('link', { name: 'Αρχική πηγή' }).getAttribute('href'), entry.sourceUrl);
    assert.equal(await page.getByRole('link', { name: entry.license.name }).getAttribute('href'), entry.license.url);
    report.clips.push({ id, phrase: entry.phrase, duration: await page.locator('#library-video').evaluate(video => video.duration),
      sha256: entry.sha256, sourceUrl: entry.sourceUrl, license: entry.license.id });
  }

  assert.deepEqual(pageErrors, []);
  report.passed = true;
} catch (error) {
  report.failure = error.message;
  process.exitCode = 1;
} finally {
  await browser?.close();
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify(report, null, 2));
}
