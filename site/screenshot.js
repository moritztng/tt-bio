// Screenshot the site at real device metrics. Plain --window-size in headless Chrome has a
// ~500px floor, so a 393px request silently lays out at 500 and crops -- which looks exactly
// like a horizontal-overflow bug. Emulation.setDeviceMetricsOverride (what page.setViewport
// drives) is the only way to lay out at a true 393px.
const puppeteer = require('puppeteer');

const URL = process.argv[2] || 'http://localhost:8899/index.html';
const OUT = process.argv[3] || 'shots';

const VIEWS = [
  { name: 'desktop', width: 1280, height: 900, deviceScaleFactor: 1, isMobile: false },
  // Pixel 10 reference: 393 CSS px at DPR 2.75.
  { name: 'mobile', width: 393, height: 852, deviceScaleFactor: 2.75, isMobile: true,
    hasTouch: true },
];

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  for (const v of VIEWS) {
    for (const theme of ['light', 'dark']) {
      const page = await browser.newPage();
      await page.setViewport(v);
      await page.emulateMediaFeatures([
        { name: 'prefers-color-scheme', value: theme },
      ]);
      await page.goto(URL, { waitUntil: 'networkidle0' });
      await page.evaluate(t => { document.documentElement.dataset.theme = t; }, theme);
      await new Promise(r => setTimeout(r, 700));
      // Any horizontal overflow now is a real bug, not a tooling artifact -- report it.
      const over = await page.evaluate(() => ({
        doc: document.documentElement.scrollWidth,
        vw: window.innerWidth,
        wide: [...document.querySelectorAll('body *')]
          .filter(e => e.getBoundingClientRect().right > window.innerWidth + 1)
          .slice(0, 6).map(e => e.tagName + '.' + (e.className || '') + ' @'
            + Math.round(e.getBoundingClientRect().right)),
      }));
      console.log(`${v.name}/${theme}  doc=${over.doc} vw=${over.vw}`
        + (over.wide.length ? `  OVERFLOW: ${over.wide.join(' | ')}` : '  clean'));
      await page.screenshot({ path: `${OUT}/${v.name}-${theme}.png`, fullPage: true });
      await page.close();
    }
  }
  await browser.close();
})();
