// The viewer page's actual click handlers under a minimal DOM stub.
// stdin: {run, data, config, records} payloads from the live server objects.
// stdout: a JSON report of what the handlers did.
import { readFileSync } from "node:fs";

const payload = JSON.parse(readFileSync(0, "utf8"));

// ---- minimal DOM -----------------------------------------------------------
const elements = {};
function fakeElement(id) {
  const el = {
    id, hidden: false, textContent: "", innerHTML: "", dataset: {},
    style: {}, listeners: {},
    parentElement: { clientWidth: 1200 },
    setAttribute() {}, getAttribute() { return null; },
    addEventListener(kind, fn) { (this.listeners[kind] ||= []).push(fn); },
    querySelectorAll() { return this._rows || []; },
    insertAdjacentHTML(_where, _html) {
      const stub = fakeElement(`${id}:raw${this._rawRows = (this._rawRows || 0) + 1}`);
      this._rows = this._rows || [];
      this._rows.push(stub);
    },
    remove() {},
    classList: { contains() { return false; }, add() {}, remove() {} },
  };
  if (id === "cv" || id === "dv") {
    el.width = 0; el.height = 0;
    el.getContext = () => new Proxy({}, {
      get: (target, prop) => (
        prop === "measureText" ? () => ({ width: 0 }) : () => {}
      ),
      set: () => true,
    });
    el.getBoundingClientRect = () => ({ left: 0, top: 0 });
  }
  if (id === "tbl" || id === "recs") {
    let lastHTML = "";
    Object.defineProperty(el, "innerHTML", {
      get() { return lastHTML; },
      set(value) {
        lastHTML = value;
        const rows = [];
        for (const match of value.matchAll(/<tr data-([a-z-]+)="([^"]*)"/g)) {
          const row = fakeElement(`${id}:row${rows.length}`);
          row.dataset[match[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase())]
            = match[2];
          rows.push(row);
        }
        this._rows = rows;
      },
    });
  }
  if (id === "pairs") {
    let markup = "";
    Object.defineProperty(el, "innerHTML", {
      get() { return markup; },
      set(value) {
        markup = value;
        this._rows = [...value.matchAll(/<button[^>]*data-k="([^"]*)"[^>]*>/g)]
          .map((m, i) => {
            const button = fakeElement("pairButton" + i);
            button.dataset.k = m[1];
            return button;
          });
      },
    });
  }
  if (id === "runsel") {
    el.value = "0";
  }
  return el;
}

globalThis.document = {
  getElementById(id) { return (elements[id] ||= fakeElement(id)); },
};
globalThis.window = { addEventListener() {}, devicePixelRatio: 1 };
globalThis.innerWidth = 1600;
globalThis.fetch = async (path) => {
  const body = path.startsWith("/api/run") ? payload.run
    : path.startsWith("/api/data") ? payload.data
    : path.startsWith("/api/config") ? payload.config
    : path.startsWith("/api/records") ? payload.records
    : { runs: [{ run: 0, dir: "fixture/run" }] };
  return { ok: true, json: async () => body };
};
globalThis.matchMedia = () => ({ addEventListener() {} });
globalThis.addEventListener = () => {};
globalThis.setTimeout = setTimeout;
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#000000" });
globalThis.proxyStub = undefined;

const report = { booted: false, noJsError: true };
try {
  const html = readFileSync(process.argv[2], "utf8");
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)]
    .map(match => match[1]);
  const driver = `
    ;await new Promise(r => setTimeout(r, 50));
    report.booted = document.getElementById("runkv").innerHTML.length > 0;
    report.tableRendered =
      (document.getElementById("tbl").innerHTML.match(/<tr data-k=/g) || [])
        .length;
    const firstDot = D.dots[0];
    await selectByDot(firstDot);
    await new Promise(r => setTimeout(r, 20));
    report.dotClickSelected =
      document.getElementById("dtitle").textContent.includes("·");
    // clicking the second of two equal-duration operations must select the
    // request that carries exactly that dot's transport key, with that
    // operation's timestamps drawn (equal ms, distinct acceptance)
    const equal = D.dots.filter(d => d.ms != null && Math.abs(d.ms - firstDot.ms) < 1e-9);
    report.equalDurationDots = equal.length;
    if (equal.length > 1) {
      const second = equal[1];
      await selectByDot(second);
      await new Promise(r => setTimeout(r, 20));
      // the spinner payload gave us the records for rows[0]; the selected
      // request is asserted through the drawn span endpoints
      report.secondDotT = second.t;
      report.selectedKeyMatch = (sel.transport_keys || []).includes(second.key);
      const span = (sel.spans || []).find(s => s.key === second.key);
      report.selectedSpanStartsAtDot = !!span && span.t0 === second.t;
      report.selectedTitle = document.getElementById("dtitle").textContent;
    }
    const recs = document.getElementById("recs");
    report.recordRows = (recs.innerHTML.match(/<tr data-i=/g) || []).length;
    const firstRecRow = recs.querySelectorAll()[0];
    if (firstRecRow && firstRecRow.listeners.click) {
      firstRecRow.listeners.click[0]();
      report.recordExpanded = true;
      firstRecRow.listeners.click[0]();   // collapse again
      report.recordCollapsed = true;
    } else {
      report.recordExpanded = false;
    }
    // the plotted window must contain every measured operation endpoint of
    // the selected request
    const detailWindow = (typeof window !== "undefined" && window.__detail)
      || { t0: view[0], t1: view[1] };
    report.plotCoversSpans = (sel.spans || []).every(s =>
      s.t0 == null || s.t1 == null ||
      (detailWindow.t0 <= s.t0 && s.t1 <= detailWindow.t1));
    report.plotWindow = detailWindow;
    // the compute timeline's phases are inside the plotted window too
    const phases = ((sel.timeline || {}).available && sel.timeline.spans) || [];
    report.phaseCount = phases.length;
    report.plotCoversPhases = phases.every(s =>
      detailWindow.t0 <= s.t0 && s.t1 <= detailWindow.t1);
    // durations sort as numbers, not as text
    report.numericSort = [100, 2, 10].sort(cmpSort);
    report.header = document.getElementById("runkv").innerHTML;
    report.tableHtml = document.getElementById("tbl").innerHTML;
    // the header follows the pair filter: click each pair, then "all pairs"
    const headerAfter = key => {
      const button = document.getElementById("pairs").querySelectorAll("button")
        .find(b => b.dataset.k === key);
      if (!button) return null;
      button.listeners.click[0]();
      return document.getElementById("runkv").innerHTML;
    };
    report.headerByPair = Object.fromEntries(
      [...D.pairs, ""].map(key => [key, headerAfter(key)]));
  `;
  const combined = scripts.join("\n;\n") + "\n;\n" + driver;
  const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
  await new AsyncFunction("report", combined)(report);
} catch (error) {
  report.noJsError = false;
  report.error = String((error && error.stack) || error);
}
console.log(JSON.stringify(report));
