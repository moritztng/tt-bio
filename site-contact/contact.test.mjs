// node --test site-contact/ : the endpoint against a real SQLite store behind a D1-shaped shim.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import { handle } from "./src/contact.js";

const SCHEMA = readFileSync(new URL("./schema.sql", import.meta.url), "utf8");
const SITE = "https://moritztng.github.io/tt-bio/";
const IP = "203.0.113.7";

function d1() {
  const db = new DatabaseSync(":memory:");
  db.exec(SCHEMA);
  const shim = {
    down: false, db,
    prepare(sql) {
      return { bind: (...a) => ({
        run: async () => { if (shim.down) throw new Error("d1 down"); db.prepare(sql).run(...a); },
        first: async () => { if (shim.down) throw new Error("d1 down"); return db.prepare(sql).get(...a); },
      }) };
    },
  };
  return shim;
}

class EmailMessage {
  constructor(from, to, raw) { Object.assign(this, { from, to, raw }); }
}

function setup({ mailDown = false, tgDown = false } = {}) {
  const sent = [], pinged = [];
  const env = {
    DB: d1(), SITE_URL: SITE, MAIL_FROM: "contact@tt-bio.com", MAIL_TO: "team@example.com",
    ALLOWED_ORIGINS: "https://moritztng.github.io https://tt-bio.com",
    TELEGRAM_BOT_TOKEN: "123:secret", TELEGRAM_CHAT_ID: "42",
    MAIL: { send: async m => { if (mailDown) throw new Error("mail down"); sent.push(m); } },
  };
  globalThis.fetch = async (url, init) => {
    pinged.push({ url, body: JSON.parse(init.body) });
    return new Response("{}", { status: tgDown ? 500 : 200 });
  };
  return { env, sent, pinged };
}

const FIELDS = { name: "Ada Lovelace", email: "ada@example.org", org: "Analytical Bio",
  message: "We want to size a Galaxy for Boltz-2 screening." };

function post(env, fields = {}, { json = true, origin = "https://moritztng.github.io", ip = IP } = {}) {
  const headers = { Origin: origin, "CF-Connecting-IP": ip,
    "Content-Type": "application/x-www-form-urlencoded" };
  if (json) headers.Accept = "application/json";
  const req = new Request("https://tt-bio.com/api/contact", { method: "POST", headers,
    body: new URLSearchParams({ ...FIELDS, ...fields }) });
  return handle(req, env, EmailMessage);
}

const rows = env => env.DB.db.prepare("SELECT * FROM messages").all();
const body = raw => Buffer.from(raw.split("\r\n\r\n")[1].replace(/\r\n/g, ""), "base64").toString();

test("a valid message lands on all three legs", async () => {
  const { env, sent, pinged } = setup();
  const r = await post(env);
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), "https://moritztng.github.io");
  const { ref } = await r.json();
  assert.match(ref, /^TB-[0-9A-Z]{6}$/);

  const [row] = rows(env);
  assert.equal(row.ref, ref);
  assert.equal(row.ip, IP);
  assert.equal(row.mail, "ok");
  assert.equal(row.notify, "ok");

  assert.equal(sent.length, 1);
  const raw = sent[0].raw;
  assert.match(raw, /^Reply-To: =\?UTF-8\?B\?[^?]+\?= <ada@example\.org>$/m);
  assert.match(body(raw), /size a Galaxy/);
  assert.match(body(raw), new RegExp(ref));
  assert.ok(!raw.includes(IP) && !body(raw).includes(IP), "the IP stays in the store");

  assert.equal(pinged.length, 1);
  assert.equal(pinged[0].body.chat_id, "42");
  assert.match(pinged[0].body.text, new RegExp(`${ref}: Ada Lovelace \\(Analytical Bio\\) <ada@example\\.org>`));
  assert.ok(!pinged[0].body.text.includes(IP));
});

test("without JavaScript the browser is sent back to the thanks anchor", async () => {
  const { env } = setup();
  const r = await post(env, {}, { json: false });
  assert.equal(r.status, 303);
  assert.equal(r.headers.get("Location"), SITE + "#contact-sent");
  assert.equal(rows(env).length, 1);
});

test("a filled honeypot gets a quiet success and goes nowhere", async () => {
  const { env, sent, pinged } = setup();
  const r = await post(env, { website: "http://spam.example" });
  assert.equal(r.status, 200);
  assert.equal((await r.json()).ok, true);
  assert.equal(rows(env).length + sent.length + pinged.length, 0);
});

test("too long, bad email and missing fields are refused with a reason", async () => {
  const { env } = setup();
  for (const [fields, re] of [
    [{ message: "x".repeat(5001) }, /under 5000/],
    [{ email: "ada@example" }, /does not look right/],
    [{ email: "ada@example.org>\r\nBcc: x@y.z" }, /does not look right/],
    [{ name: "  " }, /name/],
    [{ message: "" }, /message/],
  ]) {
    const r = await post(env, fields);
    assert.equal(r.status, 400);
    assert.match((await r.json()).error, re);
  }
  const r = await post(env, { message: "x".repeat(5001) }, { json: false });
  assert.equal(r.headers.get("Location"), SITE + "#contact-error");
  assert.equal(rows(env).length, 0);
});

test("a name cannot inject a mail header", async () => {
  const { env, sent } = setup();
  assert.equal((await post(env, { name: "Eve\r\nBcc: victim@example.com" })).status, 200);
  assert.ok(!/^Bcc:/m.test(sent[0].raw));
});

test("one IP gets five messages an hour", async () => {
  const { env } = setup();
  for (let i = 0; i < 5; i++) assert.equal((await post(env)).status, 200);
  const r = await post(env);
  assert.equal(r.status, 429);
  assert.match((await r.json()).error, /try again later/);
  assert.equal((await post(env, {}, { ip: "198.51.100.1" })).status, 200, "other visitors unaffected");
});

test("a mail outage still succeeds and is recorded on the row", async () => {
  const { env, pinged } = setup({ mailDown: true });
  assert.equal((await post(env)).status, 200);
  assert.match(rows(env)[0].mail, /mail down/);
  assert.equal(rows(env)[0].notify, "ok");
  assert.equal(pinged.length, 1);
});

test("a Telegram outage still succeeds and never leaks the token", async () => {
  const { env } = setup({ tgDown: true });
  assert.equal((await post(env)).status, 200);
  assert.equal(rows(env)[0].notify, "Error: telegram http 500");
});

test("a store outage still delivers by mail and Telegram", async () => {
  const { env, sent, pinged } = setup();
  env.DB.down = true;
  assert.equal((await post(env)).status, 200);
  assert.equal(sent.length + pinged.length, 2);
});

test("only when every leg fails does the visitor see an error", async () => {
  const { env } = setup({ mailDown: true, tgDown: true });
  env.DB.down = true;
  const r = await post(env);
  assert.equal(r.status, 502);
  assert.match((await r.json()).error, /try again/);
});

test("another site cannot post to the form", async () => {
  const { env } = setup();
  const r = await post(env, {}, { origin: "https://evil.example" });
  assert.equal(r.status, 403);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), null);
  assert.equal(rows(env).length, 0);
});
