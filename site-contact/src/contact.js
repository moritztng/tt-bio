/* The tt-bio.com contact form: POST /api/contact.
 *
 * Every accepted message goes out on three legs at once, each failure-isolated, so none is
 * silently lost:
 *
 *   store   one row in D1 (schema.sql). The only place the submitter's IP is kept.
 *   mail    to MAIL_TO through Email Routing, From MAIL_FROM, Reply-To the submitter, so
 *           answering it answers them.
 *   notify  one Telegram line from the coworker bot.
 *
 * The visitor sees success if any leg delivered, and the row records how the other two went.
 * The mail and the Telegram line carry a short Ref instead of the IP: the mail is the leg that
 * leaves our control (a reply quotes it, a forward hands it on), and the Ref ties it back to
 * the stored row when that is needed.
 *
 * One handler serves both ways the page posts. With JavaScript the page asks for JSON and shows
 * the answer inline; without it the browser posts the form and is redirected back to the page
 * at #contact-sent or #contact-error, which the page styles with :target.
 *
 * Abuse: only the site's own origins may post, a filled honeypot gets a quiet fake success,
 * fields are length-capped, and one IP gets 5 messages an hour (300 an hour across everyone,
 * which only bounds mail volume).
 */

export const MAX = { name: 120, email: 200, org: 160, message: 5000 };
const PER_IP_PER_HOUR = 5;
const ALL_PER_HOUR = 300;
const HOUR_MS = 3600_000;

// Loose on purpose: an address is proven by mailing it. This only refuses what cannot be one,
// plus the characters that would break the Reply-To header it lands in.
const EMAIL_RE = /^(?!.*\.\.)(?!\.)[^@\s<>,;"]+(?<!\.)@(?:[^@\s.<>,;"]+\.)+[^@\s.<>,;"]+$/;

export async function handle(request, env, EmailMessage) {
  const origin = request.headers.get("Origin") || "";
  const allowed = env.ALLOWED_ORIGINS.split(/\s+/).includes(origin);
  const wantsJson = (request.headers.get("Accept") || "").includes("application/json");
  const cors = allowed ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : { Vary: "Origin" };

  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: { ...cors,
      "Access-Control-Allow-Methods": "POST", "Access-Control-Allow-Headers": "Content-Type" } });
  }
  const reply = (status, body) => wantsJson
    ? Response.json(body, { status, headers: cors })
    : Response.redirect(env.SITE_URL + (status === 200 ? "#contact-sent" : "#contact-error"), 303);

  // Someone who opens the endpoint itself is sent to the form.
  if (request.method === "GET") return Response.redirect(env.SITE_URL + "#contact", 302);
  if (request.method !== "POST") return reply(405, { error: "Use POST." });
  if (!allowed) return reply(403, { error: "This form only accepts messages from tt-bio.com." });

  let form;
  try {
    form = await request.formData();
  } catch {
    return reply(400, { error: "That did not arrive as a form." });
  }
  // A bot fills every field it sees. Tell it nothing.
  if (str(form, "website")) return reply(200, { ok: true, ref: newRef() });

  const msg = {
    name: oneLine(str(form, "name")),
    email: str(form, "email"),
    org: oneLine(str(form, "org")),
    message: str(form, "message"),
  };
  const bad = invalid(msg);
  if (bad) return reply(400, { error: bad });

  const ip = request.headers.get("CF-Connecting-IP") || "";
  const now = Date.now();
  if (await rateLimited(env.DB, ip, now)) {
    return reply(429, { error: "That is a lot of messages in one hour. Please try again later." });
  }

  msg.ref = newRef();
  const [stored, mailed, notified] = await Promise.allSettled([
    store(env.DB, msg, ip, now),
    mail(env, msg, EmailMessage),
    notify(env, msg),
  ]);
  for (const [leg, r] of [["store", stored], ["mail", mailed], ["notify", notified]]) {
    if (r.status === "rejected") console.error(`contact ${msg.ref}: ${leg} failed: ${r.reason}`);
  }
  if (stored.status === "fulfilled") {
    await env.DB.prepare("UPDATE messages SET mail = ?, notify = ? WHERE ref = ?")
      .bind(outcome(mailed), outcome(notified), msg.ref).run()
      .catch(e => console.error(`contact ${msg.ref}: recording leg outcomes failed: ${e}`));
  }
  if ([stored, mailed, notified].every(r => r.status === "rejected")) {
    return reply(502, { error: "We could not take your message just now. Please try again in a few minutes." });
  }
  return reply(200, { ok: true, ref: msg.ref });
}

function str(form, key) {
  const v = form.get(key);
  return typeof v === "string" ? v.trim() : "";
}
const oneLine = s => s.replace(/\s+/g, " ");
const outcome = r => r.status === "fulfilled" ? "ok" : String(r.reason).slice(0, 300);

export function invalid(m) {
  if (!m.name) return "Please add your name.";
  if (!m.email) return "Please add your email, so we can reply.";
  if (!m.message) return "Please write a message.";
  if (!EMAIL_RE.test(m.email)) return "That email address does not look right.";
  for (const [k, cap] of Object.entries(MAX)) {
    if (m[k].length > cap) return `Please keep the ${k === "org" ? "organisation" : k} under ${cap} characters.`;
  }
  return "";
}

// Six characters of Crockford base32, about a billion values: short enough to read out.
function newRef() {
  const abc = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
  return "TB-" + Array.from(crypto.getRandomValues(new Uint8Array(6)), b => abc[b & 31]).join("");
}

async function rateLimited(db, ip, now) {
  try {
    const since = now - HOUR_MS;
    const row = await db.prepare(
      "SELECT COUNT(*) AS everyone, COALESCE(SUM(ip = ?), 0) AS mine FROM messages WHERE at > ?",
    ).bind(ip, since).first();
    return row.mine >= PER_IP_PER_HOUR || row.everyone >= ALL_PER_HOUR;
  } catch (e) {
    // The store being down must not also close the two legs that still work.
    console.error(`contact: rate limit check failed, letting the message through: ${e}`);
    return false;
  }
}

function store(db, m, ip, now) {
  return db.prepare(
    "INSERT INTO messages (ref, at, name, email, org, message, ip) VALUES (?, ?, ?, ?, ?, ?, ?)",
  ).bind(m.ref, now, m.name, m.email, m.org, m.message, ip).run();
}

const summary = m => m.org ? `${m.name} (${m.org})` : m.name;

export function mime(env, m) {
  const date = new Date().toUTCString().replace("GMT", "+0000");
  const body = [
    `${m.name} <${m.email}>`,
    m.org ? `Organisation: ${m.org}` : null,
    "",
    m.message,
    "",
    "--",
    `Ref ${m.ref}, from the contact form on tt-bio.com. Reply to answer ${m.name}.`,
  ].filter(l => l !== null).join("\r\n");
  return [
    `From: TT-Bio <${env.MAIL_FROM}>`,
    `To: <${env.MAIL_TO}>`,
    `Reply-To: ${word(m.name)} <${m.email}>`,
    `Subject: ${word(`TT-Bio contact: ${summary(m)}`)}`,
    `Date: ${date}`,
    `Message-ID: <${m.ref}.${Date.now()}@tt-bio.com>`,
    "MIME-Version: 1.0",
    "Content-Type: text/plain; charset=utf-8",
    "Content-Transfer-Encoding: base64",
    "",
    b64(body).replace(/.{76}/g, "$&\r\n"),
  ].join("\r\n");
}

// RFC 2047 encoded-word: header-safe for any name, including one that tries to inject a header.
const word = s => `=?UTF-8?B?${b64(s)}?=`;
const b64 = s => btoa(String.fromCharCode(...new TextEncoder().encode(s)));

async function mail(env, m, EmailMessage) {
  await env.MAIL.send(new EmailMessage(env.MAIL_FROM, env.MAIL_TO, mime(env, m)));
}

// Errors carry a status or an error name only, never the request: its URL holds the bot token.
async function notify(env, m) {
  const text = `tt-bio.com contact ${m.ref}: ${summary(m)} <${m.email}>\n\n${m.message.slice(0, 600)}`;
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text, disable_web_page_preview: true }),
    signal: AbortSignal.timeout(8000),
  }).catch(e => { throw new Error(`telegram ${e.name}`); });
  if (!r.ok) throw new Error(`telegram http ${r.status}`);
}
