import { env } from "./env.mjs";

const BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const RETRY = new Set([429, 500, 502, 503, 529]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let keyIndex = 0;

/**
 * Returns all configured Gemini API keys from .env.
 * Supports:
 *   1. Multiple variables: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3...
 *   2. Comma-separated keys in GEMINI_API_KEY: "key1,key2"
 */
export function getKeys() {
  const keys = [];
  const primary = env("GEMINI_API_KEY");
  if (primary && !primary.includes("YOUR_")) {
    keys.push(...primary.split(",").map((s) => s.trim()).filter((s) => Boolean(s) && !s.includes("YOUR_")));
  }
  for (let i = 2; i <= 10; i++) {
    const k = env(`GEMINI_API_KEY_${i}`);
    if (k && !k.includes("YOUR_")) {
      keys.push(k.trim());
    }
  }
  return [...new Set(keys)];
}

/**
 * Pick the next API key in round-robin sequence.
 * If excludeKey is provided (e.g. on 429 rate limit), it picks an alternate key.
 */
export function nextKey(excludeKey = null) {
  const keys = getKeys();
  if (!keys.length) {
    const fallback = env("GEMINI_API_KEY");
    if (!fallback) throw new Error("GEMINI_API_KEY is not set in .env");
    return fallback;
  }
  if (keys.length === 1) return keys[0];

  if (excludeKey && keys.includes(excludeKey)) {
    const others = keys.filter((k) => k !== excludeKey);
    const chosen = others[keyIndex % others.length];
    keyIndex = (keyIndex + 1) % keys.length;
    return chosen;
  }

  const k = keys[keyIndex % keys.length];
  keyIndex = (keyIndex + 1) % keys.length;
  return k;
}

// Running token tally for the whole app, shown in the UI.
export const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, calls: 0 };
function addUsage(u = {}) {
  usage.input  += u.promptTokenCount      || 0;
  usage.output += u.candidatesTokenCount  || 0;
  usage.calls  += 1;
}

/**
 * Recursively prepare a JSON schema for Gemini's responseSchema field.
 * Gemini does not accept: additionalProperties, $schema, $ref, const.
 * Types must be uppercase (STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT).
 */
function cleanSchema(schema) {
  if (!schema || typeof schema !== "object") return schema;
  // eslint-disable-next-line no-unused-vars
  const { additionalProperties, $schema, $ref, const: _const, ...rest } = schema;
  if (rest.type) rest.type = rest.type.toUpperCase();
  if (rest.properties) {
    rest.properties = Object.fromEntries(
      Object.entries(rest.properties).map(([k, v]) => [k, cleanSchema(v)])
    );
  }
  if (rest.items) rest.items = cleanSchema(rest.items);
  return rest;
}

function formatSystem(sys) {
  if (!sys) return undefined;
  if (typeof sys === "string") return { parts: [{ text: sys }] };
  if (Array.isArray(sys)) {
    const text = sys.map((b) => (typeof b === "string" ? b : b.text || "")).filter(Boolean).join("\n\n");
    return { parts: [{ text }] };
  }
  if (typeof sys === "object" && sys.text) return { parts: [{ text: sys.text }] };
  return { parts: [{ text: String(sys) }] };
}

/**
 * Stream a message; calls onText(delta) as text arrives. Returns the full text.
 * Matches the anthropic.mjs stream() signature exactly.
 */
export async function stream({ model, system, messages, maxTokens = 2000, onText = () => {} }) {
  let targetModel = model;
  const sysInst = formatSystem(system);
  let activeKey = nextKey();
  for (let attempt = 0; attempt < 6; attempt++) {
    const url = `${BASE}/${targetModel}:streamGenerateContent?alt=sse&key=${activeKey}`;
    const body = {
      contents: messages.map((m) => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      })),
      generationConfig: { maxOutputTokens: maxTokens },
    };
    if (sysInst) body.systemInstruction = sysInst;

    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (netErr) {
      if (attempt < 5) {
        await sleep(Math.min(10000, 1000 * 2 ** attempt));
        continue;
      }
      throw new Error(`Gemini stream network error (${netErr.message || netErr})`);
    }

    if (res.status === 404 && targetModel !== "gemini-2.5-flash") {
      targetModel = "gemini-2.5-flash";
      continue;
    }
    if (RETRY.has(res.status)) {
      const errText = (await res.text().catch(() => "")).slice(0, 250);
      console.warn(`[Gemini stream retry ${attempt + 1}/6] status ${res.status}: ${errText}`);
      activeKey = nextKey(activeKey); // Switch to next API key on rate limit
      await sleep(Math.min(30000, 1500 * 2 ** attempt));
      continue;
    }
    if (!res.ok) throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 300)}`);

    let text = "";
    let buf = "";
    const dec = new TextDecoder();
    let finalUsage = {};

    for await (const chunk of res.body) {
      buf += dec.decode(chunk, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        let ev;
        try { ev = JSON.parse(line.slice(5)); } catch { continue; }
        if (ev.error) throw new Error(`Gemini stream error: ${JSON.stringify(ev.error).slice(0, 300)}`);
        const delta = ev.candidates?.[0]?.content?.parts?.[0]?.text;
        if (delta) { text += delta; onText(delta); }
        if (ev.usageMetadata) finalUsage = ev.usageMetadata;
      }
    }
    addUsage(finalUsage);
    return text;
  }
  throw new Error("Gemini API kept refusing after retries");
}

/**
 * One call with structured JSON output matching `schema`.
 * Matches the anthropic.mjs json() signature exactly.
 */
export async function json({ model, system, prompt, schema, maxTokens = 4000 }) {
  let targetModel = model;
  const sysInst = formatSystem(system);
  let activeKey = nextKey();
  for (let attempt = 0; attempt < 6; attempt++) {
    const url = `${BASE}/${targetModel}:generateContent?key=${activeKey}`;
    const body = {
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: Math.max(maxTokens, 8192),
        responseMimeType: "application/json",
        responseSchema: cleanSchema(schema),
      },
    };
    if (sysInst) body.systemInstruction = sysInst;

    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (netErr) {
      if (attempt < 5) {
        await sleep(Math.min(10000, 1000 * 2 ** attempt));
        continue;
      }
      throw new Error(`Gemini JSON network error (${netErr.message || netErr})`);
    }

    if (res.status === 404 && targetModel !== "gemini-2.5-flash") {
      targetModel = "gemini-2.5-flash";
      continue;
    }
    if (RETRY.has(res.status)) {
      const errText = (await res.text().catch(() => "")).slice(0, 250);
      console.warn(`[Gemini stream retry ${attempt + 1}/6] status ${res.status}: ${errText}`);
      activeKey = nextKey(activeKey); // Switch to next API key on rate limit
      await sleep(Math.min(30000, 1500 * 2 ** attempt));
      continue;
    }
    if (!res.ok) throw new Error(`Gemini ${res.status}: ${(await res.text()).slice(0, 300)}`);

    const j = await res.json();
    addUsage(j.usageMetadata);
    if (j.error) throw new Error(`Gemini API error: ${JSON.stringify(j.error).slice(0, 300)}`);
    const candidate = j.candidates?.[0];
    if (candidate?.finishReason === "MAX_TOKENS") {
      throw new Error("Gemini output hit token limit (MAX_TOKENS) — response was truncated");
    }
    let text = candidate?.content?.parts?.map((p) => p.text || "").join("") || "";
    if (!text) throw new Error("Gemini returned empty JSON response");

    // Clean any markdown code blocks if the model wrapped the JSON in ```json
    text = text.trim();
    if (text.startsWith("```")) {
      text = text.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "").trim();
    }

    try {
      return JSON.parse(text);
    } catch (parseErr) {
      // If trailing commas or small formatting flaw, try to parse after cleaning trailing commas
      const cleaned = text.replace(/,(\s*[}\]])/g, "$1");
      return JSON.parse(cleaned);
    }
  }
  throw new Error("Gemini API kept refusing after retries");
}
