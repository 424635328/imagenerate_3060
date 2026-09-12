// Same-origin proxy for the local GPU backend.
// BACKEND_URL and API_TOKEN are Netlify environment variables only; the browser
// never sees the tunnel address or the API key.
const BACKEND = (process.env.BACKEND_URL || "").replace(/\/+$/, "");
const TOKEN = process.env.API_TOKEN || "";
const UA = "landscape-art-proxy/3.0";
const IMMUTABLE = "public, max-age=31536000, immutable";
const MAX_BODY = 6 * 1024 * 1024; // Netlify function payload ceiling.
const MAX_CHUNK = 4000000;

// Ops that cost GPU time.  A page on any other site can otherwise POST to this
// public function and burn the owner's GPU; browsers always send Origin on
// such requests, so cross-site callers are rejected.  Requests with no origin
// information at all (curl, monitors) pass unless PROXY_STRICT=1.
const STATE_CHANGING = new Set(["generate", "cancel", "warmup", "gc"]);
const PROXY_STRICT = String(process.env.PROXY_STRICT || "").toLowerCase() === "1";
const EXTRA_ORIGINS = String(process.env.ALLOWED_ORIGINS || "")
  .split(",").map((value) => value.trim().replace(/\/+$/, "")).filter(Boolean);

function header(event, name) {
  const headers = event.headers || {};
  return headers[name] || headers[name.toLowerCase()] || headers[name.toUpperCase()] || "";
}

function originVerdict(event) {
  const origin = String(header(event, "origin") || "").replace(/\/+$/, "");
  const referer = String(header(event, "referer") || "");
  const fetchSite = String(header(event, "sec-fetch-site") || "").toLowerCase();
  const candidate = origin || (referer ? referer.split("/").slice(0, 3).join("/") : "");
  if (!candidate) return PROXY_STRICT ? "block" : "allow";   // non-browser client
  if (EXTRA_ORIGINS.includes(candidate)) return "allow";
  if (/^https?:\/\/([a-z0-9-]+\.)*netlify\.app$/i.test(candidate)) return "allow";
  if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(candidate)) return "allow";
  if (candidate.startsWith("file://")) return "allow";
  if (fetchSite === "same-origin" || fetchSite === "same-site") return "allow";
  return "block";
}

function json(statusCode, value, extra = {}) {
  return {
    statusCode,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...extra },
    body: JSON.stringify(value),
  };
}

function binary(buf, contentType, cacheControl, extra = {}) {
  return {
    statusCode: 200,
    headers: { "Content-Type": contentType, "Cache-Control": cacheControl, ...extra },
    body: buf.toString("base64"),
    isBase64Encoded: true,
  };
}

async function upstream(path, init) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25000);
  try {
    return await fetch(`${BACKEND}${path}`, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

exports.handler = async (event) => {
  try {
    if (!BACKEND) return json(503, { error: "backend is not configured" });
    const q = event.queryStringParameters || {};
    const op = q.op || "health";
    const headers = { "X-API-Key": TOKEN, "User-Agent": UA, Accept: "application/json" };
    const id = encodeURIComponent(q.id || "");

    if (STATE_CHANGING.has(op) && originVerdict(event) === "block") {
      return json(403, { error: "origin not allowed",
        hint: PROXY_STRICT
          ? "PROXY_STRICT=1: only same-site browser requests are accepted; scripts should call the backend directly"
          : "add this origin to ALLOWED_ORIGINS, or call without an Origin header" });
    }

    // Final image (immutable) and in-progress preview (never cached).
    // `i` selects one image of a server-side batch (0-based, defaults to 0).
    if (op === "image" || op === "preview") {
      const isPreview = op === "preview";
      const idx = Math.max(0, Math.min(3, Number(q.i || 0)));
      const suffix = isPreview ? "" : `?i=${idx}`;
      const r = await upstream(`/${isPreview ? "preview" : "result"}/${id}${suffix}`, { headers });
      if (!r.ok) return json(r.status, { error: isPreview ? "preview not ready" : "result not ready" });
      const buf = Buffer.from(await r.arrayBuffer());
      return binary(
        buf,
        r.headers.get("content-type") || "image/webp",
        isPreview ? "no-store" : IMMUTABLE,
        isPreview
          ? { "X-Preview-Rev": r.headers.get("x-preview-rev") || "0", "X-Preview-Kind": r.headers.get("x-preview-kind") || "" }
          : {
              ETag: `"${id}-${idx}-${buf.length}"`,
              "X-Job-Count": r.headers.get("x-job-count") || "1",
              "X-Job-Index": r.headers.get("x-job-index") || String(idx),
            }
      );
    }

    if (op === "chunk") {
      const off = Math.max(0, Number(q.off || 0));
      const len = Math.min(MAX_CHUNK, Math.max(1, Number(q.len || 2000000)));
      const idx = Math.max(0, Math.min(3, Number(q.i || 0)));
      const r = await upstream(`/chunk/${id}?off=${off}&length=${len}&i=${idx}`, { headers });
      if (!r.ok) return json(r.status, { error: "chunk failed" });
      const buf = Buffer.from(await r.arrayBuffer());
      return binary(buf, "application/octet-stream", IMMUTABLE, {
        "X-Total": r.headers.get("x-total") || "",
        "X-Offset": r.headers.get("x-offset") || String(off),
        "X-Length": r.headers.get("x-length") || String(buf.length),
        "X-Job-Count": r.headers.get("x-job-count") || "1",
        "X-Job-Index": r.headers.get("x-job-index") || String(idx),
      });
    }

    let path = "/health";
    let method = "GET";
    let body = null;
    if (op === "generate") {
      path = "/generate";
      method = "POST";
      body = event.body || "{}";
      const size = event.isBase64Encoded ? Buffer.byteLength(body, "base64") : Buffer.byteLength(body);
      if (size > MAX_BODY) return json(413, { error: "request too large; compress the reference image further" });
      if (event.isBase64Encoded) body = Buffer.from(body, "base64").toString("utf8");
    } else if (op === "warmup") {
      path = "/warmup";
      method = "POST";
      body = event.body || "{}";
    } else if (op === "gc") {
      path = `/gc?unload=${q.unload === "1" ? "true" : "false"}`;
      method = "POST";
      body = "{}";
    } else if (op === "job") {
      path = `/jobs/${id}`;
    } else if (op === "cancel") {
      // Cancel a queued job (DELETE). Running GPU work refuses with 409.
      path = `/jobs/${id}`;
      method = "DELETE";
      body = null;
    } else if (op === "jobs") {
      path = `/jobs?limit=${Math.min(100, Math.max(1, Number(q.limit || 20)))}`;
    }

    const init = { method, headers: { ...headers } };
    if (body !== null) {
      init.headers["Content-Type"] = "application/json";
      init.body = body;
    }
    const r = await upstream(path, init);
    const text = await r.text();
    return {
      statusCode: r.status,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": op === "health" || op === "job" ? "no-store" : "no-cache",
      },
      body: text,
    };
  } catch (error) {
    const detail = String(error?.message || error).slice(0, 180);
    return json(502, { error: "backend unavailable", detail });
  }
};
