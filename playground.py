"""FeatureScope Playground — poke the model's mind, live.

Serves a local page: type a sentence -> see sentiment readout + which SAE features fire;
pick a feature + dose -> steer and watch the readout move. stdlib-only server.
"""
import json, torch, numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformer_lens import HookedTransformer
from sae_lens import SAE

print("loading model (bf16)...", flush=True)
model = HookedTransformer.from_pretrained("gemma-2-2b", device="mps", dtype=torch.bfloat16)
sae = SAE.from_pretrained("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82", device="mps")
HOOK = sae.cfg.metadata["hook_name"]
Wdec = sae.W_dec.detach()
FEW = ("Review: A wonderful, heartwarming film.\nSentiment: positive\n"
       "Review: A boring, pointless waste of time.\nSentiment: negative\n")
POS = model.to_tokens(" positive", prepend_bos=False)[0, 0]
NEG = model.to_tokens(" negative", prepend_bos=False)[0, 0]
try:
    KNOWN = {f["feature"]: f for f in json.load(open("screen_sentiment.json"))["features"]}
except Exception:
    KNOWN = {}

def readout(text, steer=None):
    hooks = [(HOOK, lambda r, hook: r + steer)] if steer is not None else []
    with torch.no_grad():
        lg = model.run_with_hooks(model.to_tokens(FEW + f"Review: {text}\nSentiment:"), fwd_hooks=hooks)
    return float(lg[0, -1, POS] - lg[0, -1, NEG])

def analyze(text):
    with torch.no_grad():
        toks = model.to_tokens(text[:300])
        _, c = model.run_with_cache(toks, names_filter=[HOOK])
        per_tok = sae.encode(c[HOOK][0]).float().cpu().numpy()   # [seq, d_sae]
    acts = per_tok.max(0)
    top = np.argsort(-acts)[:12]
    tok_strs = [model.to_string(t) for t in toks[0]]
    feats = [{"id": int(f), "act": round(float(acts[f]), 2),
              "verdict": KNOWN.get(int(f), {}).get("verdict", "unscreened"),
              "tok_acts": [round(float(x), 1) for x in per_tok[:, f]]} for f in top if acts[f] > 0]
    return {"readout": round(readout(text), 3), "tokens": tok_strs, "features": feats}

def screen_sentence(text):
    """Auto-screen the top firing features: push each, rank by causal effect vs a noise floor."""
    base = readout(text)
    with torch.no_grad():
        _, c = model.run_with_cache(model.to_tokens(text[:300]), names_filter=[HOOK])
        acts = sae.encode(c[HOOK][0]).float().max(0).values.cpu().numpy()
    top = [int(f) for f in np.argsort(-acts)[:8] if acts[f] > 0]
    g = torch.Generator().manual_seed(0)
    null_max = 0.0
    for _ in range(4):                      # noise floor: random pushes at the hard dose
        r = torch.randn(Wdec.shape[1], generator=g).to("mps")
        null_max = max(null_max, abs(readout(text, (r / r.norm() * 30).to(model.cfg.dtype)) - base))
    rows = []
    DOSES = (8, 15, 30, 60)
    for f in top:
        d = Wdec[f]; u = d / d.norm()
        curve = [round(readout(text, (u * k).to(model.cfg.dtype)) - base, 3) for k in DOSES]
        d15, d30 = curve[1], curve[2]
        grows = abs(d30) >= abs(d15) * 1.2 and (d15 * d30 > 0)
        if abs(d30) > 2 * null_max and grows: tag = "DRIVES"
        elif abs(d30) > null_max: tag = "WEAK"
        else: tag = "INERT"
        sat = abs(curve[3]) < abs(curve[2]) * 0.7
        rows.append({"id": f, "act": round(float(acts[f]), 1), "curve": curve,
                     "d30": d30, "tag": tag + (" (saturates!)" if sat and tag == "DRIVES" else "")})
    rows.sort(key=lambda r: -abs(r["d30"]))
    return {"base": round(base, 3), "null_floor": round(null_max, 3), "rows": rows}

def steer(text, feat, dose):
    d = Wdec[feat]; v = (d / d.norm() * dose).to(model.cfg.dtype)
    base = readout(text); pushed = readout(text, v)
    return {"base": round(base, 3), "steered": round(pushed, 3), "delta": round(pushed - base, 3)}

PAGE = open("playground.html").read()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype="application/json"):
        b = body.encode(); self.send_response(200)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", len(b))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(PAGE, "text/html")
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/analyze": self._send(json.dumps(analyze(req["text"])))
        elif self.path == "/screen": self._send(json.dumps(screen_sentence(req["text"])))
        elif self.path == "/steer": self._send(json.dumps(steer(req["text"], int(req["feature"]), float(req["dose"]))))

print("PLAYGROUND READY -> http://localhost:8765", flush=True)
HTTPServer(("127.0.0.1", 8765), H).serve_forever()
