"""Validate the review-sanctioned mixed-precision fallback (bf16 trunk + fp32 ln_final/unembed)
by exactly reproducing the banked fp32 L12/comparison steer cell and comparing all quantities."""
import json, time, numpy as np, torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from featurescope import FeatureScope
from featurescope.core import _robust_z
from featurescope.concepts import ReadoutSpec
import computed_concepts as cc

model = HookedTransformer.from_pretrained("gemma-2-2b", device="mps", dtype=torch.bfloat16)
import mixed_head




print("mixed-precision model ready (bf16 trunk, fp32 head)", flush=True)

a1 = json.load(open("phase_a_results.json")); a2 = json.load(open("phase_a2_results.json"))
few = cc.CMP_FEWSHOT_4 if a1["comparison"]["few_shot"] == "4shot" else cc.CMP_FEWSHOT_2
probes = a2["comparison_probes"]["true"] + a2["comparison_probes"]["false"]
sp0 = cc.cmp_spec(few)
spec = ReadoutSpec(name=sp0.name, few_shot=sp0.few_shot, template=sp0.template,
                   pos_word=sp0.pos_word, neg_word=sp0.neg_word, probes=probes)

ckpt = json.load(open("phase_bc_results.json"))
stored = ckpt["steer"]["comparison"]["12"]
med_norm = {int(k): v for k, v in ckpt["med_norm"].items()}
alpha = 12.0 / med_norm[12]; base_scale = alpha * med_norm[12]

L = 12
sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{L}/width_16k/canonical", device="mps")
t0 = time.time()
fs = FeatureScope(layer=L, concept=spec, model=model, sae=sae).fit(cc.CMP_POS, cc.CMP_NEG)
mixed_head.attach(fs)
fs._gen.manual_seed(0)
du = fs._diffmeans / fs._diffmeans.norm()
fs._ensure_base()
STRENGTHS = (1.0, 2.0, 4.0)
dose, shifts4 = [], None
for m in STRENGTHS:
    pp = fs._dir_shift(du * (base_scale * m))
    dose.append(float(np.median(pp))); shifts4 = pp
cl = np.asarray(cc.CMP_CLUSTER); cids = np.unique(cl)
Vr = (np.stack([fs.model.run_with_cache(fs.model.to_tokens(t), names_filter=lambda n: n == fs.hook)[1][fs.hook][0].float().mean(0).cpu().numpy() for t in cc.CMP_POS])
      - np.stack([fs.model.run_with_cache(fs.model.to_tokens(t), names_filter=lambda n: n == fs.hook)[1][fs.hook][0].float().mean(0).cpu().numpy() for t in cc.CMP_NEG])).astype(np.float64)
rs = np.random.RandomState(100 + L)
nulls = []
for _ in range(16):
    sgn = rs.choice([-1, 1], size=len(cids))[np.searchsorted(cids, cl)]
    nd = torch.tensor((Vr * sgn[:, None]).mean(0), dtype=torch.float32)
    nd = (nd / nd.norm()).to(fs._diffmeans.device)
    nulls.append(float(np.median(fs._dir_shift(nd * (base_scale * STRENGTHS[-1])))))
z = _robust_z(dose[-1], np.array(nulls))
print(f"cell wall time: {time.time()-t0:.0f}s", flush=True)

sd = np.array(stored["shifts4"]); nw = np.array(shifts4)
print(f"dose   stored {[round(x,3) for x in stored['dose']]}  mixed {[round(x,3) for x in dose]}")
print(f"z      stored {stored['z']:.2f}  mixed {float(z):.2f}")
print(f"shifts4 corr {np.corrcoef(sd, nw)[0,1]:.4f}  max|diff| {np.max(np.abs(sd-nw)):.4f}  effect scale {np.mean(np.abs(sd)):.3f}")
sp = stored["dose"][-1] > max(stored["nulls"]) and stored["z"] >= 3 and stored["sustained"]
np_ = dose[-1] > max(nulls) and z >= 3 and (max(dose) > 1e-6 and dose[-1] >= 0.5*max(dose))
print(f"gate verdict: stored pass={sp}  mixed pass={np_}  AGREE={sp==np_}")
