"""Independent translation-quality check (reviewer 2 #9, reviewer 3 #4).

The manuscript's translations were produced with GPT-4o-mini. Here every origin sentence is re-translated with an
independent open-source system (NLLB-200) and the two translations are compared:
  * LaBSE cosine similarity: original<->GPT, original<->NLLB, GPT<->NLLB
  * chrF between the two systems' outputs
  * round-trip (back-translation) fidelity of the NLLB output
Everything is reported per translation direction and per emotion. Model names, versions, device and date are written to
translation_manifest.json (reviewer 2 #13).

Inputs : emotion_data/dataset_experiment{2,3,4}_*.csv (row-aligned; column original_text holds the untranslated sentence)
Outputs: outputs/translation_check/*.csv, translation_manifest.json
Env vars: NLLB_MODEL (default facebook/nllb-200-distilled-1.3B), N_PER_ORIGIN (0 = all 3000 per origin; use e.g. 4 for a dry run)
"""
import os, sys, json, time, subprocess, datetime
from pathlib import Path


def ensure(pkg, import_name=None, no_deps=False):
    # --no-deps for sentence-transformers: never touch the already-installed torch/transformers (a training run may be using them)
    try:
        __import__(import_name or pkg)
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + (["--no-deps"] if no_deps else []) + [pkg]
        subprocess.run(cmd, check=True)


ensure("sacrebleu")
ensure("sentence-transformers", "sentence_transformers", no_deps=True)

import numpy as np
import pandas as pd
import torch
import transformers
import sentence_transformers
import sacrebleu
from sacrebleu.metrics import CHRF
from scipy import stats
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

DATA = Path("emotion_data")
OUT = Path("outputs/translation_check")
OUT.mkdir(parents=True, exist_ok=True)
NLLB_MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")
LABSE_MODEL = "sentence-transformers/LaBSE"
N_PER_ORIGIN = int(os.environ.get("N_PER_ORIGIN", "0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG = {"ko": "kor_Hang", "zh": "zho_Hans", "en": "eng_Latn"}
FILES = {"ko": "dataset_experiment2_korean.csv", "zh": "dataset_experiment3_chinese.csv", "en": "dataset_experiment4_english.csv"}
BATCH = 64 if DEVICE == "cuda" else 8
NUM_BEAMS = 4
MAX_NEW = 256


def section(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90, flush=True)


def load_base():
    frames = {k: pd.read_csv(DATA / f) for k, f in FILES.items()}
    base = frames["ko"][["original_text", "original_language", "emotion", "split"]].copy()
    for k in FILES:
        assert (frames[k]["original_text"].values == base["original_text"].values).all(), "files are not row-aligned"
        base[f"gpt_{k}"] = frames[k]["text"].astype(str).values
    base = base.reset_index().rename(columns={"index": "row_id"})
    if N_PER_ORIGIN > 0:
        base = base.groupby("original_language", group_keys=False).head(N_PER_ORIGIN).reset_index(drop=True)
    return base


def nllb_translate(texts, src, tgt, tok, model):
    tok.src_lang = LANG[src]
    forced = tok.convert_tokens_to_ids(LANG[tgt])
    order = np.argsort([len(t) for t in texts])
    out = [None] * len(texts)
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW)
        for j, d in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = d
    return out


def translate_all(base):
    path = OUT / "translations_nllb.csv"
    if path.exists() and N_PER_ORIGIN == 0:
        print(f"Reusing cached {path}")
        return pd.read_csv(path)
    section(f"NLLB translation + back-translation  model={NLLB_MODEL}  device={DEVICE}")
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
    rows = []
    for origin in LANG:
        part = base[base.original_language == origin].reset_index(drop=True)
        for target in LANG:
            if target == origin:
                continue
            t0 = time.time()
            fwd = nllb_translate(part["original_text"].tolist(), origin, target, tok, model)
            back = nllb_translate(fwd, target, origin, tok, model)
            print(f"{origin}->{target}: {len(part)} sentences, {time.time() - t0:.0f}s", flush=True)
            rows.append(pd.DataFrame({
                "row_id": part["row_id"], "origin": origin, "target": target, "direction": f"{origin}->{target}",
                "emotion": part["emotion"], "split": part["split"], "original": part["original_text"],
                "gpt": part[f"gpt_{target}"], "nllb": fwd, "back": back,
            }))
    df = pd.concat(rows, ignore_index=True)
    if N_PER_ORIGIN == 0:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return df


def embed_cos(model, a, b):
    ea = model.encode(a, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    eb = model.encode(b, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    return (ea * eb).sum(axis=1)


def main():
    t_start = time.time()
    base = load_base()
    print(f"rows={len(base)}  origins={base.original_language.value_counts().to_dict()}")
    df = translate_all(base)

    section("LaBSE similarities + chrF")
    labse = SentenceTransformer(LABSE_MODEL, device=DEVICE)
    df["sim_orig_gpt"] = embed_cos(labse, df["original"].tolist(), df["gpt"].tolist())
    df["sim_orig_nllb"] = embed_cos(labse, df["original"].tolist(), df["nllb"].tolist())
    df["sim_gpt_nllb"] = embed_cos(labse, df["gpt"].tolist(), df["nllb"].tolist())
    df["sim_orig_back"] = embed_cos(labse, df["original"].tolist(), df["back"].tolist())
    chrf = CHRF()
    df["chrf_gpt_vs_nllb"] = [chrf.sentence_score(h, [r]).score for h, r in zip(df["nllb"], df["gpt"])]
    df["chrf_orig_vs_back"] = [chrf.sentence_score(h, [r]).score for h, r in zip(df["back"], df["original"])]
    df["len_ratio_gpt"] = df["gpt"].str.len() / df["original"].str.len().clip(lower=1)
    df["len_ratio_nllb"] = df["nllb"].str.len() / df["original"].str.len().clip(lower=1)
    df.to_csv(OUT / "pairs_with_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = ["sim_orig_gpt", "sim_orig_nllb", "sim_gpt_nllb", "sim_orig_back", "chrf_gpt_vs_nllb", "chrf_orig_vs_back"]
    by_dir = df.groupby("direction")[metrics].agg(["mean", "std"]).round(4)
    by_dir["n"] = df.groupby("direction").size()
    by_dir.to_csv(OUT / "summary_by_direction.csv")
    by_emo = df.groupby("emotion")[metrics].agg(["mean", "std"]).round(4)
    by_emo.to_csv(OUT / "summary_by_emotion.csv")
    by_both = df.groupby(["direction", "emotion"])[metrics].mean().round(4)
    by_both.to_csv(OUT / "summary_by_direction_emotion.csv")

    section("Summary: mean LaBSE similarity by direction")
    print(df.groupby("direction")[["sim_orig_gpt", "sim_orig_nllb", "sim_gpt_nllb", "sim_orig_back"]].mean().round(3).to_string())
    section("Summary: mean LaBSE similarity (original<->GPT translation) by emotion")
    print(df.groupby("emotion")[["sim_orig_gpt", "sim_orig_nllb", "sim_gpt_nllb"]].mean().round(3).to_string())
    kw = {}
    for m in ["sim_orig_gpt", "sim_orig_nllb"]:
        groups = [g[m].values for _, g in df.groupby("emotion")]
        if len(groups) > 1 and min(len(g) for g in groups) > 1:
            h, p = stats.kruskal(*groups)
            kw[m] = {"H": float(h), "p": float(p)}
            print(f"Kruskal-Wallis across emotions on {m}: H={h:.2f}, p={p:.4g}")
    thr = {}
    for m in ["sim_orig_gpt", "sim_orig_nllb", "sim_gpt_nllb"]:
        thr[m] = float((df[m] < 0.7).mean())
        print(f"share of pairs with {m} < 0.70: {thr[m] * 100:.2f}%")
    df.nsmallest(60, "sim_orig_gpt").to_csv(OUT / "lowest_similarity_gpt.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "",
        "nllb_model": NLLB_MODEL, "labse_model": LABSE_MODEL, "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW,
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__, "sacrebleu": sacrebleu.__version__,
        "n_pairs": int(len(df)), "kruskal_wallis": kw, "share_below_0.70": thr,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    (OUT / "translation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDone.", json.dumps({k: manifest[k] for k in ["nllb_model", "n_pairs", "runtime_sec"]}))


if __name__ == "__main__":
    main()
