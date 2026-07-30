"""Profile each stage of a FineWeb/DCLM-style pipeline on identical bytes.
Single core. Reports MB/s of *input HTML* so stages are directly comparable,
plus an internal breakdown of where each stage's time actually goes.
"""
import time, re, hashlib, random, sys, os, pickle, collections
import corpus

TARGET = int(os.environ.get("MB", "5")) * 1_000_000

# ---------------------------------------------------------------- corpus cache
CACHE = f"/home/claude/corpus_{TARGET}.pkl"
if os.path.exists(CACHE):
    htmls, texts = pickle.load(open(CACHE, "rb"))
else:
    htmls, texts = corpus.make_corpus(TARGET)
    pickle.dump((htmls, texts), open(CACHE, "wb"))

HTML_MB = sum(len(x) for x in htmls) / 1e6
TEXT_MB = sum(len(x) for x in texts) / 1e6
print(f"corpus: {len(htmls)} docs, {HTML_MB:.2f} MB HTML, {TEXT_MB:.2f} MB text\n")

results = {}
def report(stage, secs, breakdown=None, basis="html"):
    mb = HTML_MB if basis == "html" else TEXT_MB
    results[stage] = (secs, mb / secs)
    print(f"{stage:<34} {secs:7.2f}s   {mb/secs:7.2f} MB/s")
    if breakdown:
        tot = sum(breakdown.values())
        for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"    {k:<28} {v/tot:6.1%}  ({v:.2f}s)")
    print()

# ============================================================ 1. HTML -> text
from lxml import html as lhtml, etree

BOILER = {"script", "style", "nav", "footer", "header", "aside", "form",
          "noscript", "svg", "iframe", "button"}

t_parse = t_walk = t_ser = 0.0
t0 = time.perf_counter()
out_texts = []
for h in htmls:
    a = time.perf_counter()
    try:
        tree = lhtml.fromstring(h)
    except Exception:
        out_texts.append("")
        continue
    b = time.perf_counter(); t_parse += b - a

    # density-based main-content heuristic, trafilatura/resiliparse style
    best, best_score = None, -1.0
    for node in tree.iter():
        if not isinstance(node.tag, str) or node.tag in BOILER:
            continue
        txt = node.text_content()
        if len(txt) < 200:
            continue
        links = sum(len(a_.text_content() or "") for a_ in node.iter("a"))
        tags = sum(1 for _ in node.iter())
        density = (len(txt) - links) / max(tags, 1)
        if density > best_score:
            best_score, best = density, node
    c = time.perf_counter(); t_walk += c - b

    if best is None:
        out_texts.append("")
    else:
        parts = []
        for p in best.iter("p", "h1", "h2", "h3", "li"):
            s = p.text_content().strip()
            if s:
                parts.append(s)
        out_texts.append("\n".join(parts))
    t_ser += time.perf_counter() - c
report("1. HTML->text extraction", time.perf_counter() - t0,
       {"lxml parse": t_parse, "DOM density walk": t_walk, "serialize": t_ser})

DOCS = [t for t in out_texts if len(t) > 200]
DOC_MB = sum(len(d) for d in DOCS) / 1e6
print(f"    -> {len(DOCS)} docs kept, {DOC_MB:.2f} MB extracted text\n")

# ============================================================ 2. quality filter
STOP = set(list(corpus.VOCAB[:200]))
WORD_RE = re.compile(r"[a-z]+")

t_tok = t_basic = t_rep = 0.0
t0 = time.perf_counter()
kept = 0
for d in DOCS:
    a = time.perf_counter()
    dl = d.lower()
    words = WORD_RE.findall(dl)
    b = time.perf_counter(); t_tok += b - a

    n = len(words)
    if n < 50:
        t_basic += time.perf_counter() - b
        continue
    mean_len = sum(map(len, words)) / n
    stop_frac = sum(1 for w in words if w in STOP) / n
    sym_frac = (dl.count("#") + dl.count("...")) / max(n, 1)
    ok = 3.0 <= mean_len <= 10.0 and stop_frac > 0.005 and sym_frac < 0.1
    c = time.perf_counter(); t_basic += c - b

    # Gopher repetition signals: duplicate n-gram fractions. This is where the
    # time goes in every real implementation.
    if ok:
        for k in (2, 3, 4):
            grams = [tuple(words[i:i+k]) for i in range(n - k + 1)]
            cnt = collections.Counter(grams)
            dup = sum(c_ for c_ in cnt.values() if c_ > 1)
            if grams and dup / len(grams) > 0.30:
                ok = False
                break
    t_rep += time.perf_counter() - c
    if ok:
        kept += 1
report("2. Gopher-style quality filter", time.perf_counter() - t0,
       {"word tokenize": t_tok, "cheap stats": t_basic, "dup n-gram ratios": t_rep})
print(f"    -> {kept}/{len(DOCS)} passed\n")

# ============================================================ 3. MinHash + LSH
NPERM, NGRAM, BANDS = 128, 5, 16
MOD = (1 << 61) - 1
rng = random.Random(7)
PERMS = [(rng.randrange(1, MOD), rng.randrange(0, MOD)) for _ in range(NPERM)]

t_shingle = t_hash = t_sig = t_band = 0.0
t0 = time.perf_counter()
buckets = collections.defaultdict(list)
for i, d in enumerate(DOCS):
    a = time.perf_counter()
    shingles = {d[j:j+NGRAM] for j in range(len(d) - NGRAM + 1)}
    b = time.perf_counter(); t_shingle += b - a

    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
          for s in shingles]
    c = time.perf_counter(); t_hash += c - b

    sig = [min(((aa * h + bb) % MOD) for h in hs) for aa, bb in PERMS]
    d_ = time.perf_counter(); t_sig += d_ - c

    rows = NPERM // BANDS
    for bi in range(BANDS):
        key = (bi, tuple(sig[bi*rows:(bi+1)*rows]))
        buckets[key].append(i)
    t_band += time.perf_counter() - d_
report("3. MinHash fuzzy dedup", time.perf_counter() - t0,
       {"char shingling": t_shingle, "shingle hashing": t_hash,
        "128 permutations": t_sig, "LSH banding": t_band})
cands = sum(1 for v in buckets.values() if len(v) > 1)
print(f"    -> {cands} colliding bands\n")

# ============================================================ 4. BPE training
# Naive-but-indexed trainer on the pretoken frequency table, the way HF does it.
t_pretok = t_count = t_merge = 0.0
t0 = time.perf_counter()
a = time.perf_counter()
freq = collections.Counter()
for d in DOCS:
    for w in re.findall(r"\S+", d):
        freq[w] += 1
words = {tuple(w.encode()): c for w, c in freq.items()}
t_pretok += time.perf_counter() - a

VOCAB_TARGET = 1200
merges = []
for step in range(VOCAB_TARGET):
    a = time.perf_counter()
    pairs = collections.Counter()
    for w, c in words.items():
        for i in range(len(w) - 1):
            pairs[(w[i], w[i+1])] += c
    t_count += time.perf_counter() - a
    if not pairs:
        break
    a = time.perf_counter()
    best = max(pairs.items(), key=lambda kv: kv[1])[0]
    merges.append(best)
    new = {}
    x, y = best
    for w, c in words.items():
        if len(w) < 2:
            new[w] = c; continue
        out, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and w[i] == x and w[i+1] == y:
                out.append(w[i] + w[i+1] if isinstance(w[i], str) else (x, y))
                i += 2
            else:
                out.append(w[i]); i += 1
        new[tuple(out)] = new.get(tuple(out), 0) + c
    words = new
    t_merge += time.perf_counter() - a
report(f"4. BPE training ({VOCAB_TARGET} merges)", time.perf_counter() - t0,
       {"pretokenize+count words": t_pretok, "pair recount per merge": t_count,
        "apply merge": t_merge})
print(f"    -> {len(merges)} merges; unique pretokens={len(freq)}\n")

# ============================================================ 5. packing
SEQ = 2048
t0 = time.perf_counter()
tok_lens = [max(1, len(d) // 4) for d in DOCS]
order = sorted(range(len(tok_lens)), key=lambda i: -tok_lens[i])
bins, waste = [], 0
cur, curlen = [], 0
for i in order:
    L = min(tok_lens[i], SEQ)
    if curlen + L > SEQ:
        waste += SEQ - curlen
        bins.append(cur); cur, curlen = [], 0
    cur.append(i); curlen += L
if cur:
    waste += SEQ - curlen; bins.append(cur)
report("5. sequence packing (greedy FFD)", time.perf_counter() - t0)
tot = len(bins) * SEQ
print(f"    -> {len(bins)} sequences, {waste/tot:.2%} padding waste\n")

# ============================================================ summary
print("=" * 62)
print(f"{'stage':<34} {'share of wall-clock':>26}")
grand = sum(v[0] for v in results.values())
for k, (s, r) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:<34} {s/grand:>10.1%}  ({s:6.2f}s, {r:6.2f} MB/s)")
print(f"\ntotal {grand:.1f}s for {HTML_MB:.2f} MB  =>  {HTML_MB/grand:.2f} MB/s single core")
print(f"extrapolated: 1 PB of HTML on 1000 cores = "
      f"{1e9/(HTML_MB/grand)/1000/3600:.0f} core-hours-equivalent wall days"
      f" -> {1e9/(HTML_MB/grand)/1000/86400:.1f} days")
