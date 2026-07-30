"""Naive vs incremental BPE training.

Naive (what HF's BpeTrainer effectively costs at scale): recount all pairs over
all words every merge -> O(corpus_pretokens * vocab_size).

Incremental: maintain pair->count plus an inverted index pair->{word ids}, and on
each merge only touch the words that actually contain the merged pair, applying
delta updates to neighbouring pair counts. -> O(affected words) per merge.
"""
import time, re, collections, heapq, os, pickle, sys

TARGET = int(os.environ.get("MB", "4")) * 1_000_000
CACHE = f"/home/claude/corpus_{TARGET}.pkl"
htmls, texts = pickle.load(open(CACHE, "rb"))
DOCS = [t for t in texts if len(t) > 200]

freq = collections.Counter()
for d in DOCS:
    for w in re.findall(r"\S+", d):
        freq[w] += 1
print(f"{len(DOCS)} docs, {sum(freq.values())} pretokens, {len(freq)} unique\n")

BASE = [(tuple(bytes([b]) for b in w.encode()), c) for w, c in freq.items()]
N_MERGES = int(os.environ.get("MERGES", "1200"))

# ------------------------------------------------------------------ naive
def naive(n_merges):
    words = {w: c for w, c in BASE}
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            for i in range(len(w) - 1):
                pairs[(w[i], w[i + 1])] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        x, y = best
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == x and w[i + 1] == y:
                    out.append(x + y); i += 2
                else:
                    out.append(w[i]); i += 1
            t = tuple(out)
            new[t] = new.get(t, 0) + c
        words = new
    return merges

# ------------------------------------------------------------ incremental
def incremental(n_merges):
    # word i -> mutable list of symbols, plus its count
    W = [list(w) for w, c in BASE]
    C = [c for w, c in BASE]

    pair_count = collections.Counter()
    pair_where = collections.defaultdict(set)   # pair -> set of word ids
    for i, w in enumerate(W):
        c = C[i]
        for j in range(len(w) - 1):
            p = (w[j], w[j + 1])
            pair_count[p] += c
            pair_where[p].add(i)

    # lazy max-heap; stale entries filtered on pop
    heap = [(-v, k) for k, v in pair_count.items()]
    heapq.heapify(heap)

    merges = []
    for _ in range(n_merges):
        best = None
        while heap:
            negv, p = heapq.heappop(heap)
            if pair_count.get(p, 0) == -negv and pair_count[p] > 0:
                best = p; break
        if best is None:
            break
        merges.append(best)
        x, y = best
        xy = x + y
        touched = list(pair_where[best])

        dirty = set()
        for i in touched:
            w = W[i]; c = C[i]
            # remove this word's contribution for pairs it currently has
            j = 0
            out = []
            k = 0
            while k < len(w):
                if k + 1 < len(w) and w[k] == x and w[k + 1] == y:
                    # decrement left neighbour pair
                    if out:
                        lp = (out[-1], x)
                        pair_count[lp] -= c; dirty.add(lp)
                    # decrement the merged pair itself
                    pair_count[best] -= c
                    # decrement right neighbour pair
                    if k + 2 < len(w):
                        rp = (y, w[k + 2])
                        pair_count[rp] -= c; dirty.add(rp)
                    # add new pairs formed by the merged symbol
                    if out:
                        np_ = (out[-1], xy)
                        pair_count[np_] += c; pair_where[np_].add(i); dirty.add(np_)
                    if k + 2 < len(w):
                        np_ = (xy, w[k + 2])
                        pair_count[np_] += c; pair_where[np_].add(i); dirty.add(np_)
                    out.append(xy); k += 2
                else:
                    out.append(w[k]); k += 1
            W[i] = out
        pair_count.pop(best, None)
        pair_where.pop(best, None)
        for p in dirty:
            v = pair_count.get(p, 0)
            if v > 0:
                heapq.heappush(heap, (-v, p))
    return merges

for n in (200, 600, N_MERGES):
    t0 = time.perf_counter(); m1 = naive(n); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); m2 = incremental(n); t2 = time.perf_counter() - t0
    agree = sum(1 for a, b in zip(m1, m2) if a == b)
    print(f"merges={n:>5}   naive {t1:7.2f}s   incremental {t2:6.2f}s   "
          f"speedup {t1/t2:6.1f}x   first-divergence@{agree if agree<len(m1) else 'none'}")
