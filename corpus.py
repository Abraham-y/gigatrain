"""Generate a synthetic web-like corpus: Zipfian token distribution, realistic
boilerplate-to-content ratio, long-tailed document lengths, and near-duplicates.
Deterministic so all benchmarks see identical bytes.
"""
import random, string, math

random.seed(1337)

# --- Zipfian vocabulary -------------------------------------------------------
def make_vocab(n=60000):
    words = []
    for i in range(n):
        L = 2 + int(abs(random.gauss(0, 3)))
        L = min(max(L, 1), 18)
        words.append("".join(random.choice(string.ascii_lowercase) for _ in range(L)))
    return words

VOCAB = make_vocab()
# Zipf weights: p(rank) ~ 1/rank^1.05, the empirical shape of web text
WEIGHTS = [1.0 / (r ** 1.05) for r in range(1, len(VOCAB) + 1)]
CUM = []
s = 0.0
for w in WEIGHTS:
    s += w
    CUM.append(s)
TOTAL = s

def sample_word():
    x = random.random() * TOTAL
    lo, hi = 0, len(CUM) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if CUM[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return VOCAB[lo]

def sentence(nw=None):
    nw = nw or random.randint(8, 28)
    ws = [sample_word() for _ in range(nw)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + random.choice(". . . ! ?".split())

def paragraph(ns=None):
    return " ".join(sentence() for _ in range(ns or random.randint(2, 7)))

# --- documents ----------------------------------------------------------------
NAV_ITEMS = [sample_word().capitalize() for _ in range(40)]
FOOTER = " | ".join(sample_word().capitalize() for _ in range(25))

def make_html(doc_id):
    """Boilerplate-heavy page. Real CC pages are ~70-90% boilerplate by bytes."""
    # long-tailed content length
    npara = max(1, int(random.paretovariate(1.5)))
    npara = min(npara, 40)
    content = "\n".join(f"<p>{paragraph()}</p>" for _ in range(npara))

    nav = "".join(f'<li><a href="/{w.lower()}/{doc_id % 97}">{w}</a></li>'
                  for w in random.sample(NAV_ITEMS, 18))
    sidebar = "".join(
        f'<div class="widget"><h4>{sample_word().capitalize()}</h4>'
        f'<ul>{"".join(f"<li><a href=#>{sample_word()}</a></li>" for _ in range(6))}</ul></div>'
        for _ in range(4))
    scripts = "".join(
        f'<script>var _cfg{i}={{"k":"{"".join(random.choice(string.hexdigits) for _ in range(48))}"}};</script>'
        for i in range(3))
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><title>{sentence(6)}</title>
<meta name="description" content="{sentence(14)}">
<link rel="stylesheet" href="/static/site.css">{scripts}
<style>.a{{color:#333}}.b{{margin:0 auto}}</style>
</head><body>
<header><nav><ul>{nav}</ul></nav></header>
<div id="wrap"><main><article>
<h1>{sentence(7)}</h1>
<div class="byline">By {sample_word().capitalize()} {sample_word().capitalize()} &middot; {doc_id}</div>
{content}
</article>
<aside class="comments">{"".join(f"<div class=c><span>{sample_word()}</span><p>{sentence()}</p></div>" for _ in range(8))}</aside>
</main><aside id="sidebar">{sidebar}</aside></div>
<footer><p>{FOOTER}</p><p>&copy; 2026 {sample_word().capitalize()}. All rights reserved.</p></footer>
</body></html>"""

def make_corpus(target_bytes):
    """Returns (list_of_html, list_of_plaintext). ~8% near-duplicates injected."""
    htmls, texts = [], []
    total = 0
    i = 0
    while total < target_bytes:
        if htmls and random.random() < 0.08:
            # near-duplicate: copy an earlier doc, perturb a little
            j = random.randrange(len(htmls))
            h = htmls[j]
            k = h.find("<p>")
            if k > 0:
                h = h[:k] + f"<p>{sentence(10)}</p>" + h[k:]
            t = texts[j] + " " + sentence(10)
        else:
            h = make_html(i)
            t = None
        htmls.append(h)
        if t is None:
            # cheap ground-truth text: the paragraphs only
            import re
            t = " ".join(re.findall(r"<p>(.*?)</p>", h, flags=re.S))
        texts.append(t)
        total += len(h)
        i += 1
    return htmls, texts

if __name__ == "__main__":
    h, t = make_corpus(5_000_000)
    hb = sum(len(x) for x in h)
    tb = sum(len(x) for x in t)
    print(f"docs={len(h)} html={hb/1e6:.2f}MB text={tb/1e6:.2f}MB yield={tb/hb:.1%}")
