# Check tokenizer/rwkv_world.js against BlinkDL's original TRIE_TOKENIZER (pip package `rwkv`).
# usage: python3 tests/tokenizer-ref.py /path/to/site-packages/rwkv
import sys, json, subprocess, os
sys.path.insert(0, os.path.dirname(sys.argv[1].rstrip('/')))
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(os.path.join(sys.argv[1], 'rwkv_vocab_v20230424.txt'))
texts = ["Hello, world!", "The quick brown fox jumps over the lazy dog.\n\n", "Kehai は日本語も話せます — émigré, naïve, Ωmega.",
  "def f(x):\n    return x ** 2  # squares\n", "🐸🌀✨ ConsciousNode SoftWorks ♡ (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "a" * 50 + " " + "\t\t\r\n" * 3, "1234567890 3.14159 1e-15 0x7fffffff"]
here = os.path.dirname(os.path.abspath(__file__))
js = subprocess.run(['node', '--input-type=module', '-e', f"import {{ RWKVWorldTokenizer }} from '{here}/../tokenizer/rwkv_world.js'; const t = new RWKVWorldTokenizer(); const xs = JSON.parse(process.argv[1]); console.log(JSON.stringify(xs.map(x => [t.encode(x), t.decode(t.encode(x)) === x])));", json.dumps(texts)], capture_output=True, text=True, check=True)
res = json.loads(js.stdout); bad = 0
for x, (ids, rt) in zip(texts, res):
    ref = tok.encode(x); ok = ids == ref and rt
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {len(ref):3d} tokens  {x[:40]!r}")
print('ALL MATCH' if not bad else f'{bad} MISMATCHES'); sys.exit(1 if bad else 0)
