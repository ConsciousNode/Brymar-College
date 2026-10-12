# legacy/v3.3.4

The model engine from Brymar College ≤ v3.3.4, kept only so the audit can be re-run:

```bash
node legacy/v3.3.4/tests/weight-map.mjs [real-rwkv7.safetensors]   # what loading a real checkpoint did
node legacy/v3.3.4/tests/gradcheck.mjs                            # its backward vs finite differences
```

It is not used by Brymar any more. See [../AUDIT.md](../AUDIT.md) for what was wrong, and `model/rwkv7.js` for the replacement.
