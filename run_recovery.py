#!/usr/bin/env python3
"""Driver for the held-out recovery experiment. Needs a key; not in the gate.

    python3 run_recovery.py <provider> <model> <n>

Every response is cached by (provider, model, prompt, nonce, max_tokens), so a
re-run is free and the numbers are reproducible without re-paying.
"""
import sys
import proposer as P
import vocab as V

prov_name, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
style = sys.argv[4] if len(sys.argv) > 4 else "both"
styles = {"constrained": (True,), "freeform": (False,), "both": (True, False)}[style]
prov = P.PROVIDERS[prov_name](model)
V.run(prov.ask, f"{prov_name} / {model} [{style}]", n=n, styles=styles)
