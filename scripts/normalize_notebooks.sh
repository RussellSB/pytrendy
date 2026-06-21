#!/usr/bin/env bash
# Normalise notebook text outputs from list-of-strings to a single joined string.
#
# JupyterLite has a bug where list-form `text` fields (older nbformat style) are
# rendered with commas instead of newlines on initial load.  Running this script
# converts them to the canonical modern format so the pre-saved outputs display
# correctly without needing to re-run cells in the browser.
#
# Usage: ./scripts/normalize_notebooks.sh
#   Defaults to docs/examples/ but accepts an optional path argument.
#
# Example: ./scripts/normalize_notebooks.sh docs/examples

TARGET="${1:-docs/examples}"

python3 - "$TARGET" << 'EOF'
import json
import glob
import sys

target = sys.argv[1]

for nb_path in sorted(glob.glob(f"{target}/**/*.ipynb", recursive=True)):
    with open(nb_path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            if "text" in output and isinstance(output["text"], list):
                output["text"] = "".join(output["text"])
                changed = True
            if "data" in output:
                for mime_key in list(output["data"].keys()):
                    val = output["data"][mime_key]
                    if isinstance(val, list):
                        output["data"][mime_key] = "".join(val)
                        changed = True

    if changed:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"Normalised: {nb_path}")
    else:
        print(f"No change:  {nb_path}")
EOF
