#!/usr/bin/env python3
"""Create a SWC-aware Hay L5PC template.

The original `L5PCtemplate_2.hoc` imports Neurolucida `.asc` morphology files
with `Import3d_Neurolucida3`.  Allen Cell Types morphology candidates are SWC
files.  This script generates a local template variant that selects
`Import3d_SWC_read` for `.swc` paths and keeps the original importer for all
other morphology paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SOURCE_ROOT = (
    Path(__file__).resolve().parent
    / "source_models"
    / "neuron_as_deep_net"
    / "L5PC_NEURON_simulation"
)

OLD_INIT = """proc init() {localobj nl,import
\tall = new SectionList()
"""

NEW_INIT = """proc init() {localobj nl,import,strm
\tall = new SectionList()
"""

OLD_IMPORT = """  nl = new Import3d_Neurolucida3()
  nl.quiet = 1
  nl.input($s1)
"""

NEW_IMPORT = """  strm = new StringFunctions()
  if (strm.substr($s1, ".swc") >= 0 || strm.substr($s1, ".SWC") >= 0) {
    nl = new Import3d_SWC_read()
  } else {
    nl = new Import3d_Neurolucida3()
  }
  nl.quiet = 1
  nl.input($s1)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-template", default=str(DEFAULT_SOURCE_ROOT / "L5PCtemplate_2.hoc"))
    parser.add_argument("--output-template", default="docs/verify-arbor-simulator/artifacts/hay_swc_template/L5PCtemplate_swc.hoc")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_template)
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    if OLD_INIT not in text:
        raise ValueError("source template does not match expected init localobj block")
    if OLD_IMPORT not in text:
        raise ValueError("source template does not match expected morphology importer block")
    patched = text.replace(OLD_INIT, NEW_INIT, 1).replace(OLD_IMPORT, NEW_IMPORT, 1)
    _validate_patched_template(patched)
    if args.check_only:
        print("SWC template patch validates")
        return 0
    output = Path(args.output_template)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    print(output)
    return 0


def _validate_patched_template(text: str) -> None:
    required = [
        "Import3d_SWC_read",
        "Import3d_Neurolucida3",
        'strm.substr($s1, ".swc")',
        "localobj nl,import,strm",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"patched template is missing required snippets: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())

