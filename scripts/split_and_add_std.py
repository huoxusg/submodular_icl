#!/usr/bin/env python3
"""Split the local raw log dump into one file per retriever.

The input/output files live under results/logs/, which is a local-only
artifact directory and is not committed to the repository.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / 'results' / 'logs'
infile = LOGS_DIR / 'log_diff_bootstrp.txt'
out_files = {
    'bge-large': LOGS_DIR / 'log_diff_bootstrp_bge-large.txt',
    'bmretriever': LOGS_DIR / 'log_diff_bootstrp_bmretriever.txt',
    'medcpt': LOGS_DIR / 'log_diff_bootstrp_medcpt.txt',
}

text = infile.read_text()

# Split into blocks by the dashed separator line
blocks = re.split(r"^[-]{4,}\s*$", text, flags=re.M)

groups = {k: [] for k in out_files}

retriever_re = re.compile(r"Retriever:\s*([^,\n]+)")

for blk in blocks:
    blk_strip = blk.strip()
    if not blk_strip:
        continue
    # determine retriever
    m = retriever_re.search(blk_strip)
    if not m:
        # if no retriever, skip
        continue
    retr = m.group(1).strip()
    if retr not in groups:
        # ignore unknown retrievers
        continue

    out_lines = []
    for line in blk.splitlines(True):
        out_lines.append(line)

    # re-add separator after block
    groups[retr].append(''.join(out_lines).rstrip() + '\n' + ('-'*50) + '\n\n')

for retr, parts in groups.items():
    if parts:
        (out_files[retr]).write_text(''.join(parts))
    else:
        (out_files[retr]).write_text('')

print('Wrote files:')
for p in out_files.values():
    print(p)
