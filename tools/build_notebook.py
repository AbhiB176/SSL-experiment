"""Build experiments.ipynb from experiments.py (percent format: `# %%` / `# %% [markdown]`).

    python tools/build_notebook.py

If notes/narrative.md exists, also builds experiments_annotated.ipynb with each
`<!-- section: <title> -->` block inserted under the matching section title.
Code-cell outputs are kept across rebuilds and shared between both notebooks
(matched by cell source), so either notebook can be the one that was run.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'experiments.py'
NOTES = ROOT / 'notes' / 'narrative.md'
CLEAN, ANNOTATED = ROOT / 'experiments.ipynb', ROOT / 'experiments_annotated.ipynb'


def parse(text):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == 'markdown':
            lines = [l[2:] if l.startswith('# ') else l.lstrip('#') for l in lines]
        cells.append((kind, '\n'.join(lines)))
    for line in text.splitlines():
        if line.startswith('# %%'):
            flush()
            kind, buf = ('markdown' if '[markdown]' in line else 'code'), []
        else:
            buf.append(line)
    flush()
    return cells


def parse_notes(path):
    if not path.exists():
        return {}
    notes, title = {}, None
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.match(r'<!-- section: (.*) -->\s*$', line)
        if m:
            title = m.group(1).strip()
            notes[title] = []
        elif title is not None:
            notes[title].append(line)
    return {t: '\n'.join(v).strip() for t, v in notes.items()}


def collect_outputs(*paths):
    out = {}
    for p in paths:
        if p.exists():
            for c in json.loads(p.read_text(encoding='utf-8-sig'))['cells']:
                if c['cell_type'] == 'code' and c.get('outputs'):
                    out.setdefault(''.join(c['source']), (c['outputs'], c.get('execution_count')))
    return out


def write(path, cells, outputs):
    nb_cells = []
    for kind, src in cells:
        cell = {'cell_type': kind, 'metadata': {}, 'source': src.splitlines(keepends=True)}
        if kind == 'code':
            o, n = outputs.get(src, ([], None))
            cell.update(outputs=o, execution_count=n)
        nb_cells.append(cell)
    nb = {'cells': nb_cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {path.name} ({len(nb_cells)} cells)')


def main():
    cells = parse(SRC.read_text(encoding='utf-8'))
    # most recently saved notebook wins when both have outputs for the same cell
    outputs = collect_outputs(*sorted([p for p in (ANNOTATED, CLEAN) if p.exists()],
                                      key=lambda p: p.stat().st_mtime, reverse=True))
    write(CLEAN, cells, outputs)
    notes = parse_notes(NOTES)
    if notes:
        annotated = []
        for kind, src in cells:
            annotated.append((kind, src))
            title = src.lstrip('#').strip() if kind == 'markdown' else None
            if title in notes and notes[title]:
                annotated.append(('markdown', notes[title]))
        write(ANNOTATED, annotated, outputs)


if __name__ == '__main__':
    main()
