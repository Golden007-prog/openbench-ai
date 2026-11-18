"""Dataset cards (v0.5.0, issue #48).

A dataset card says what a dataset contains, where it came from and its known limits.
``openbench dataset-card PATH`` fills in everything OpenBench can compute (counts, checksum,
category and difficulty mix, length summaries, quality-check results) and leaves the parts only
the author knows (purpose, source and licence, how answers were checked, limitations) as
visible placeholders. It never invents those parts.

The output has no timestamps, so the same dataset always gives the same card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openbench.catalog import DatasetCatalog
from openbench.dataset import DatasetLoader
from openbench.quality import DatasetAuditor

PLACEHOLDER = "TODO (author): "
MANUAL_SECTIONS = (
    ("purpose", "Purpose", "What is this dataset for, and what should a high score mean?"),
    ("source_and_licence", "Source and licence", "Where do the questions and answers come from? Under what licence can they be used?"),
    ("answer_checking", "How answers were checked", "Who wrote and checked the reference answers, and how?"),
    ("limitations", "Known limitations", "What does this dataset not cover? Which answers are ambiguous or disputed?"),
)


def _provenance(path: str | Path) -> dict[str, dict[str, Any]]:
    """Report ``metadata.license`` / ``metadata.source`` only as recorded; never guess a value."""
    records = DatasetLoader.load(path).records
    out: dict[str, dict[str, Any]] = {}
    for key in ("license", "source"):
        values = []
        for r in records:
            v = r.metadata.get(key)
            values.append(v.strip() if isinstance(v, str) and v.strip() else None)
        distinct = sorted({v for v in values if v is not None})
        missing = sum(1 for v in values if v is None)
        shared = distinct[0] if len(distinct) == 1 and missing == 0 else None
        out[key] = {"shared": shared, "distinct": distinct, "records_missing": missing}
    return out


def build_card(path: str | Path) -> dict[str, Any]:
    stats = DatasetCatalog.compute_stats(path)
    audit = DatasetAuditor().audit(path).to_dict()
    summary = audit["summary"]
    findings = audit["findings"]
    provenance = _provenance(path)
    manual = {key: {"title": title, "text": PLACEHOLDER + question} for key, title, question in MANUAL_SECTIONS}
    found = []
    for key, label in (("license", "licence"), ("source", "source")):
        p = provenance[key]
        if p["shared"] is None:
            values = ", ".join(p["distinct"]) or "none"
            found.append(f"{label} values found: {values} ({p['records_missing']} record(s) without one)")
    if found:
        manual["source_and_licence"]["text"] += " Records disagree or lack a value: " + "; ".join(found) + "."
    return {
        "schema_version": "0.5.0",
        "kind": "dataset_card",
        "generated": {
            "name": stats["name"],
            "version": stats["version"],
            "checksum_sha256": stats["checksum"],
            "files": stats["files_count"],
            "records": stats["total_records"],
            "categories": stats["categories"],
            "difficulties": stats["difficulties"],
            "languages": stats["languages"],
            "sources": stats["sources"],
            "license": provenance["license"]["shared"],
            "source": provenance["source"]["shared"],
            "provenance": provenance,
            "prompt_length_chars": stats["prompt_char_length"],
            "answer_length_chars": stats["expected_answer_char_length"],
            "records_with_keywords": stats["records_with_explicit_keywords"],
            "records_with_rationale": stats["records_with_rationale"],
            "quality_checks": {
                "errors": summary["errors"],
                "warnings": summary["warnings"],
                "info": summary["info"],
                "by_check": summary["by_check"],
                "top_findings": [
                    {"check_id": f["check_id"], "severity": f["severity"], "message": f["message"]}
                    for f in findings if f["severity"] in ("error", "warning")
                ][:10],
            },
        },
        "manual": manual,
        "note": "Generated sections come from the data; sections marked TODO must be written by the dataset author.",
    }


def _mix(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    return ", ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _facts(g: dict[str, Any]) -> list[list[Any]]:
    def lengths(field: str) -> str:
        summary = g[field]
        return f"min {summary['min']}, mean {summary['avg']}, max {summary['max']}"

    rows = [
        ["Version", g["version"]],
        ["Records", f"{g['records']} in {g['files']} file(s)"],
        ["SHA-256 checksum", g["checksum_sha256"]],
        ["Categories", _mix(g["categories"])],
        ["Difficulty", _mix(g["difficulties"])],
        ["Languages", _mix(g["languages"])],
        ["Prompt length (characters)", lengths("prompt_length_chars")],
        ["Answer length (characters)", lengths("answer_length_chars")],
        ["Records with keywords / rationale", f"{g['records_with_keywords']} / {g['records_with_rationale']}"],
    ]
    for key, label in (("license", "Licence"), ("source", "Source")):
        if g.get(key):
            rows.append([label, f"{g[key]} (all records)"])
    return rows


def _quality_text(q: dict[str, Any]) -> str:
    return f"{q['errors']} error(s), {q['warnings']} warning(s), {q['info']} info."


def _render_html(card: dict[str, Any]) -> str:
    from openbench.html_report import _page, _table, esc

    g = card["generated"]
    q = g["quality_checks"]
    title = f"Dataset card: {g['name']}"
    parts = [
        f"<h1>{esc(title)}</h1>",
        f'<p class="meta">{esc(card["note"])}</p>',
        "<h2>At a glance</h2>",
        _table(["Field", "Value"], _facts(g)),
        "<h2>Quality checks</h2>",
        f"<p><code>openbench validate-dataset --quality-checks</code>: {esc(_quality_text(q))}</p>",
    ]
    if q["top_findings"]:
        parts.append("<ul>")
        for f in q["top_findings"]:
            parts.append(f"<li><code>{esc(f['check_id'])}</code> ({esc(f['severity'])}): {esc(f['message'])}</li>")
        parts.append("</ul>")
    for section in card["manual"].values():
        parts.append(f'<h2>{esc(section["title"])}</h2><p class="banner">{esc(section["text"])}</p>')
    return _page(title, "".join(parts))


def render_card(card: dict[str, Any], fmt_name: str = "markdown") -> str:
    from openbench.reporting import escape_markdown_cell
    from openbench.reporting_v040 import to_json

    if fmt_name == "json":
        return to_json(card)
    if fmt_name == "html":
        return _render_html(card)
    if fmt_name != "markdown":
        raise ValueError("format must be 'markdown', 'json', or 'html'")
    g = card["generated"]
    q = g["quality_checks"]
    rows = []
    for label, value in _facts(g):
        text = escape_markdown_cell(value)
        if label == "SHA-256 checksum":
            text = f"`{text}`"
        rows.append(f"| {label} | {text} |")
    lines = [
        f"# Dataset card: {escape_markdown_cell(g['name'])}",
        "",
        "> Generated by `openbench dataset-card`. Sections marked **TODO (author)** must be written by the dataset author; OpenBench never fills them in.",
        "",
        "## At a glance",
        "",
        "| Field | Value |",
        "|---|---|",
        *rows,
        "",
        "## Quality checks",
        "",
        f"`openbench validate-dataset --quality-checks`: {_quality_text(q)}",
        "",
    ]
    if q["top_findings"]:
        lines += [f"- `{f['check_id']}` ({f['severity']}): {escape_markdown_cell(f['message'])}" for f in q["top_findings"]] + [""]
    for section in card["manual"].values():
        lines += [f"## {section['title']}", "", f"**{section['text']}**", ""]
    return "\n".join(lines)
