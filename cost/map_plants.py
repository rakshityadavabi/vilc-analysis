from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "pnp.csv"
DEFAULT_TARGET = Path(__file__).resolve().with_name("hl.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("pnp_to_hl_mapping.csv")

LEADING_PREFIXES = ("dc", "cd", "cdr", "hq")
DROP_TOKENS = {
    "entity",
    "lessee",
    "pwh",
    "trpt",
    "subco",
    "extwh",
    "bls",
    "gvz",
    "wh",
    "entitys",
}

SHORT_GENERIC_TARGETS = {"na", "hq", "la", "mp", "col", "gran"}


@dataclass(frozen=True)
class Candidate:
    name: str
    raw_key: str
    compact_key: str
    sort_key: str


def _read_entries(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_basic(text: str) -> str:
    text = _strip_accents(text)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\\]+", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact_key(text: str) -> str:
    text = _normalize_basic(text)
    tokens = [token for token in text.split() if token not in DROP_TOKENS]
    while tokens and tokens[0] in LEADING_PREFIXES:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def _raw_key(text: str) -> str:
    return _normalize_basic(text)


def _sort_key(text: str) -> str:
    tokens = [token for token in _compact_key(text).split() if token]
    tokens.sort()
    return " ".join(tokens)


def _candidate(name: str) -> Candidate:
    return Candidate(name=name, raw_key=_raw_key(name), compact_key=_compact_key(name), sort_key=_sort_key(name))


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def _is_export_like(text: str) -> bool:
    lowered = text.lower().strip()
    return lowered.startswith("export from") or lowered.startswith("import from") or lowered.startswith("export ")


def _score_pair(source: Candidate, target: Candidate) -> tuple[float, str]:
    scores = [
        (1.0 if source.raw_key and source.raw_key == target.raw_key else 0.0, "raw_exact"),
        (0.995 if source.compact_key and source.compact_key == target.compact_key else 0.0, "compact_exact"),
        (0.99 if source.sort_key and source.sort_key == target.sort_key else 0.0, "sort_exact"),
        (_ratio(source.raw_key, target.raw_key), "raw_fuzzy"),
        (_ratio(source.compact_key, target.compact_key), "compact_fuzzy"),
        (_ratio(source.sort_key, target.sort_key), "sort_fuzzy"),
        (_jaccard(source.compact_key, target.compact_key), "jaccard"),
    ]
    best_score, best_method = max(scores, key=lambda item: item[0])

    if source.compact_key and target.compact_key:
        if source.compact_key in target.compact_key or target.compact_key in source.compact_key:
            best_score = max(best_score, 0.94)
            if best_method == "raw_fuzzy":
                best_method = "substring"

    if source.sort_key and target.sort_key:
        source_tokens = set(source.sort_key.split())
        target_tokens = set(target.sort_key.split())
        if source_tokens & target_tokens and len(source_tokens & target_tokens) >= 2:
            best_score = max(best_score, 0.9)

    target_compact_len = len(target.compact_key)
    if target_compact_len <= 2 and best_method not in {"raw_exact", "compact_exact", "sort_exact"}:
        best_score *= 0.2
    elif target.compact_key in SHORT_GENERIC_TARGETS and best_method not in {"raw_exact", "compact_exact", "sort_exact"}:
        best_score *= 0.5

    return min(best_score, 1.0), best_method


def _select_best(source: Candidate, targets: list[Candidate]) -> tuple[Candidate, float, str]:
    if _is_export_like(source.name):
        for target in targets:
            if target.compact_key == "caribe export" or target.raw_key == "caribe export entity":
                return target, 0.85, "generic_export"
        for target in targets:
            if "export" in target.raw_key and "entity" in target.raw_key:
                return target, 0.6, "generic_export"

    best_target = targets[0]
    best_score = -1.0
    best_method = ""
    best_len_gap = abs(len(source.compact_key) - len(best_target.compact_key))

    for target in targets:
        score, method = _score_pair(source, target)
        len_gap = abs(len(source.compact_key) - len(target.compact_key))
        candidate_rank = (score, -len_gap)
        best_rank = (best_score, -best_len_gap)
        if candidate_rank > best_rank:
            best_target = target
            best_score = score
            best_method = method
            best_len_gap = len_gap

    return best_target, best_score, best_method


def build_mapping(source_path: Path, target_path: Path) -> list[dict[str, str]]:
    sources = _read_entries(source_path)
    targets = _read_entries(target_path)
    target_candidates = [_candidate(name) for name in targets]

    rows: list[dict[str, str]] = []
    for index, source_name in enumerate(sources, start=1):
        source_candidate = _candidate(source_name)
        target_candidate, score, method = _select_best(source_candidate, target_candidates)
        rows.append(
            {
                "source_index": str(index),
                "source_name": source_name,
                "source_key": source_candidate.compact_key,
                "target_index": str(targets.index(target_candidate.name) + 1),
                "target_name": target_candidate.name,
                "target_key": target_candidate.compact_key,
                "match_score": f"{score:.3f}",
                "match_method": method,
                "review_flag": "yes" if score < 0.75 else "no",
            }
        )
    return rows


def write_mapping(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "source_name",
        "source_key",
        "target_index",
        "target_name",
        "target_key",
        "match_score",
        "match_method",
        "review_flag",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Map pnp plant names to hl plant names.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source list file (default: pnp.csv)")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target list file (default: hl.csv)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output mapping CSV path")
    args = parser.parse_args()

    rows = build_mapping(args.source, args.target)
    write_mapping(rows, args.output)
    print(f"Wrote {len(rows)} mappings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())