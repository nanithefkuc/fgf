#!/usr/bin/env python3
"""Inventory integer routing thresholds and tuning parameters in fgf.

The scan is deliberately source-only: it reads the kernel dispatch surface,
the extension-storage contracts, and the row-operation facade, and reports
every integer ``const`` item plus integer literals in the contexts that can
select a route or set a kernel schedule.  Keys are file + name or expression,
never line numbers, so the output is stable under unrelated edits.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

TARGET_GLOBS = (
    "src/kernel/**/*.rs",
    "src/field/dispatch.rs",
    "src/field/extension/storage.rs",
    "src/ops/**/*.rs",
)

SKIP_DIR_NAMES = {"tests"}
SKIP_FILE_NAMES = {"tests.rs"}

INTEGER_TYPES = {
    "usize",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
}
MASK_TYPES = {"__mmask8", "__mmask16", "__mmask32", "__mmask64"}

ITEM_KEYWORDS = {
    "mod",
    "fn",
    "struct",
    "enum",
    "union",
    "trait",
    "impl",
    "const",
    "static",
    "type",
    "use",
    "macro_rules",
    "extern",
}

SCOPE_KEYWORDS = ("fn", "impl", "trait", "mod", "struct", "enum", "union", "macro_rules")

COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}

ALLOWED_EVAL_OPS = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "~", "(", ")"}

ROUTING = "routing_threshold"
TUNING = "tuning_parameter"
STRUCTURAL = "structural"

UNITS = {"bytes", "lanes", "elements", "rows", "terms", "count", "other"}


class Token:
    __slots__ = ("text", "kind", "start", "end")

    def __init__(self, text: str, kind: str, start: int, end: int) -> None:
        self.text = text
        self.kind = kind
        self.start = start
        self.end = end

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.text!r}, {self.kind}, {self.start}:{self.end})"


class Comment:
    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


def tokenize(source: str) -> tuple[list[Token], list[Comment]]:
    """Split Rust source into tokens and comments.

    The lexer keeps offsets, understands nested block comments, raw strings,
    byte strings, and character literals, and leaves comments out of the
    token stream used for structure.
    """

    tokens: list[Token] = []
    comments: list[Comment] = []
    n = len(source)
    i = 0

    def is_ident_start(c: str) -> bool:
        return c.isascii() and (c.isalpha() or c == "_")

    def is_ident_cont(c: str) -> bool:
        return c.isascii() and (c.isalnum() or c == "_")

    while i < n:
        c = source[i]
        if c.isspace():
            i += 1
            continue
        if source.startswith("//", i):
            start = i
            newline = source.find("\n", i)
            end = n if newline < 0 else newline
            comments.append(Comment(source[start:end], start, end))
            i = end
            continue
        if source.startswith("/*", i):
            start = i
            depth = 1
            i += 2
            while i < n and depth:
                if source.startswith("/*", i):
                    depth += 1
                    i += 2
                elif source.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            comments.append(Comment(source[start:i], start, i))
            continue
        if c == "#":
            tokens.append(Token("#", "punct", i, i + 1))
            i += 1
            continue
        if c == '"' or (c in "br" and i + 1 < n and source[i + 1] == '"'):
            start = i
            if c == "b":
                i += 1
            if source.startswith("r", i):
                i += 1
                hashes = 0
                while i < n and source[i] == "#":
                    hashes += 1
                    i += 1
                if i < n and source[i] == '"':
                    i += 1
                    closing = '"' + "#" * hashes
                    end = source.find(closing, i)
                    i = n if end < 0 else end + len(closing)
                    tokens.append(Token(source[start:i], "string", start, i))
                    continue
            else:
                if i < n and source[i] == '"':
                    i += 1
                    while i < n:
                        if source[i] == "\\":
                            i += 2
                        elif source[i] == '"':
                            i += 1
                            break
                        else:
                            i += 1
                    tokens.append(Token(source[start:i], "string", start, i))
                    continue
            # A bare `b` or `r` identifier that was not a string.
            i = start + 1
            tokens.append(Token(source[start:i], "ident", start, i))
            continue
        if c == "'":
            start = i
            # Lifetime (no closing quote immediately) or character literal.
            j = i + 1
            if j < n and source[j] == "\\":
                j += 2
                while j < n and source[j] != "'":
                    j += 1
                if j < n:
                    j += 1
                tokens.append(Token(source[start:j], "char", start, j))
                i = j
                continue
            if j + 1 < n and source[j + 1] == "'":
                tokens.append(Token(source[start : j + 2], "char", start, j + 2))
                i = j + 2
                continue
            j += 1
            while j < n and is_ident_cont(source[j]):
                j += 1
            tokens.append(Token(source[start:j], "lifetime", start, j))
            i = j
            continue
        if is_ident_start(c):
            start = i
            i += 1
            while i < n and is_ident_cont(source[i]):
                i += 1
            tokens.append(Token(source[start:i], "ident", start, i))
            continue
        if c.isdigit():
            start = i
            if source.startswith(("0x", "0X"), i):
                i += 2
                while i < n and (source[i].isdigit() or source[i].lower() in "abcdef_"):
                    i += 1
            elif source.startswith(("0b", "0B"), i):
                i += 2
                while i < n and source[i] in "01_":
                    i += 1
            elif source.startswith(("0o", "0O"), i):
                i += 2
                while i < n and source[i] in "01234567_":
                    i += 1
            else:
                while i < n and (source[i].isdigit() or source[i] == "_"):
                    i += 1
                if i < n and source[i] == "." and not source.startswith("..", i):
                    j = i + 1
                    if j < n and source[j].isdigit():
                        i = j
                        while i < n and (source[i].isdigit() or source[i] == "_"):
                            i += 1
                if i < n and source[i] in "eE":
                    j = i + 1
                    if j < n and source[j] in "+-":
                        j += 1
                    if j < n and source[j].isdigit():
                        i = j
                        while i < n and (source[i].isdigit() or source[i] == "_"):
                            i += 1
            # Integer/float suffix.
            while i < n and is_ident_cont(source[i]):
                i += 1
            tokens.append(Token(source[start:i], "number", start, i))
            continue
        matched = None
        for op in ("..=", "...", "<<=", ">>=", "::", "->", "=>", ">=", "<=", "==", "!=", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ".."):
            if source.startswith(op, i):
                matched = op
                break
        if matched is not None:
            tokens.append(Token(matched, "punct", i, i + len(matched)))
            i += len(matched)
            continue
        tokens.append(Token(c, "punct", i, i + 1))
        i += 1
    return tokens, comments
def match_delim(tokens: list[Token], index: int) -> int:
    """Return the index of the delimiter closing the one at ``index``."""

    opener = tokens[index].text
    closer = {"(": ")", "[": "]", "{": "}"}[opener]
    depth = 0
    for j in range(index, len(tokens)):
        text = tokens[j].text
        if text == opener:
            depth += 1
        elif text == closer:
            depth -= 1
            if depth == 0:
                return j
    return len(tokens) - 1


def match_delim_backward(tokens: list[Token], index: int) -> int:
    """Return the index of the delimiter opening the one at ``index``."""

    closer = tokens[index].text
    opener = {")": "(", "]": "[", "}": "{"}[closer]
    depth = 0
    for j in range(index, -1, -1):
        text = tokens[j].text
        if text == closer:
            depth += 1
        elif text == opener:
            depth -= 1
            if depth == 0:
                return j
    return 0


def is_cfg_test(attribute: str) -> bool:
    compact = re.sub(r"\s+", "", attribute)
    if not re.search(r"\bcfg\(", compact):
        return False
    compact = re.sub(r'"[^"]*"', '""', compact)
    if "not(test)" in compact:
        return False
    return bool(re.search(r"\btest\b", compact))


def blank_span(chars: list[str], start: int, end: int) -> None:
    for i in range(start, min(end, len(chars))):
        if chars[i] != "\n":
            chars[i] = " "


def item_end(tokens: list[Token], keyword_index: int) -> int:
    keyword = tokens[keyword_index].text
    if keyword in {"use", "type", "static", "const"}:
        for j in range(keyword_index + 1, len(tokens)):
            if tokens[j].text == ";":
                return j
    parens = 0
    brackets = 0
    angles = 0
    j = keyword_index
    while j < len(tokens):
        text = tokens[j].text
        if text == "(":
            parens += 1
        elif text == ")":
            parens = max(0, parens - 1)
        elif text == "[":
            brackets += 1
        elif text == "]":
            brackets = max(0, brackets - 1)
        elif text == "<":
            angles += 1
        elif text == ">":
            angles = max(0, angles - 1)
        elif text == ">>":
            angles = max(0, angles - 2)
        elif text in {"{", ";"} and parens == 0 and brackets == 0 and angles == 0:
            if text == ";":
                return j
            return match_delim(tokens, j)
        j += 1
    return len(tokens) - 1


def strip_cfg_test(source: str) -> str:
    """Blank every ``#[cfg(test)]`` item or module, preserving newlines."""

    tokens, _ = tokenize(source)
    chars = list(source)
    i = 0
    while i < len(tokens):
        if tokens[i].text == "#" and i + 1 < len(tokens) and tokens[i + 1].text == "[":
            attribute_start = tokens[i].start
            attributes: list[str] = []
            j = i
            while j + 1 < len(tokens) and tokens[j].text == "#" and tokens[j + 1].text == "[":
                close = match_delim(tokens, j + 1)
                attributes.append(source[tokens[j].start : tokens[close].end])
                j = close + 1
            k = j
            while k < len(tokens):
                text = tokens[k].text
                if text == "pub":
                    k += 1
                    if k < len(tokens) and tokens[k].text == "(":
                        k = match_delim(tokens, k) + 1
                    continue
                if text in {"unsafe", "async", "default"}:
                    k += 1
                    continue
                if text == "extern":
                    k += 1
                    if k < len(tokens) and tokens[k].kind == "string":
                        k += 1
                    continue
                break
            if k < len(tokens) and tokens[k].text in ITEM_KEYWORDS and any(
                is_cfg_test(attribute) for attribute in attributes
            ):
                end_token = item_end(tokens, k)
                blank_span(chars, attribute_start, tokens[end_token].end)
                i = end_token + 1
                continue
        i += 1
    return "".join(chars)


def normalize_expr(source: str, tokens: list[Token], start: int, end: int) -> str:
    """Exact source text for a token range, with comments and runs of space removed."""

    text = source[tokens[start].start : tokens[end].end]
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_expr(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def parse_number(text: str) -> int | None:
    """Parse a Rust integer literal, returning ``None`` for floats."""

    if not text or not text[0].isdigit():
        return None
    value_text = text.replace("_", "")
    value_text = re.sub(r"(?:u|i)(?:8|16|32|64|128|size)$", "", value_text)
    try:
        if value_text.startswith(("0x", "0X")):
            return int(value_text[2:], 16)
        if value_text.startswith(("0b", "0B")):
            return int(value_text[2:], 2)
        if value_text.startswith(("0o", "0O")):
            return int(value_text[2:], 8)
        if any(ch in value_text for ch in ".eE"):
            return None
        return int(value_text, 10)
    except ValueError:
        return None


MAX_CONSTANTS = {
    ("u8", "MAX"): 2**8 - 1,
    ("u16", "MAX"): 2**16 - 1,
    ("u32", "MAX"): 2**32 - 1,
    ("u64", "MAX"): 2**64 - 1,
    ("u128", "MAX"): 2**128 - 1,
    ("usize", "MAX"): 2**64 - 1,
    ("i8", "MAX"): 2**7 - 1,
    ("i16", "MAX"): 2**15 - 1,
    ("i32", "MAX"): 2**31 - 1,
    ("i64", "MAX"): 2**63 - 1,
    ("i128", "MAX"): 2**127 - 1,
    ("isize", "MAX"): 2**63 - 1,
    ("i8", "MIN"): -(2**7),
    ("i16", "MIN"): -(2**15),
    ("i32", "MIN"): -(2**31),
    ("i64", "MIN"): -(2**63),
    ("i128", "MIN"): -(2**127),
    ("isize", "MIN"): -(2**63),
}


def eval_numeric(tokens: list[Token], known: dict[str, int]) -> int | None:
    """Evaluate a rustc integer expression, or return ``None``.

    Only literals, named numeric constants from the same file, unary and
    binary integer operators, and ``as`` casts are accepted.
    """

    parts: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.text == "as":
            i += 1
            depth = 0
            while i < len(tokens):
                text = tokens[i].text
                if text in {"<", "[", "("}:
                    depth += 1
                elif text in {">", "]", ")"}:
                    depth = max(0, depth - 1)
                elif depth == 0 and (text in ALLOWED_EVAL_OPS or text == ";"):
                    break
                i += 1
            continue
        if token.kind == "number":
            value = parse_number(token.text)
            if value is None:
                return None
            parts.append(f"({value})")
            i += 1
            continue
        if token.kind == "ident":
            name = token.text
            if (
                i + 2 < len(tokens)
                and tokens[i + 1].text == "::"
                and tokens[i + 2].kind == "ident"
            ):
                key = (name, tokens[i + 2].text)
                if key in MAX_CONSTANTS:
                    parts.append(f"({MAX_CONSTANTS[key]})")
                    i += 3
                    continue
            if name in known:
                parts.append(f"(({known[name]}))")
                i += 1
                continue
            return None
        if token.text in ALLOWED_EVAL_OPS:
            parts.append("//" if token.text == "/" else token.text)
            i += 1
            continue
        return None
    expression = " ".join(parts)
    if not expression:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.Expression,
                ast.Constant,
                ast.BinOp,
                ast.UnaryOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.FloorDiv,
                ast.Mod,
                ast.LShift,
                ast.RShift,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.Invert,
                ast.USub,
                ast.UAdd,
            ),
        ):
            return None
    try:
        result = eval(compile(tree, "<threshold>", "eval"), {"__builtins__": {}})
    except (ArithmeticError, ValueError):
        return None
    return result if isinstance(result, int) else None
class SourceModel:
    def __init__(self, relative: str, source: str) -> None:
        self.relative = relative
        self.source = strip_cfg_test(source)
        self.tokens, self.comments = tokenize(self.source)
        self.scopes, self.scope_opens, self.scope_decls = self._build_scopes()
        self.generic_tokens = self._build_generic_tokens()
        self.attribute_tokens = self._build_attribute_tokens()

    def _scope_label(self, start: int, brace: int) -> tuple[str | None, int | None, int | None]:
        keyword_index = None
        keyword = None
        for i in range(start, brace):
            text = self.tokens[i].text
            if text in SCOPE_KEYWORDS:
                keyword_index = i
                keyword = text
                break
        if keyword is None or keyword_index is None:
            return None, None, None
        if keyword == "fn":
            name_index = keyword_index + 1
            while name_index < brace and self.tokens[name_index].kind != "ident":
                name_index += 1
            if name_index < brace:
                return f"fn {self.tokens[name_index].text}", self.tokens[brace].start, keyword_index
            return "fn", self.tokens[brace].start, keyword_index
        if keyword == "impl":
            text = normalize_expr(self.source, self.tokens, keyword_index, brace - 1)
            while "<" in text:
                stripped = re.sub(r"<[^<>]*>", "", text)
                if stripped == text:
                    break
                text = stripped
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 90:
                text = text[:90].rstrip()
            return text, self.tokens[brace].start, keyword_index
        if keyword in {"trait", "mod", "struct", "enum", "union"}:
            name_index = keyword_index + 1
            while name_index < brace and self.tokens[name_index].kind != "ident":
                name_index += 1
            if name_index < brace:
                return (
                    f"{keyword} {self.tokens[name_index].text}",
                    self.tokens[brace].start,
                    keyword_index,
                )
            return keyword, self.tokens[brace].start, keyword_index
        if keyword == "macro_rules":
            name_index = keyword_index + 2
            while name_index < brace and self.tokens[name_index].kind != "ident":
                name_index += 1
            if name_index < brace:
                return (
                    f"macro_rules! {self.tokens[name_index].text}",
                    self.tokens[brace].start,
                    keyword_index,
                )
            return "macro_rules!", self.tokens[brace].start, keyword_index
        return None, None, None

    def _build_scopes(self) -> tuple[list[tuple[str, ...]], list[int], list[int]]:
        labels: list[tuple[str, ...]] = []
        opens: list[int] = []
        declarations: list[int] = []
        stack: list[tuple[str | None, int | None, int | None]] = []
        current: list[str] = []
        current_opens: list[int] = []
        current_declarations: list[int] = []
        last_boundary = 0
        for i, token in enumerate(self.tokens):
            if token.text == "{":
                label, open_start, declaration = self._scope_label(last_boundary, i)
                stack.append((label, open_start, declaration))
                if label is not None and open_start is not None and declaration is not None:
                    current.append(label)
                    current_opens.append(open_start)
                    current_declarations.append(declaration)
                labels.append(tuple(current))
                opens.append(current_opens[-1] if current_opens else 0)
                declarations.append(current_declarations[-1] if current_declarations else 0)
                last_boundary = i + 1
            elif token.text == "}":
                labels.append(tuple(current))
                opens.append(current_opens[-1] if current_opens else 0)
                declarations.append(current_declarations[-1] if current_declarations else 0)
                if stack:
                    label, open_start, declaration = stack.pop()
                    if label is not None and current and current[-1] == label:
                        current.pop()
                        current_opens.pop()
                        current_declarations.pop()
                last_boundary = i + 1
            else:
                labels.append(tuple(current))
                opens.append(current_opens[-1] if current_opens else 0)
                declarations.append(current_declarations[-1] if current_declarations else 0)
        return labels, opens, declarations

    def _build_generic_tokens(self) -> set[int]:
        marked: set[int] = set()
        i = 0
        while i < len(self.tokens):
            if self.tokens[i].text != "<":
                i += 1
                continue
            previous = self.tokens[i - 1].text if i > 0 else ""
            before_character = self.source[self.tokens[i].start - 1] if self.tokens[i].start > 0 else ""
            turbofish = previous == "::" and i > 1 and (
                self.tokens[i - 2].kind == "ident" or self.tokens[i - 2].text in {">", ">>"}
            )
            attached = previous and not before_character.isspace() and (
                self.tokens[i - 1].kind == "ident"
                or previous in {">", ">>", "]", "::"}
            )
            if not (turbofish or attached):
                i += 1
                continue
            depth = 1
            j = i + 1
            while j < len(self.tokens) and depth:
                text = self.tokens[j].text
                if text == "<":
                    depth += 1
                elif text == ">":
                    depth -= 1
                elif text == ">>":
                    depth = max(0, depth - 2)
                j += 1
            if depth == 0:
                for k in range(i, j):
                    marked.add(k)
                i = j
            else:
                i += 1
        return marked

    def _build_attribute_tokens(self) -> set[int]:
        marked: set[int] = set()
        i = 0
        while i < len(self.tokens):
            if self.tokens[i].text == "#" and i + 1 < len(self.tokens) and self.tokens[i + 1].text == "[":
                close = match_delim(self.tokens, i + 1)
                for k in range(i, close + 1):
                    marked.add(k)
                i = close + 1
            else:
                i += 1
        return marked

    def scope_key(self, index: int) -> str:
        return "::".join(self.scopes[index])

    def scope_open(self, index: int) -> int:
        return self.scope_opens[index]

    def scope_decl(self, index: int) -> int:
        return self.scope_decls[index]

    def _declaration_start(self, index: int) -> int:
        start = index
        modifiers = {"pub", "const", "unsafe", "async", "default", "extern"}
        changed = True
        while changed:
            changed = False
            if start > 0 and self.tokens[start - 1].text in modifiers:
                start -= 1
                changed = True
            elif start >= 2 and self.tokens[start - 1].text == "]":
                opening = match_delim_backward(self.tokens, start - 1)
                if opening > 0 and self.tokens[opening - 1].text == "#":
                    start = opening - 1
                    changed = True
            elif start > 1 and self.tokens[start - 1].text == ")":
                opening = match_delim_backward(self.tokens, start - 1)
                if opening > 0 and self.tokens[opening - 1].text == "pub":
                    start = opening - 1
                    changed = True
        return start

    def citation(
        self, index: int, family_citations: dict[str, str], use_scope: bool = True
    ) -> str | None:
        declaration = self.scope_decl(index) if use_scope else index
        if declaration <= 0:
            declaration = index
        start = self._declaration_start(declaration)
        previous_end = self.tokens[start - 1].end if start > 0 else 0
        found: list[str] = []
        mentions_record = False
        for comment in self.comments:
            if previous_end <= comment.start < self.tokens[declaration].start:
                if "3.0.2" in comment.text or "same record" in comment.text or "the record" in comment.text:
                    mentions_record = True
                found.extend(re.findall(r"https://[^\s>)]+", comment.text))
        found = [url.rstrip(".,;") for url in found]
        if found:
            return found[-1]
        if mentions_record:
            for marker, url in family_citations.items():
                if marker in self.relative:
                    return url
        return None


def collect_files(root: Path) -> list[str]:
    files: list[str] = []
    for pattern in TARGET_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            parts = set(relative.split("/"))
            if parts & SKIP_DIR_NAMES or path.name in SKIP_FILE_NAMES:
                continue
            files.append(relative)
    return sorted(set(files))


def crate_citations(root: Path) -> dict[str, str]:
    citations: dict[str, str] = {
        "gf8": GF8_RECORD,
        "gf16": GF16_RECORD,
        "residue": RESIDUE_RECORD,
    }
    agents = root / "AGENTS.md"
    if not agents.is_file():
        return citations
    text = agents.read_text(encoding="utf-8", errors="replace")
    for url in re.findall(r"https://[^\s>)]+", text):
        url = url.rstrip(".,;")
        if "v3.0.2" not in url:
            continue
        if "gf8.md" in url:
            citations.setdefault("gf8", url)
        elif "gf16.md" in url:
            citations.setdefault("gf16", url)
        elif "residue.md" in url:
            citations.setdefault("residue", url)
    return citations
def add_const_item(items: dict[str, dict], item: dict) -> None:
    base = item["key"]
    key = base
    suffix = 2
    while key in items:
        key = f"{base}~{suffix}"
        suffix += 1
    item["key"] = key
    items[key] = item


def add_item(items: dict[str, dict], item: dict) -> None:
    """Add one expression item; identical file+expression keys collapse."""

    if item["key"] not in items:
        items[item["key"]] = item


def make_item(
    relative: str,
    key: str,
    name_or_expr: str,
    value: object,
    unit: str,
    axis: str,
    item_class: str,
    routes: str,
    citation: str | None,
) -> dict:
    return {
        "key": key,
        "file": relative,
        "name_or_expr": name_or_expr,
        "value": value,
        "unit": unit,
        "axis": axis,
        "class": item_class,
        "routes": routes,
        "citation": citation,
    }


def mark_position(
    item: dict,
    model: "SourceModel",
    index: int,
    *,
    condition_start: int | None = None,
    condition_end: int | None = None,
    const_name: str | None = None,
    operator: str | None = None,
    threshold_side: str | None = None,
) -> None:
    """Attach an in-memory source position used only by ``--positions``.

    ``scan`` removes the marker before classification and serialization, so
    ``inventory.json`` is byte-identical whether or not positions are
    requested.
    """

    tokens = model.tokens
    item["_position"] = {
        "file": model.relative,
        "offset": tokens[index].start,
        "condition_start": tokens[condition_start].start if condition_start is not None else None,
        "condition_end": tokens[condition_end].end if condition_end is not None else None,
        "const_name": const_name,
        "operator": operator,
        "threshold_side": threshold_side,
    }


def extract_consts(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    raw: list[dict] = []
    i = 0
    while i < len(tokens):
        if tokens[i].text != "const":
            i += 1
            continue
        if i > 0 and tokens[i - 1].text in {"<", ","}:
            i += 1
            continue
        name_index = i + 1
        if name_index >= len(tokens):
            i += 1
            continue
        if tokens[name_index].text in {"fn", "_"} or tokens[name_index].kind != "ident":
            i += 1
            continue
        if name_index + 1 >= len(tokens) or tokens[name_index + 1].text != ":":
            i += 1
            continue
        depth = 0
        equals_index = None
        j = name_index + 2
        while j < len(tokens):
            text = tokens[j].text
            if text in {"(", "[", "{", "<"}:
                depth += 1
            elif text in {")", "]", "}", ">"}:
                depth = max(0, depth - 1)
            elif text == ">>":
                depth = max(0, depth - 2)
            elif text == "=" and depth == 0:
                equals_index = j
                break
            elif text == ";" and depth == 0:
                break
            j += 1
        if equals_index is None:
            i += 1
            continue
        depth = 0
        semicolon = None
        j = equals_index + 1
        while j < len(tokens):
            text = tokens[j].text
            if text in {"(", "[", "{"}:
                depth += 1
            elif text in {")", "]", "}"}:
                depth = max(0, depth - 1)
            elif text == ";" and depth == 0:
                semicolon = j
                break
            j += 1
        if semicolon is None:
            i += 1
            continue
        value_tokens = tokens[equals_index + 1 : semicolon]
        if any(token.text == "$" for token in value_tokens):
            i = semicolon + 1
            continue
        type_text = re.sub(r"\s+", "", source[tokens[name_index + 2].start : tokens[equals_index].start])
        type_text = re.sub(r"^::", "", type_text)
        if type_text not in INTEGER_TYPES and type_text not in MASK_TYPES:
            i = semicolon + 1
            continue
        raw.append(
            {
                "name": tokens[name_index].text,
                "type": type_text,
                "value_tokens": value_tokens,
                "value_start": equals_index + 1,
                "value_end": semicolon - 1,
                "start": tokens[i].start,
                "scope": model.scope_key(i),
                "index": i,
            }
        )
        i = semicolon + 1

    known: dict[str, int] = {}
    for entry in raw:
        value = eval_numeric(entry["value_tokens"], known)
        if value is not None:
            known.setdefault(entry["name"], value)
    changed = True
    while changed:
        changed = False
        for entry in raw:
            value = eval_numeric(entry["value_tokens"], known)
            if value is not None and entry["name"] not in known:
                known[entry["name"]] = value
                changed = True

    for entry in raw:
        value = eval_numeric(entry["value_tokens"], known)
        if value is None:
            value = compact_expr(normalize_expr(source, tokens, entry["value_start"], entry["value_end"]))
        scope = entry["scope"]
        key = f"{model.relative}::{scope + '::' if scope else ''}{entry['name']}"
        item = make_item(
            model.relative,
            key,
            entry["name"],
            value,
            "other",
            "",
            STRUCTURAL,
            "",
            model.citation(entry["index"], family_citations, use_scope=False),
        )
        item["type_text"] = entry["type"]
        mark_position(item, model, entry["index"], const_name=entry["name"])
        add_const_item(items, item)


def pure_number(tokens: list[Token], start: int, end: int) -> tuple[int, int] | None:
    """Return ``(index, value)`` when a token range is exactly one integer."""

    while start <= end and tokens[start].text in {"(", ")"}:
        start += 1
    while end >= start and tokens[end].text in {"(", ")"}:
        end -= 1
    if start == end and tokens[start].kind == "number":
        value = parse_number(tokens[start].text)
        if value is not None:
            return start, value
    if end == start + 1 and tokens[start].text in {"-", "+"} and tokens[end].kind == "number":
        value = parse_number(tokens[end].text)
        if value is not None:
            return end, -value if tokens[start].text == "-" else value
    return None


def operand_left(tokens: list[Token], operator: int) -> int:
    j = operator - 1
    if j < 0:
        return 0
    if tokens[j].text in {")", "]"}:
        j = match_delim_backward(tokens, j)
        j -= 1
    while j >= 1 and tokens[j - 1].text == "::":
        j -= 1
        if j >= 1 and tokens[j - 1].kind == "ident":
            j -= 1
    while j >= 1 and tokens[j - 1].text == ".":
        j -= 1
        if j >= 1 and tokens[j - 1].kind == "ident":
            j -= 1
        if j >= 1 and tokens[j - 1].text in {")", "]"}:
            j = match_delim_backward(tokens, j - 1)
    return max(0, j)


def operand_right(tokens: list[Token], operator: int, limit: int) -> int:
    j = operator + 1
    if j >= limit:
        return limit - 1
    if tokens[j].text in {"-", "+"} and j + 1 < limit and tokens[j + 1].kind == "number":
        j += 1
    if tokens[j].text in {"(", "["}:
        j = match_delim(tokens, j)
    return min(j, limit - 1)


def extract_comparisons(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    for i, token in enumerate(tokens):
        if token.text not in COMPARISONS:
            continue
        if i in model.attribute_tokens or i in model.generic_tokens:
            continue
        left_start = operand_left(tokens, i)
        right_end = operand_right(tokens, i, len(tokens))
        left_number = pure_number(tokens, left_start, i - 1)
        right_number = pure_number(tokens, i + 1, right_end)
        arithmetic = {"%", "*", "/", "<<", ">>", "&", "|", "^"}
        candidates = []
        if left_number is not None:
            if left_start == 0 or tokens[left_start - 1].text not in arithmetic:
                candidates.append(("left", left_number[0], left_number[1]))
        if right_number is not None and right_number[0] != (left_number[0] if left_number else -1):
            if right_end + 1 >= len(tokens) or tokens[right_end + 1].text not in arithmetic:
                candidates.append(("right", right_number[0], right_number[1]))
        for side, number_index, value in candidates:
            expression = normalize_expr(source, tokens, left_start, right_end)
            scope = model.scope_key(i)
            key = f"{model.relative}::{scope + '::' if scope else ''}cmp:{compact_expr(expression)}:{side}"
            item = make_item(
                model.relative,
                key,
                expression,
                value,
                "other",
                "",
                STRUCTURAL,
                "",
                model.citation(number_index, family_citations),
            )
            mark_position(
                item,
                model,
                i,
                condition_start=left_start,
                condition_end=right_end,
                operator=token.text,
                threshold_side=side,
            )
            comparison = cutoff_comparison(token.text, side)
            if comparison is not None:
                item["comparison"] = comparison
            add_item(items, item)


def extract_matches(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    for i, token in enumerate(tokens):
        if token.text != "match" or i in model.attribute_tokens:
            continue
        brace = None
        j = i + 1
        while j < len(tokens) and tokens[j].text not in {"{", ";"}:
            j += 1
        if j >= len(tokens) or tokens[j].text != "{":
            continue
        brace = j
        close = match_delim(tokens, brace)
        scrutinee = normalize_expr(source, tokens, i + 1, brace - 1) if brace > i + 1 else ""
        pos = brace + 1
        while pos < close:
            depth = 0
            arrow = None
            p = pos
            while p < close:
                text = tokens[p].text
                if text in {"(", "[", "{"}:
                    depth += 1
                elif text in {")", "]", "}"}:
                    depth = max(0, depth - 1)
                elif text == "=>" and depth == 0:
                    arrow = p
                    break
                p += 1
            if arrow is None:
                break
            depth = 0
            comma = close
            q = arrow + 1
            while q < close:
                text = tokens[q].text
                if text in {"(", "[", "{"}:
                    depth += 1
                elif text in {")", "]", "}"}:
                    depth = max(0, depth - 1)
                elif text == "," and depth == 0:
                    comma = q
                    break
                q += 1
            pattern_end = arrow - 1
            value_end = comma - 1
            pattern = normalize_expr(source, tokens, pos, pattern_end) if pattern_end >= pos else ""
            value = normalize_expr(source, tokens, arrow + 1, value_end) if value_end > arrow else ""
            if "$" not in pattern and "$" not in value:
                pattern_numbers = [
                    index for index in range(pos, arrow) if tokens[index].kind == "number"
                ]
                value_number = pure_number(tokens, arrow + 1, value_end) if value_end > arrow else None
                for number_index in pattern_numbers:
                    literal = parse_number(tokens[number_index].text)
                    if literal is None:
                        continue
                    expression = f"match {scrutinee} {{ {pattern} => {value} }}"
                    scope = model.scope_key(i)
                    key = (
                        f"{model.relative}::{scope + '::' if scope else ''}"
                        f"match:{compact_expr(scrutinee)}:{compact_expr(pattern)}=>{compact_expr(value)[:80]}:pattern"
                    )
                    item = make_item(
                        model.relative,
                        key,
                        expression,
                        literal,
                        "other",
                        "",
                        STRUCTURAL,
                        "",
                        model.citation(number_index, family_citations),
                    )
                    mark_position(item, model, number_index)
                    add_item(items, item)
                if value_number is not None:
                    expression = f"match {scrutinee} {{ {pattern} => {value} }}"
                    scope = model.scope_key(i)
                    key = (
                        f"{model.relative}::{scope + '::' if scope else ''}"
                        f"match:{compact_expr(scrutinee)}:{compact_expr(pattern)}=>{compact_expr(value)[:80]}:value"
                    )
                    item = make_item(
                        model.relative,
                        key,
                        expression,
                        value_number[1],
                        "other",
                        "",
                        STRUCTURAL,
                        "",
                        model.citation(value_number[0], family_citations),
                    )
                    mark_position(item, model, value_number[0])
                    add_item(items, item)
            pos = comma + 1


def extract_returns(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    for i, token in enumerate(tokens):
        if token.text != "return":
            continue
        end = i + 1
        depth = 0
        while end < len(tokens):
            text = tokens[end].text
            if text in {"(", "[", "{"}:
                depth += 1
            elif text in {")", "]", "}"}:
                if depth == 0:
                    break
                depth -= 1
            elif text == ";" and depth == 0:
                break
            end += 1
        if end <= i + 1:
            continue
        number = pure_number(tokens, i + 1, end - 1)
        if number is None:
            continue
        expression = normalize_expr(source, tokens, i, end - 1)
        scope = model.scope_key(i)
        key = f"{model.relative}::{scope + '::' if scope else ''}return:{compact_expr(expression)}"
        item = make_item(
            model.relative,
            key,
            expression,
            number[1],
            "other",
            "",
            STRUCTURAL,
            "",
            model.citation(number[0], family_citations),
        )
        mark_position(item, model, number[0])
        add_item(items, item)


def extract_tail_literals(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    for i, token in enumerate(tokens):
        if token.kind != "number" or i + 1 >= len(tokens) or tokens[i + 1].text != "}":
            continue
        previous = tokens[i - 1].text if i > 0 else ""
        if previous not in {"{", ";", "}", "else", "=>"}:
            continue
        if not any(label.startswith("fn ") for label in model.scopes[i]):
            continue
        value = parse_number(token.text)
        if value is None:
            continue
        expression = normalize_expr(source, tokens, i, i)
        scope = model.scope_key(i)
        key = f"{model.relative}::{scope + '::' if scope else ''}tail:{compact_expr(expression)}"
        item = make_item(
            model.relative,
            key,
            expression,
            value,
            "other",
            "",
            STRUCTURAL,
            "",
            model.citation(i, family_citations),
        )
        mark_position(item, model, i)
        add_item(items, item)


def extract_let_bindings(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    for i, token in enumerate(tokens):
        if token.text != "let":
            continue
        depth = 0
        equals = None
        j = i + 1
        while j < len(tokens):
            text = tokens[j].text
            if text in {"(", "[", "{"}:
                depth += 1
            elif text in {")", "]", "}"}:
                depth = max(0, depth - 1)
            elif text == "=" and depth == 0:
                equals = j
                break
            elif text == ";" and depth == 0:
                break
            j += 1
        if equals is None:
            continue
        depth = 0
        semicolon = None
        j = equals + 1
        while j < len(tokens):
            text = tokens[j].text
            if text in {"(", "[", "{"}:
                depth += 1
            elif text in {")", "]", "}"}:
                depth = max(0, depth - 1)
            elif text == ";" and depth == 0:
                semicolon = j
                break
            j += 1
        if semicolon is None:
            continue
        names = [
            tokens[k].text
            for k in range(i + 1, equals)
            if tokens[k].kind == "ident" and tokens[k].text not in {"mut", "ref"}
        ]
        if not names:
            continue
        name = names[-1]
        init_range = (equals + 1, semicolon - 1)
        if init_range[0] > init_range[1]:
            continue
        compared = False
        base_scope = model.scope_key(i)
        for k in range(semicolon + 1, min(len(tokens), semicolon + 600)):
            if model.scope_key(k) != base_scope:
                break
            if tokens[k].text == name and k > 0 and k + 1 < len(tokens):
                if tokens[k - 1].text in COMPARISONS or tokens[k + 1].text in COMPARISONS:
                    compared = True
                    break
        if not compared:
            continue
        number = pure_number(tokens, init_range[0], init_range[1])
        if number is None:
            continue
        initializer = normalize_expr(source, tokens, init_range[0], init_range[1])
        number_index, value = number
        scope = base_scope
        key = (
            f"{model.relative}::{scope + '::' if scope else ''}"
            f"let {name}:{compact_expr(initializer)}"
        )
        item = make_item(
            model.relative,
            key,
            f"let {name} = {initializer}",
            value,
            "other",
            "",
            STRUCTURAL,
            "",
            model.citation(number_index, family_citations),
        )
        mark_position(item, model, number_index, const_name=name)
        add_item(items, item)


def generic_close(tokens: list[Token], open_index: int) -> int | None:
    depth = 1
    j = open_index + 1
    while j < len(tokens):
        text = tokens[j].text
        if text == "<":
            depth += 1
        elif text == ">":
            depth -= 1
        elif text == ">>":
            depth = max(0, depth - 2)
        if depth == 0:
            return j
        j += 1
    return None


def extract_generic_arguments(model: SourceModel, items: dict[str, dict], family_citations: dict[str, str]) -> None:
    tokens = model.tokens
    source = model.source
    for i, token in enumerate(tokens):
        if token.text != "::" or i + 1 >= len(tokens) or tokens[i + 1].text != "<":
            continue
        if i in model.attribute_tokens:
            continue
        close = generic_close(tokens, i + 1)
        if close is None:
            continue
        callee_index = i - 1
        if callee_index < 0 or tokens[callee_index].kind != "ident":
            continue
        callee = tokens[callee_index].text
        arguments: list[tuple[int, int]] = []
        depth = 0
        start = i + 2
        j = i + 2
        while j < close:
            text = tokens[j].text
            if text in {"<", "(", "["}:
                depth += 1
            elif text in {">", ")", "]"}:
                depth = max(0, depth - 1)
            elif text == ">>":
                depth = max(0, depth - 2)
            elif text == "," and depth == 0:
                arguments.append((start, j - 1))
                start = j + 1
            j += 1
        if start < close:
            arguments.append((start, close - 1))
        arguments_text = normalize_expr(source, tokens, i + 2, close - 1) if close > i + 2 else ""
        for argument_index, (start, end) in enumerate(arguments):
            number = pure_number(tokens, start, end)
            if number is None:
                continue
            scope = model.scope_key(i)
            key = (
                f"{model.relative}::{scope + '::' if scope else ''}"
                f"generic:{callee}::<{compact_expr(arguments_text)}>:{argument_index}"
            )
            item = make_item(
                model.relative,
                key,
                f"{callee}::<{arguments_text}>",
                number[1],
                "other",
                "",
                STRUCTURAL,
                "",
                model.citation(number[0], family_citations),
            )
            mark_position(item, model, number[0])
            add_item(items, item)
# Reviewed overrides, keyed by the exact item key.  Each entry names the
# class, unit, axis, routes, and optionally a citation anchor.  Items not
# listed here fall back to the structural default; the scan itself only
# discovers and keys items.
GF8_RECORD = "https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf8.md"
GF16_RECORD = "https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf16.md"
RESIDUE_RECORD = "https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md"


def _override(item_class: str, unit: str, axis: str, routes: str, citation: str | None = None) -> dict:
    entry = {"class": item_class, "unit": unit, "axis": axis, "routes": routes}
    if citation is not None:
        entry["citation"] = citation
    return entry


OVERRIDES: dict[str, dict] = {
    # Storage layout and representation constants.
    "src/field/extension/storage.rs::impl StorageBuffer for Components::BYTES": _override(
        STRUCTURAL, "bytes", "element width", ""
    ),
    "src/field/extension/storage.rs::impl StorageBuffer for Components::STORAGE_BITS": _override(
        STRUCTURAL, "other", "element bit width", ""
    ),
    "src/kernel/tables.rs::IDENTITY_MAP": _override(STRUCTURAL, "other", "representation map", ""),
    "src/kernel/tables.rs::RIJNDAEL16_B": _override(STRUCTURAL, "other", "representation map", ""),
    "src/kernel/tables.rs::impl ByteReprData for Cantor::AES_MAP_FORWARD": _override(
        STRUCTURAL, "other", "representation map", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Cantor::AES_MAP_INVERSE": _override(
        STRUCTURAL, "other", "representation map", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Cantor::REDUCTION_LOW": _override(
        STRUCTURAL, "other", "reduction constant", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Normal::AES_MAP_FORWARD": _override(
        STRUCTURAL, "other", "representation map", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Normal::AES_MAP_INVERSE": _override(
        STRUCTURAL, "other", "representation map", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Normal::REDUCTION_LOW": _override(
        STRUCTURAL, "other", "reduction constant", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Polynomial::AES_MAP_FORWARD": _override(
        STRUCTURAL, "other", "representation map", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Polynomial::AES_MAP_INVERSE": _override(
        STRUCTURAL, "other", "representation map", ""
    ),
    "src/kernel/tables.rs::impl ByteReprData for Polynomial::REDUCTION_LOW": _override(
        STRUCTURAL, "other", "reduction constant", ""
    ),
    "src/kernel/x86/goldilocks.rs::P": _override(STRUCTURAL, "other", "modulus", ""),
    "src/kernel/x86/goldilocks.rs::PM1": _override(STRUCTURAL, "other", "modulus minus one", ""),
    "src/kernel/x86/goldilocks.rs::EPS": _override(STRUCTURAL, "other", "modulus mask", ""),
    "src/kernel/x86/goldilocks/avx2.rs::SIGN": _override(STRUCTURAL, "other", "lane sign bit", ""),
    "src/kernel/x86/goldilocks/avx2.rs::SHIFTED_P": _override(
        STRUCTURAL, "other", "modulus offset", ""
    ),
    "src/kernel/x86/goldilocks/avx512.rs::ODD_LANES": _override(
        STRUCTURAL, "lanes", "lane mask", ""
    ),
    "src/kernel/x86/goldilocks/sse42.rs::M32": _override(STRUCTURAL, "other", "modulus mask", ""),
    "src/kernel/x86/mersenne31.rs::P": _override(STRUCTURAL, "other", "modulus", ""),
    "src/kernel/x86/quad_mersenne31/avx512.rs::ODD_LANES": _override(
        STRUCTURAL, "lanes", "lane mask", ""
    ),
    "src/ops/bits/count.rs::fn weight::SEGMENT_BITS": _override(
        STRUCTURAL, "count", "segment length", ""
    ),
    "src/ops/recip.rs::SCRATCH_BYTES": _override(STRUCTURAL, "bytes", "scratch size", ""),
    # Routing thresholds.
    "src/kernel/routing.rs::XOR_INLINE_MAX": _override(
        ROUTING, "bytes", "buffer length", "inline scalar XOR <-> dispatched SIMD XOR"
    ),
    "src/kernel/routing.rs::QUAD_VECTOR_ADD_MIN_BYTES": _override(
        ROUTING, "bytes", "destination length", "scalar add <-> AVX2/AVX-512 add"
    ),
    "src/kernel/routing.rs::QUAD_VECTOR_MUL_MIN_BYTES": _override(
        ROUTING, "bytes", "destination length", "scalar multiply <-> AVX2/AVX-512 multiply"
    ),
    "src/kernel/routing.rs::DELAYED_MIN_SOURCES": _override(
        ROUTING, "count", "source count", "repeated single-row composition <-> delayed tile"
    ),
    "src/kernel/routing.rs::DELAYED_INTO_MIN_DEPTH": _override(
        ROUTING, "count", "delay depth", "repeated single-row overwrite <-> delayed tile"
    ),
    "src/kernel/routing.rs::DELAYED_ADD_MIN_DEPTH": _override(
        ROUTING, "count", "delay depth", "repeated single-row accumulation <-> delayed tile"
    ),
    "src/kernel/routing.rs::NT_STORE_MIN": _override(
        ROUTING, "bytes", "destination length", "temporal stores <-> streaming stores"
    ),
    "src/kernel/routing.rs::HALF_LANE_PEEL_MIN": _override(
        ROUTING,
        "bytes",
        "destination length",
        "unaligned half-lane head <-> peeled aligned body",
        GF8_RECORD,
    ),
    "src/kernel/routing.rs::ALIGN_PEEL_MIN": _override(
        ROUTING,
        "bytes",
        "destination length",
        "unpeeled body <-> peeled aligned body",
        GF8_RECORD,
    ),
    "src/kernel/routing.rs::XOR_AVX512_MIN": _override(
        ROUTING, "bytes", "buffer length", "AVX2 XOR <-> AVX-512 XOR"
    ),
    "src/kernel/routing.rs::XOR_PEEL_MIN": _override(
        ROUTING, "bytes", "destination length", "unpeeled XOR <-> peeled XOR"
    ),
    "src/kernel/routing.rs::MUL_ADD_PEEL_MIN": _override(
        ROUTING,
        "bytes",
        "destination length",
        "unpeeled GFNI body <-> peeled AVX-512 body",
        GF16_RECORD,
    ),
    "src/kernel/routing.rs::GATHER_PEEL_MIN": _override(
        ROUTING, "bytes", "destination length", "unpeeled gather <-> peeled AVX-512 gather"
    ),
    "src/kernel/routing.rs::MATRIX_PEEL_MIN": _override(
        ROUTING, "bytes", "row length", "unpeeled matrix rows <-> peeled AVX-512 matrix rows"
    ),
    "src/kernel/routing.rs::MATRIX_RESOLVE_MIN": _override(
        ROUTING,
        "bytes",
        "row length",
        "direct coefficient reads <-> chunk-resolved coefficients",
        GF8_RECORD,
    ),
    "src/kernel/routing.rs::SCATTER_PEEL_MIN": _override(
        ROUTING, "bytes", "row length", "unpeeled scatter <-> peeled AVX-512 scatter"
    ),
    # Kernel tuning parameters and schedules.
    "src/kernel/x86.rs::PREFETCH_MIN": _override(
        TUNING, "bytes", "destination length", "prefetch branch"
    ),
    "src/kernel/x86.rs::PREFETCH_AHEAD": _override(
        TUNING, "bytes", "prefetch distance", "prefetch distance"
    ),
    "src/kernel/x86/gf16.rs::TERM_TILE": _override(
        TUNING, "terms", "terms per resolved tile", "gather and matrix term tile"
    ),
    "src/kernel/x86/gf16/avx512.rs::fn matrix_group::TERM_BLOCK": _override(
        TUNING, "terms", "terms per destination block", "matrix term block"
    ),
    "src/kernel/x86/gf16/gfni/gather.rs::SOURCE_GROUP": _override(
        TUNING, "count", "sources per register group", "source register group"
    ),
    "src/kernel/x86/gf16/gfni/matrix.rs::TERM_BLOCK": _override(
        TUNING, "terms", "terms per destination block", "matrix term block"
    ),
    "src/kernel/x86/gf8.rs::RESOLVE_CHUNK": _override(
        TUNING, "terms", "terms per resolve chunk", "coefficient resolve chunk"
    ),
    "src/kernel/x86/gf8.rs::MATRIX_COLUMN_BLOCK": _override(
        TUNING, "bytes", "column block", "whole-row <-> column-blocked matrix walk"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::FEW_TERMS": _override(
        TUNING, "terms", "term count", "four-row <-> six-row row group"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::MAX_GROUP_ROWS": _override(
        TUNING, "rows", "row group width", "row group width"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::THREE_LANE_MIN_TERMS": _override(
        TUNING, "terms", "term count", "two-lane <-> three-lane row group"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::THREE_LANE_MIN_ROW": _override(
        TUNING, "bytes", "row length", "two-lane <-> three-lane row group"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::THREE_LANE_MAX_ROW": _override(
        TUNING, "bytes", "row length", "three-lane <-> two-lane row group"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::MATRIX_PREFETCH_DISTANCE": _override(
        TUNING, "bytes", "prefetch distance", "prefetch distance"
    ),
    "src/kernel/x86/gf8/gfni/rows.rs::MATRIX_PREFETCH_MIN": _override(
        TUNING, "bytes", "row length", "prefetch branch"
    ),
    "src/kernel/x86/mersenne31/avx512.rs::SCREEN_LANES": _override(
        TUNING, "lanes", "lanes per screen group", "register screening group"
    ),
    "src/kernel/x86/quad_mersenne31/avx512.rs::SCREEN_LANES": _override(
        TUNING, "lanes", "lanes per screen group", "register screening group"
    ),
    "src/kernel/x86/residue/avx2/matrix.rs::TILE": _override(
        TUNING, "rows", "outputs per delayed tile", "delayed tile width"
    ),
    "src/kernel/x86/residue/avx512/matrix.rs::TILE": _override(
        TUNING, "rows", "outputs per delayed tile", "delayed tile width"
    ),
    "src/ops/recip.rs::CHUNK_ELEMS": _override(
        TUNING, "elements", "elements per batch chunk", "batch inversion chunk"
    ),
    # Routing thresholds written as literals.
    "src/kernel/routing.rs::GF8_NIBBLE_MUL_ASSIGN_MIN": _override(
        ROUTING, "bytes", "destination length", "GFNI affine multiply <-> AVX2 nibble multiply"
    ),
    "src/kernel/routing.rs::fn routes_vector::cmp:p<=127:right": _override(
        ROUTING, "other", "modulus", "scalar <-> vector u8 rows"
    ),
    # The u8 kernels keep two nibble-table values below P whose sum must fit in
    # a byte (2P <= 254); `routes_vector` already routes P > 127 to scalar, so
    # these local guards are correctness bounds, not crossovers.
    "src/kernel/residue.rs::fn mul_into_prepared::cmp:P<=127:right": _override(
        STRUCTURAL, "other", "u8 lane overflow bound", ""
    ),
    "src/kernel/residue.rs::fn mul_add_prepared::cmp:P<=127:right": _override(
        STRUCTURAL, "other", "u8 lane overflow bound", ""
    ),
    "src/kernel/residue.rs::fn mul_assign_prepared::cmp:P<=127:right": _override(
        STRUCTURAL, "other", "u8 lane overflow bound", ""
    ),
    "src/kernel/routing.rs::DELAYED_TILE_ROWS": _override(
        ROUTING, "rows", "output row count", "repeated single-row composition <-> delayed tile"
    ),
    "src/kernel/routing.rs::DELAYED_MIN_ROW_BYTES": _override(
        ROUTING, "bytes", "row length", "repeated single-row composition <-> delayed tile"
    ),
    "src/kernel/routing.rs::fn avx512_floor_op_len::match:(bytes,op):(4,VectorOp::Assign)=>1024:value": _override(
        ROUTING, "bytes", "destination length", "AVX2 <-> AVX-512 in-place scale"
    ),
    "src/kernel/routing.rs::fn avx512_floor_op_len::match:(bytes,op):(8,VectorOp::Assign)=>256:value": _override(
        ROUTING, "bytes", "destination length", "AVX2 <-> AVX-512 in-place scale"
    ),
    "src/kernel/routing.rs::fn avx512_floor_op_len::match:(bytes,op):_=>64:value": _override(
        ROUTING, "bytes", "destination length", "AVX2 <-> AVX-512 single-row kernel"
    ),
    "src/kernel/routing.rs::fn vector_floor_into_lanes::match:bytes:1=>64:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 overwrite"
    ),
    "src/kernel/routing.rs::fn vector_floor_into_lanes::match:bytes:8=>8:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 overwrite"
    ),
    "src/kernel/routing.rs::fn vector_floor_into_lanes::match:bytes:_=>8:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 overwrite"
    ),
    "src/kernel/routing.rs::fn vector_floor_assign_lanes::match:bytes:1=>64:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 in-place scale"
    ),
    "src/kernel/routing.rs::fn vector_floor_assign_lanes::match:bytes:8=>16:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 in-place scale"
    ),
    "src/kernel/routing.rs::fn vector_floor_assign_lanes::match:bytes:_=>8:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 in-place scale"
    ),
    "src/kernel/routing.rs::fn vector_floor_add_lanes::match:bytes:1=>8:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 accumulation"
    ),
    "src/kernel/routing.rs::fn vector_floor_add_lanes::match:bytes:8=>32:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 accumulation"
    ),
    "src/kernel/routing.rs::fn vector_floor_add_lanes::match:bytes:_=>8:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 accumulation"
    ),
    "src/kernel/routing.rs::fn vector_floor_elementwise_lanes::match:op:VectorOp::Elementwise=>16:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 elementwise multiply"
    ),
    "src/kernel/routing.rs::fn vector_floor_elementwise_lanes::match:op:VectorOp::ElementwiseAssign=>8:value": _override(
        ROUTING, "lanes", "row length in lanes", "scalar <-> AVX2 elementwise multiply"
    ),
    # Tuning literals.
    "src/kernel/x86/gf8.rs::fn column_blocked::cmp:nrows==3:right": _override(
        TUNING, "rows", "output row count", "whole-row <-> column-blocked matrix walk"
    ),
    "src/kernel/x86/gf8.rs::fn column_blocked::cmp:nrows>4:right": _override(
        TUNING, "rows", "output row count", "whole-row <-> column-blocked matrix walk"
    ),
    "src/kernel/x86/gf8/gfni/gather.rs::fn mul_add_gather_gfni::cmp:dst.len()<128:right": _override(
        TUNING, "bytes", "destination length", "fused native tail <-> generic tail"
    ),
    "src/kernel/x86/gf8/gfni/gather.rs::fn mul_add_gather_gfni::cmp:coeffs.count()>2:right": _override(
        TUNING, "count", "source count", "fused native tail <-> generic tail"
    ),
    "src/kernel/x86/gf8/gfni/gather.rs::fn mul_add_gather_gfni::cmp:remainder.trailing_zeros()>=5:right": _override(
        TUNING, "count", "remainder alignment exponent", "fused native tail <-> generic tail"
    ),
    "src/kernel/x86/gf8/avx512/matrix.rs::fn matrix_at_block::cmp:2<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "two-row group <-> one-row group"
    ),
    "src/kernel/x86/gf8/avx512/matrix.rs::fn matrix_at_block::cmp:4<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "four-row group <-> smaller group"
    ),
    "src/kernel/x86/gf8/avx512/matrix.rs::fn matrix_block::cmp:2<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "two-row group <-> one-row group"
    ),
    "src/kernel/x86/gf8/avx512/matrix.rs::fn matrix_block::cmp:4<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "four-row group <-> smaller group"
    ),
    "src/kernel/x86/gf16/avx512.rs::fn mul_add_matrix_avx512_with::cmp:2<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "two-row group <-> one-row group"
    ),
    "src/kernel/x86/gf16/avx512.rs::fn mul_add_matrix_avx512_with::cmp:4<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "four-row group <-> smaller group"
    ),
    "src/kernel/x86/gf16/gfni/matrix.rs::fn mul_add_matrix_gfni_with::cmp:2<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "two-row group <-> one-row group"
    ),
    "src/kernel/x86/gf16/gfni/matrix.rs::fn mul_add_matrix_gfni_with::cmp:4<=nrows:left": _override(
        TUNING, "rows", "remaining row count", "four-row group <-> smaller group"
    ),
    "src/kernel/x86/gf8/avx512/scatter.rs::fn mul_add_scatter_impl::cmp:filled==1:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/avx512/scatter.rs::fn mul_add_scatter_impl::cmp:filled==3:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/avx512/scatter.rs::fn mul_add_scatter_impl::cmp:filled==4:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/avx512/scatter.rs::fn mul_add_scatter_impl::cmp:filled>=2:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/gfni/scatter.rs::fn mul_add_scatter_impl::cmp:filled==1:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/gfni/scatter.rs::fn mul_add_scatter_impl::cmp:filled==3:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/gfni/scatter.rs::fn mul_add_scatter_impl::cmp:filled==4:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
    "src/kernel/x86/gf8/gfni/scatter.rs::fn mul_add_scatter_impl::cmp:filled>=2:right": _override(
        TUNING, "count", "cohort fill", "partial cohort <-> full cohort"
    ),
}

SCHEDULE_NAMES = {
    "RESOLVE_CHUNK",
    "MATRIX_COLUMN_BLOCK",
    "MATRIX_PREFETCH_DISTANCE",
    "MAX_GROUP_ROWS",
    "FEW_TERMS",
    "THREE_LANE_MIN_TERMS",
    "TERM_TILE",
    "TERM_BLOCK",
    "SOURCE_GROUP",
    "TILE",
    "SCREEN_LANES",
    "CHUNK_ELEMS",
}


def classify(item: dict) -> None:
    key = item["key"]
    override = OVERRIDES.get(key)
    if override is not None:
        item["class"] = override.get("class", STRUCTURAL)
        item["unit"] = override.get("unit", "other")
        item["axis"] = override.get("axis", "")
        item["routes"] = override.get("routes", "")
        if "citation" in override:
            item["citation"] = override["citation"]
        if item["class"] == STRUCTURAL:
            item["citation"] = override.get("citation")
        return

    name = item["name_or_expr"]
    if name.startswith("let "):
        name = name.split("=", 1)[0].strip()
    bare = name.split("::")[-1]
    if bare in SCHEDULE_NAMES:
        item["class"] = TUNING
        item["unit"] = "count"
        item["axis"] = "kernel schedule"
        item["routes"] = "within one route"
    elif "PREFETCH" in bare:
        item["class"] = TUNING
        item["unit"] = "bytes"
        item["axis"] = "destination length"
        item["routes"] = "prefetch branch"
    elif (
        bare.endswith("_MIN")
        or "PEEL_MIN" in bare
        or bare == "FLOOR"
        or "MIN_BYTES" in bare
        or bare.startswith("DELAYED_")
        or bare == "XOR_INLINE_MAX"
        or bare == "NT_STORE_MIN"
    ):
        item["class"] = ROUTING
        item["unit"] = "bytes"
        item["axis"] = "destination length"
        item["routes"] = "scalar <-> vector"
    else:
        item["class"] = STRUCTURAL
        item["unit"] = "other"
        item["axis"] = ""
        item["routes"] = ""
    if item["class"] == STRUCTURAL:
        item["citation"] = None


def line_column(source: str, offset: int) -> tuple[int, int]:
    """One-based line and column of a byte offset in ``source``."""

    return source.count("\n", 0, offset) + 1, offset - source.rfind("\n", 0, offset)


def _in_use_item(tokens: list[Token], index: int) -> bool:
    """Whether the token at ``index`` sits inside a ``use`` declaration."""

    j = index - 1
    while j >= 0 and index - j <= 120:
        text = tokens[j].text
        if text == "use":
            return True
        if text == ";":
            return False
        j -= 1
    return False


def _nearby_comparison(tokens: list[Token], index: int) -> int | None:
    """The comparison operator adjacent to an operand token, if any."""

    for direction in (-1, 1):
        j = index + direction
        if direction == -1:
            while j >= 1 and tokens[j].text == "::" and tokens[j - 1].kind == "ident":
                j -= 2
        steps = 0
        while 0 <= j < len(tokens) and steps < 3:
            text = tokens[j].text
            if text in COMPARISONS:
                return j
            if text in {")", "(", "&", "*"}:
                j += direction
                steps += 1
                continue
            break
    return None


def cutoff_comparison(operator: str, threshold_side: str) -> str | None:
    """Normalize a source comparison to its low-side cutoff predicate."""
    if threshold_side == "left":
        operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(operator, operator)
    if operator in {"<", ">="}:
        return "<"
    if operator in {"<=", ">"}:
        return "<="
    if operator in {"==", "!="}:
        return "=="
    return None


def _const_uses(
    name: str,
    definition_file: str,
    definition_offset: int,
    models: dict[str, "SourceModel"],
    cross_file: bool,
) -> list[dict]:
    """Comparison and plain-use sites for a named constant or binding."""

    uses: list[dict] = []
    definition_line, _ = line_column(models[definition_file].source, definition_offset)
    for relative, model in models.items():
        if not cross_file and relative != definition_file:
            continue
        tokens = model.tokens
        source = model.source
        for index, token in enumerate(tokens):
            if token.text != name or token.kind != "ident":
                continue
            if index in model.attribute_tokens or index in model.generic_tokens:
                continue
            if relative == definition_file and token.start == definition_offset:
                continue
            line, column = line_column(source, token.start)
            if (
                relative == definition_file
                and line == definition_line
                and index + 1 < len(tokens)
                and tokens[index + 1].text in {":", "="}
            ):
                continue
            if _in_use_item(tokens, index):
                continue
            operator = _nearby_comparison(tokens, index)
            if operator is None:
                uses.append(
                    {
                        "file": relative,
                        "line": line,
                        "column": column,
                        "kind": "use",
                    }
                )
                continue
            left_start = operand_left(tokens, operator)
            right_end = operand_right(tokens, operator, len(tokens))
            path_start = index
            while (
                path_start >= 2 and tokens[path_start - 1].text == "::"
                and tokens[path_start - 2].kind == "ident"
            ):
                path_start -= 2
            expr_start, expr_end = path_start, index
            while (
                expr_start > 0
                and expr_end + 1 < len(tokens)
                and tokens[expr_start - 1].text == "("
                and tokens[expr_end + 1].text == ")"
            ):
                expr_start -= 1
                expr_end += 1
            compound_left = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", "!", "~", "."}
            compound_right = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^", ".", "as", "(", "["}
            if index < operator:
                compound = (
                    expr_end + 1 != operator
                    or (expr_start > 0 and tokens[expr_start - 1].text in compound_left)
                )
            else:
                compound = (
                    expr_start != operator + 1
                    or (expr_end + 1 < len(tokens) and tokens[expr_end + 1].text in compound_right)
                )
            condition_line, condition_column = line_column(source, tokens[left_start].start)
            end_line, end_column = line_column(source, tokens[right_end].end)
            uses.append(
                {
                    "file": relative,
                    "line": line,
                    "column": column,
                    "kind": "compare",
                    "operator": tokens[operator].text,
                    "threshold_side": "left" if index < operator else "right",
                    "condition_line": condition_line,
                    "condition_column": condition_column,
                    "condition_end_line": end_line,
                    "condition_end_column": end_column,
                    **({"cutoff_is_direct": False} if compound else {}),
                }
            )
    unique: dict[tuple, dict] = {}
    for use in uses:
        unique.setdefault(
            (use["file"], use["line"], use["column"], use["kind"]), use
        )
    return sorted(unique.values(), key=lambda use: (use["file"], use["line"], use["column"], use["kind"]))


def scan_with_positions(root: Path) -> tuple[list[dict], dict[str, dict]]:
    """Scan the crate and return the inventory plus per-item positions.

    Positions are keyed by inventory key, so they survive ``classify`` and
    sorting without touching the serialized inventory shape.
    """

    root = root.resolve()
    family_citations = crate_citations(root)
    items: dict[str, dict] = {}
    models: dict[str, SourceModel] = {}
    for relative in collect_files(root):
        path = root / relative
        source = path.read_text(encoding="utf-8")
        model = SourceModel(relative, source)
        models[relative] = model
        extract_consts(model, items, family_citations)
        extract_comparisons(model, items, family_citations)
        extract_matches(model, items, family_citations)
        extract_returns(model, items, family_citations)
        extract_tail_literals(model, items, family_citations)
        extract_let_bindings(model, items, family_citations)
        extract_generic_arguments(model, items, family_citations)

    const_files: dict[str, set[str]] = {}
    for item in items.values():
        marker = item.get("_position")
        if marker and marker.get("const_name"):
            const_files.setdefault(marker["const_name"], set()).add(marker["file"])

    result = list(items.values())
    positions: dict[str, dict] = {}
    for item in result:
        marker = item.pop("_position", None)
        item.pop("type_text", None)
        classify(item)
        if item["unit"] not in UNITS:
            raise ValueError(f"{item['key']}: invalid unit {item['unit']!r}")
        if marker is None or item["class"] == STRUCTURAL:
            continue
        source = models[marker["file"]].source
        line, column = line_column(source, marker["offset"])
        payload: dict = {
            "file": marker["file"],
            "line": line,
            "column": column,
        }
        comparisons = set()
        if marker.get("operator"):
            comparisons.add(cutoff_comparison(
                marker["operator"], marker["threshold_side"]
            ))
        if marker.get("condition_start") is not None:
            condition_line, condition_column = line_column(source, marker["condition_start"])
            end_line, end_column = line_column(source, marker["condition_end"])
            payload["condition"] = {
                "line": condition_line,
                "column": condition_column,
                "end_line": end_line,
                "end_column": end_column,
            }
        const_name = marker.get("const_name")
        if const_name:
            payload["uses"] = _const_uses(
                const_name,
                marker["file"],
                marker["offset"],
                models,
                cross_file=len(const_files.get(const_name, ())) == 1,
            )
            comparisons.update(
                cutoff_comparison(use["operator"], use["threshold_side"])
                if use.get("cutoff_is_direct", True) else None
                for use in payload["uses"] if use["kind"] == "compare"
            )
        if len(comparisons) == 1 and None not in comparisons:
            item["comparison"] = comparisons.pop()
        positions[item["key"]] = payload
    result.sort(key=lambda item: (item["file"], item["key"]))
    return result, positions


def scan(root: Path) -> list[dict]:
    items, _ = scan_with_positions(root)
    return items


def citation_status(item: dict) -> str:
    if item["class"] == STRUCTURAL:
        return "structural-exempt"
    return "cited" if item["citation"] else "uncited"


def render_markdown(items: list[dict], root: Path) -> str:
    counts = {ROUTING: 0, TUNING: 0, STRUCTURAL: 0}
    statuses = {"cited": 0, "uncited": 0, "structural-exempt": 0}
    for item in items:
        counts[item["class"]] += 1
        statuses[citation_status(item)] += 1
    lines = [
        "# Integer threshold inventory",
        "",
        "Source scan of `src/kernel/**`, `src/field/dispatch.rs`,",
        "`src/field/extension/storage.rs`, and `src/ops/**`. Items are keyed by",
        "file and const name or normalized expression; line numbers are not part",
        "of a key. Test-gated items and const-generic parameters are excluded.",
        "",
        f"- routing thresholds: {counts[ROUTING]}",
        f"- kernel tuning parameters: {counts[TUNING]}",
        f"- structural: {counts[STRUCTURAL]}",
        f"- cited: {statuses['cited']}",
        f"- uncited: {statuses['uncited']}",
        f"- structural-exempt: {statuses['structural-exempt']}",
        "",
    ]
    for title, item_class in (
        ("Routing thresholds", ROUTING),
        ("Kernel tuning parameters", TUNING),
        ("Structural", STRUCTURAL),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| key | value | unit | axis | routes | status | citation |")
        lines.append("|---|---:|---|---|---|---|---|")
        for item in items:
            if item["class"] != item_class:
                continue
            value = item["value"]
            citation = item["citation"] or ""
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    item["key"],
                    value,
                    item["unit"],
                    item["axis"],
                    item["routes"],
                    citation_status(item),
                    citation,
                )
            )
        lines.append("")
    for title, item_class in (
        ("Uncited routing thresholds", ROUTING),
        ("Uncited kernel tuning parameters", TUNING),
    ):
        uncited = [
            item
            for item in items
            if item["class"] == item_class and item["citation"] is None
        ]
        lines.append(f"## {title}")
        lines.append("")
        if uncited:
            for item in uncited:
                lines.append(f"- `{item['key']}` ({item['value']} {item['unit']})")
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def serialize_json(items: list[dict]) -> str:
    return json.dumps(items, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_json(path: Path, items: list[dict]) -> None:
    path.write_text(serialize_json(items), encoding="utf-8")


def write_checksum(script_path: Path, out_dir: Path) -> None:
    digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
    (out_dir / f"{script_path.name}.sha256").write_text(
        f"{digest}  {script_path.name}\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="crate root")
    parser.add_argument("--out", required=True, type=Path, help="inventory.json path")
    parser.add_argument(
        "--md",
        type=Path,
        default=None,
        help="inventory.md path (default: beside --out)",
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=None,
        help="optional key -> source position JSON for non-structural items",
    )
    arguments = parser.parse_args(argv)
    if not (arguments.root / "src").is_dir():
        parser.error(f"{arguments.root} is not a crate root")
    items, positions = scan_with_positions(arguments.root)
    write_json(arguments.out, items)
    markdown_path = arguments.md or arguments.out.with_name("inventory.md")
    markdown_path.write_text(render_markdown(items, arguments.root), encoding="utf-8")
    if arguments.positions is not None:
        payload = {
            "format": "threshold-positions/1",
            "items": positions,
        }
        arguments.positions.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_checksum(Path(__file__).resolve(), arguments.out.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
