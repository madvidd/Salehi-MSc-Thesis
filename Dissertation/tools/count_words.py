"""Report a reproducible dissertation text word count.

The count includes the abstract and Chapters 1--5. It excludes references,
appendices, tables, figures, displayed equations, LaTeX commands and inline
mathematics. The result is suitable for keeping the template's word-count page
current while the final experimental table is still being populated.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "chapters" / "abstract.tex",
    ROOT / "chapters" / "introduction.tex",
    ROOT / "chapters" / "theory.tex",
    ROOT / "chapters" / "technical.tex",
    ROOT / "chapters" / "technical2.tex",
    ROOT / "chapters" / "conclusions.tex",
]

EXCLUDED_ENVIRONMENTS = (
    "figure",
    "table",
    "longtable",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
)


def remove_environment(text: str, environment: str) -> str:
    pattern = re.compile(
        rf"\\begin\{{{re.escape(environment)}\}}.*?"
        rf"\\end\{{{re.escape(environment)}\}}",
        flags=re.DOTALL,
    )
    return pattern.sub(" ", text)


def visible_text(source: str) -> str:
    source = re.sub(r"(?<!\\)%.*", " ", source)
    for environment in EXCLUDED_ENVIRONMENTS:
        source = remove_environment(source, environment)
    source = re.sub(r"\$\$.*?\$\$", " ", source, flags=re.DOTALL)
    source = re.sub(r"\$.*?\$", " ", source, flags=re.DOTALL)
    source = re.sub(r"\\\[.*?\\\]", " ", source, flags=re.DOTALL)
    source = re.sub(r"\\cite[tp]?\{[^}]*\}", " ", source)
    source = re.sub(r"\\(?:ref|eqref|pageref|label|url|path)\{[^}]*\}", " ", source)
    source = re.sub(r"\\(?:chapter|section|subsection|subsubsection)\*?\{([^}]*)\}", r" \1 ", source)
    source = re.sub(r"\\text(?:bf|it|tt|sc)\{([^}]*)\}", r" \1 ", source)
    source = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", source)
    source = source.replace("{", " ").replace("}", " ")
    source = source.replace("~", " ").replace("\\&", "and")
    return source


def main() -> None:
    text = "\n".join(
        visible_text(path.read_text(encoding="utf-8", errors="ignore"))
        for path in SOURCES
    )
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text)
    print(len(words))


if __name__ == "__main__":
    main()
