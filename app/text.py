"""Shared tokenizer. Keeps tokens like c++, c#, scikit-learn, ci/cd-ish terms intact."""
import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower().rstrip(".-") for t in _TOKEN.findall(text) if t.rstrip(".-")]
