"""Static source/package checks. Never imports the attention or ML runtime."""
import ast
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parents[1]
count = 0
for folder in ("tao_attention", "examples", "tests", "tools"):
    for path in (root / folder).rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(root)))
        count += 1
metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
for name in ("README.md", "LICENSE", "NOTICE"):
    if not (root / name).is_file():
        raise FileNotFoundError(name)
print(f"Parsed {count} Python files; metadata: {metadata['project']['name']}")
