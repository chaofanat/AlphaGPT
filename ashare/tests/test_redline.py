# -*- coding: utf-8 -*-
"""红线守卫：ashare/ 内不得出现 GMtest 非数据层代码的导入。

约定（见 ashare/gm_bridge.py 模块注释）：
- `utils.*` 只允许出现在 gm_bridge.py（唯一转引点）
- `models` / `topics` / `factors` / `scorers` / `strategies` / `screening`
  （GMtest 其余顶层包）任何模块都不得导入
"""
import re
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"^\s*(?:from|import)\s+(models|topics|factors|scorers|strategies|screening)\b",
                       re.MULTILINE)
UTILS_IMPORT = re.compile(r"^\s*(?:from|import)\s+utils\b", re.MULTILINE)


def _py_sources():
    for p in PKG_DIR.rglob("*.py"):
        if "cache" in p.parts or "__pycache__" in p.parts:
            continue
        yield p


def test_no_gmtest_non_utils_imports():
    offenders = [f"{p.relative_to(PKG_DIR)}" for p in _py_sources()
                 if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"红线违规：以下文件导入了 GMtest 非数据层代码：{offenders}"


def test_utils_import_only_in_bridge():
    offenders = [f"{p.relative_to(PKG_DIR)}" for p in _py_sources()
                 if p.name != "gm_bridge.py" and UTILS_IMPORT.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"红线违规：以下文件绕过 gm_bridge 直接导入 utils：{offenders}"
