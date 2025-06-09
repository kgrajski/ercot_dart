#!/usr/bin/env python3
"""
Import Standards Checker for ERCOT DART Project

This script checks that all internal imports use absolute paths with src. prefix.
"""

import ast
import sys
from pathlib import Path
from typing import List
from typing import Tuple


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check import statements."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations = []

    def visit_Import(self, node):
        """Check import statements."""
        for alias in node.names:
            if self._is_internal_module(alias.name):
                if not alias.name.startswith("src."):
                    self.violations.append(
                        (
                            node.lineno,
                            f"import {alias.name}",
                            f"Should be: import src.{alias.name}",
                        )
                    )

    def visit_ImportFrom(self, node):
        """Check from...import statements."""
        if node.module and self._is_internal_module(node.module):
            if not node.module.startswith("src."):
                imports = ", ".join([alias.name for alias in node.names])
                self.violations.append(
                    (
                        node.lineno,
                        f"from {node.module} import {imports}",
                        f"Should be: from src.{node.module} import {imports}",
                    )
                )
        elif node.level > 0:  # Relative import (from . or from ..)
            imports = ", ".join([alias.name for alias in node.names])
            self.violations.append(
                (
                    node.lineno,
                    f"from {'.' * node.level}{node.module or ''} import {imports}",
                    "Use absolute import with src. prefix instead",
                )
            )

    def _is_internal_module(self, module_name: str) -> bool:
        """Check if module is internal to our project."""
        if not module_name:
            return False

        internal_prefixes = [
            "data.",
            "etl.",
            "features.",
            "models.",
            "visualization.",
            "workflow.",
            "tests.",
        ]

        return any(
            module_name.startswith(prefix) or module_name in prefix.rstrip(".")
            for prefix in internal_prefixes
        )


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check a single Python file for import violations."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        checker = ImportChecker(filepath)
        checker.visit(tree)
        return checker.violations

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def main():
    """Main function to check import standards."""
    if len(sys.argv) < 2:
        print("Usage: python check_imports.py <file1> <file2> ...")
        sys.exit(1)

    total_violations = 0
    files_with_violations = 0

    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists() or not path.suffix == ".py":
            continue

        violations = check_file(path)
        if violations:
            files_with_violations += 1
            total_violations += len(violations)
            print(f"\n❌ {path}:")
            for line_no, current, suggestion in violations:
                print(f"  Line {line_no}: {current}")
                print(f"    → {suggestion}")

    if total_violations == 0:
        print("✅ All imports follow the standards!")
        sys.exit(0)
    else:
        print(
            f"\n❌ Found {total_violations} violations in {files_with_violations} files"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
