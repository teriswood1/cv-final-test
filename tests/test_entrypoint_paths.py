import ast
from pathlib import Path
import runpy

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_DIRECTORIES = ("train", "eval", "predict", "test")
ENTRYPOINTS = [
    path
    for directory in ENTRYPOINT_DIRECTORIES
    for path in sorted((REPOSITORY_ROOT / directory).glob("*.py"))
    if path.name != "__init__.py"
]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)))
def test_entrypoint_resolves_repository_root(entrypoint):
    tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PROJECT_ROOT" for target in node.targets)
    )

    resolved_root = eval(
        compile(ast.Expression(assignment.value), str(entrypoint), "eval"),
        {"Path": Path, "__file__": str(entrypoint)},
    )

    assert resolved_root == REPOSITORY_ROOT


def test_entrypoints_import_without_running_main():
    for entrypoint in ENTRYPOINTS:
        runpy.run_path(str(entrypoint), run_name=f"smoke_{entrypoint.parent.name}_{entrypoint.stem}")
