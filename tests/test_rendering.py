from pathlib import Path

from dynamic_effective_carriers.rendering import generate_figures, generate_tables


def test_figure_and_table_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    figures = generate_figures(root)
    tables = generate_tables(root)
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
    assert all(path.is_file() and path.stat().st_size > 0 for path in tables)
