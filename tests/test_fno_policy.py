from pathlib import Path
import pandas as pd


def test_fno_universe_has_symbol_column_when_present():
    path=Path('data/fno_universe.csv')
    if not path.exists():
        return
    df=pd.read_csv(path)
    assert 'Symbol' in df.columns
    assert df['Symbol'].astype(str).str.strip().ne('').all()


def test_config_requires_fno_for_shorts():
    text=Path('config.yaml').read_text(encoding='utf-8')
    assert 'shorts_require_fno: true' in text
    assert 'fno_universe_file: data/fno_universe.csv' in text
