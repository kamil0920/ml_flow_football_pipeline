import pandas as pd
import pytest
from pipelines.common import FlowMixin

@pytest.fixture
def sample_df():
    rows = []
    # older seasons data: 5 rows each season
    for season in [2018, 2019]:
        for i in range(5):
            rows.append({
                'match_api_id': f'{season}_{i}',
                'season': season,
                'stage': i + 1,
                'date': pd.Timestamp(f'{season}-01-{i+1}'),
                'result_match': i % 3,
                'points_home': i,
                'points_away': i + 1,
                'feat1': season * 10 + i,
                'feat2': i * 2,
            })
    # newest season data: stages 1 to 5, 3 rows per stage
    for stage in range(1, 6):
        for j in range(3):
            rows.append({
                'match_api_id': f'2020_{stage}_{j}',
                'season': 2020,
                'stage': stage,
                'date': pd.Timestamp(f'2020-02-{stage:02d}') + pd.Timedelta(days=j),
                'result_match': (stage + j) % 3,
                'points_home': stage + j,
                'points_away': stage,
                'feat1': 2020 * 10 + stage + j,
                'feat2': (stage + j) * 2,
            })
    return pd.DataFrame(rows)

def test_splits_count_and_ids(monkeypatch, sample_df):
    # Monkeypatch load_train_dataset to return our sample data
    monkeypatch.setattr(FlowMixin, 'load_train_dataset', lambda self: sample_df)

    tr = FlowMixin()
    tr.n_older_seasons = 2
    splits = tr.create_temporal_splits(max_test_stage=5)

    assert len(splits) == 3
    for idx, split in enumerate(splits):
        assert split['split_id'] == idx
        assert split['test_stage'] == idx + 3
        assert split['val_stage'] == idx + 2

def test_split_contents(monkeypatch, sample_df):
    monkeypatch.setattr(FlowMixin, 'load_train_dataset', lambda self: sample_df)
    tr = FlowMixin()
    tr.n_older_seasons = 2
    splits = tr.create_temporal_splits(max_test_stage=5)

    split0 = splits[0]
    expected_old = sample_df[sample_df['season'].isin([2018, 2019])]
    expected_new_train = sample_df[(sample_df['season'] == 2020) & (sample_df['stage'] < 2)]
    assert split0['train_size'] == len(expected_old) + len(expected_new_train)
    expected_val = sample_df[(sample_df['season'] == 2020) & (sample_df['stage'] == 2)]
    assert split0['val_size'] == len(expected_val)
    expected_test = sample_df[(sample_df['season'] == 2020) & (sample_df['stage'] == 3)]
    assert split0['test_size'] == len(expected_test)

    assert list(split0['X_train'].columns) == ['feat1', 'feat2']
    assert list(split0['X_val'].columns) == ['feat1', 'feat2']
    assert list(split0['X_test'].columns) == ['feat1', 'feat2']

    val_rows = expected_val.reset_index(drop=True)
    for col in ['feat1', 'feat2']:
        assert all(split0['X_val'][col].values == val_rows[col].values)

@pytest.mark.parametrize("test_stage, expected_val_count, expected_test_count", [
    (4, 3, 3),
    (5, 3, 3),
])
def test_other_splits_sizes(monkeypatch, sample_df, test_stage, expected_val_count, expected_test_count):
    monkeypatch.setattr(FlowMixin, 'load_train_dataset', lambda self: sample_df)
    tr = FlowMixin()
    tr.n_older_seasons = 2
    splits = tr.create_temporal_splits(max_test_stage=5)

    split = next(s for s in splits if s['test_stage'] == test_stage)
    assert split['val_size'] == expected_val_count
    assert split['test_size'] == expected_test_count
