import json

import pandas as pd

from scripts.extract import DataExtractor
from scripts.load import DataLoader
from scripts.transform import DataTransformer, load_data, save_data


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.called = False

    def get(self, url, params=None, headers=None, timeout=30):
        self.called = True
        return _FakeResponse(self.payload)


def test_extractor_fetch_and_flatten():
    payload = [
        {"id": 1, "name": "Ada", "address": {"city": "London"}, "tags": ["one", "two"]},
        {"id": 2, "name": "Grace", "address": {"city": "Arlington"}, "tags": ["three"]},
    ]
    session = _FakeSession(payload)
    extractor = DataExtractor("https://example.com", session=session)
    extractor.fetch_data().flatten_data()
    df = extractor.get_dataframe()
    assert session.called is True
    assert "address_city" in df.columns
    assert df["tags"].apply(lambda item: json.loads(item)).iloc[0] == ["one", "two"]


def test_transform_and_save_load(tmp_path):
    df = pd.DataFrame({"name": [" Ada ", "Grace"], "username": ["ada", "grace"]})
    transformer = DataTransformer(df)
    transformer.standardize_text(columns=["name", "username"], case="title")
    cleaned = transformer.get_dataframe()

    output_path = tmp_path / "output.csv"
    save_data(cleaned, str(output_path))
    loaded = load_data(str(output_path))

    assert loaded.loc[0, "name"] == "Ada"
    assert loaded.loc[0, "username"] == "Ada"


def test_loader_roundtrip(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    db_path = tmp_path / "pipeline.db"
    with DataLoader(str(db_path)) as loader:
        loader.load_to_sqlite(df, "items")
        tables = loader.list_tables()
        assert "items" in tables["name"].tolist()
        assert loader.get_row_count("items") == 2
