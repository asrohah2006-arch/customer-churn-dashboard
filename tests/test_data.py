import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from data import load_and_clean, split_features_target, build_preprocessor

def test_dataset_and_preprocessing():
    root=Path(__file__).parents[1]
    df=load_and_clean(root/"data/raw/Telco-Customer-Churn.csv")
    assert df.shape == (7043,21)
    assert set(df.Churn.unique()) == {0,1}
    X,y=split_features_target(df); transformed=build_preprocessor(X).fit_transform(X.head(100))
    assert transformed.shape[0] == 100
    assert transformed.shape[1] > X.shape[1]
