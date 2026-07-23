import pandas as pd
import joblib
from pathlib import Path

print('Loading dataset...')
df = pd.read_csv('comprehensive_strength.csv')
print(df.head().to_string())
print('\nColumns:', list(df.columns))

for name in ['model_random_forest.joblib','model_ann.joblib','model_svr.joblib','model_rsm_quadratic.joblib','best_compressive_strength_model.joblib']:
    path = Path(name)
    if path.exists():
        print(f'\nLoading {name}...')
        m = joblib.load(path)
        print('Model type:', type(m))
        if hasattr(m, 'feature_names_in_'):
            print('Features:', list(m.feature_names_in_))
        if hasattr(m, 'n_features_in_'):
            print('N features:', m.n_features_in_)
        if hasattr(m, 'classes_'):
            print('Classes:', m.classes_)
    else:
        print(f'{name} missing')
