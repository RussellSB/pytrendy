# %%
import pandas as pd

# %%
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True, index_col="date")
df.plot(figsize=(20,3))

# %%
