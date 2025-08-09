# %%
import pandas as pd

# %%
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True, index_col="date")
df.plot(figsize=(20,3))

# %%
df['value'].values

# %%
df['diff'] = df['value'].diff(periods=1)
df.plot(figsize=(20,3))

# %%
df['rolling_avg'] = df['value'].rolling(14).mean()
df.plot(figsize=(20,3))


# %%
import matplotlib.pyplot as plt
df['diff_2'] = (df['rolling_avg'].diff(periods=1) * 20) - 14
ax = df.plot(figsize=(20,3))
ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()

# %%
