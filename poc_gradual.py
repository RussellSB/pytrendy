# %%
import pandas as pd

# %%
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True, index_col="date")
df.plot(figsize=(20,3))



# %%
import matplotlib.pyplot as plt
df['rolling_avg'] = df['value'].rolling(14, center=True).mean()
df['diff_2'] = (df['rolling_avg'].diff(periods=1) * 20) 
ax = df[['diff_2', 'value']].plot(figsize=(20,3))
ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()

# %%
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
df['sg_filter'] = savgol_filter(df['value'], window_length=14, polyorder=1, deriv=1)
df['sg_filter'] = df['sg_filter'].shift(7)
df['diff_2'] = (df['sg_filter'].diff(periods=1)) 
ax = df[['value', 'sg_filter']].plot(figsize=(20,3), secondary_y='sg_filter')
ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()
