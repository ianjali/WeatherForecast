#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#%%
#column named datetime_ut
#reading t as index
data = pd.read_csv('data.csv', parse_dates=['datetime_utc'], index_col='datetime_utc')
data.head()

# %%
data = data.loc[:,[' _conds', ' _hum', ' _tempm']]
data = data.rename(index=str, columns={' _conds': 'condition', ' _hum': 'humidity', ' _pressurem': 'pressure', ' _tempm': 'temprature'})

# %%
data.info()
data.index.dtype

# %%
data.index = pd.to_datetime(data.index) #object to datetime
data.index
# %%
