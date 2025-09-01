# PyTrendy 

PyTrendy is a robust solution for identifying and analyzing trends in time series. Unlike other trend detection packages, it considers post-processing, and handles both for gradual & abrupt trend cases with a high precision. It aims to be the best package for trend detection in python.

![alt-text](plots/pytrendy-gradual-demo.gif)

## Quickstart
Install the package from PyPi.
```
pip install pytrendy
```
Import pytrendy.
```py
import pytrendy as pt
```
Load daily time series data. In this case, we're using one of pytrendy's custom examples.
```py
df = pt.load_data('series_synthetic')
display(df)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>abrupt</th>
      <th>gradual</th>
      <th>gradual-noisy-20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-01-01</td>
      <td>19.578066</td>
      <td>12.500000</td>
      <td>27.514106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2025-01-02</td>
      <td>19.358378</td>
      <td>13.421717</td>
      <td>-6.620099</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2025-01-03</td>
      <td>19.228408</td>
      <td>13.474026</td>
      <td>22.122134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2025-01-04</td>
      <td>19.727130</td>
      <td>13.474026</td>
      <td>13.863735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2025-01-05</td>
      <td>20.773716</td>
      <td>14.505772</td>
      <td>8.884535</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2025-06-26</td>
      <td>4.718725</td>
      <td>20.616883</td>
      <td>19.790026</td>
    </tr>
    <tr>
      <th>177</th>
      <td>2025-06-27</td>
      <td>4.242065</td>
      <td>20.978084</td>
      <td>19.181404</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2025-06-28</td>
      <td>6.012296</td>
      <td>22.449495</td>
      <td>-6.563936</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2025-06-29</td>
      <td>4.603068</td>
      <td>23.486652</td>
      <td>48.291088</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2025-06-30</td>
      <td>4.435105</td>
      <td>22.240260</td>
      <td>3.343233</td>
    </tr>
  </tbody>
</table>
<p>181 rows × 4 columns</p>
</div>
</br>

Run trend detection & plot the results.
```py
results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True)
```
![alt-text](plots/pytrendy-gradual.png)

The results object can be used to summarise, further analyse, and generally inspect the trend detections.
```py
results.print_summary()
```
```
Detected: 
- 3 Uptrends. 
- 3 Downtrends.
- 3 Flats.
- 0 Noise.

The best detected trend is Down between dates 2025-05-09 - 2025-06-17

Full Results:
-------------------------------------------------------------------------------
            direction       start         end  days  total_change  change_rank
time_index                                                                   
9               Down  2025-05-09  2025-06-17    39    -73.253968            1
8                 Up  2025-04-02  2025-05-08    36     72.611833            2
5                 Up  2025-02-10  2025-03-14    32     24.632035            3
7               Down  2025-03-18  2025-04-01    14    -22.721861            4
1                 Up  2025-01-02  2025-01-24    22     14.013348            5
3               Down  2025-01-25  2025-02-05    11    -13.564214            6
4               Flat  2025-02-06  2025-02-09     3           NaN            7
6               Flat  2025-03-15  2025-03-17     2           NaN            8
10              Flat  2025-06-18  2025-06-29    11           NaN            9 
-------------------------------------------------------------------------------
```
You can directly call the object as a pandas dataframe.
```py
results.segments_df
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>direction</th>
      <th>segmenth_length</th>
      <th>start</th>
      <th>end</th>
      <th>trend_class</th>
      <th>change</th>
      <th>pct_change</th>
      <th>days</th>
      <th>total_change</th>
      <th>SNR</th>
      <th>change_rank</th>
    </tr>
    <tr>
      <th>time_index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Down</td>
      <td>38</td>
      <td>2025-05-09</td>
      <td>2025-06-17</td>
      <td>gradual</td>
      <td>-73.253968</td>
      <td>-0.805442</td>
      <td>39</td>
      <td>-73.253968</td>
      <td>21.122099</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Up</td>
      <td>34</td>
      <td>2025-04-02</td>
      <td>2025-05-08</td>
      <td>gradual</td>
      <td>73.687771</td>
      <td>3.944243</td>
      <td>36</td>
      <td>72.611833</td>
      <td>21.701162</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Up</td>
      <td>22</td>
      <td>2025-02-10</td>
      <td>2025-03-14</td>
      <td>gradual</td>
      <td>26.015512</td>
      <td>1.974942</td>
      <td>32</td>
      <td>24.632035</td>
      <td>18.871430</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Down</td>
      <td>14</td>
      <td>2025-03-18</td>
      <td>2025-04-01</td>
      <td>gradual</td>
      <td>-22.721861</td>
      <td>-0.591909</td>
      <td>14</td>
      <td>-22.721861</td>
      <td>16.762790</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Up</td>
      <td>17</td>
      <td>2025-01-02</td>
      <td>2025-01-24</td>
      <td>gradual</td>
      <td>14.013348</td>
      <td>1.044080</td>
      <td>22</td>
      <td>14.013348</td>
      <td>22.207980</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Down</td>
      <td>10</td>
      <td>2025-01-25</td>
      <td>2025-02-05</td>
      <td>gradual</td>
      <td>-13.564214</td>
      <td>-0.554982</td>
      <td>11</td>
      <td>-13.564214</td>
      <td>17.360657</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Flat</td>
      <td>9</td>
      <td>2025-02-06</td>
      <td>2025-02-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>20.126008</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Flat</td>
      <td>4</td>
      <td>2025-03-15</td>
      <td>2025-03-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>17.350339</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Flat</td>
      <td>13</td>
      <td>2025-06-18</td>
      <td>2025-06-29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11</td>
      <td>NaN</td>
      <td>19.039273</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>
</br>
By default, trends are sorted by there change_rank. This is ranks higher duration and magnitude of change to describe a trend's gravity erlative to others. You can sort by time index instead with filter_segments.
```
results.filter_segments(sort_by='time_index')
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>direction</th>
      <th>segmenth_length</th>
      <th>start</th>
      <th>end</th>
      <th>trend_class</th>
      <th>change</th>
      <th>pct_change</th>
      <th>days</th>
      <th>total_change</th>
      <th>SNR</th>
      <th>change_rank</th>
    </tr>
    <tr>
      <th>time_index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Up</td>
      <td>17</td>
      <td>2025-01-02</td>
      <td>2025-01-24</td>
      <td>gradual</td>
      <td>14.013348</td>
      <td>1.044080</td>
      <td>22</td>
      <td>14.013348</td>
      <td>22.207980</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Down</td>
      <td>10</td>
      <td>2025-01-25</td>
      <td>2025-02-05</td>
      <td>gradual</td>
      <td>-13.564214</td>
      <td>-0.554982</td>
      <td>11</td>
      <td>-13.564214</td>
      <td>17.360657</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Flat</td>
      <td>9</td>
      <td>2025-02-06</td>
      <td>2025-02-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>20.126008</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Up</td>
      <td>22</td>
      <td>2025-02-10</td>
      <td>2025-03-14</td>
      <td>gradual</td>
      <td>26.015512</td>
      <td>1.974942</td>
      <td>32</td>
      <td>24.632035</td>
      <td>18.871430</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Flat</td>
      <td>4</td>
      <td>2025-03-15</td>
      <td>2025-03-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>17.350339</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Down</td>
      <td>14</td>
      <td>2025-03-18</td>
      <td>2025-04-01</td>
      <td>gradual</td>
      <td>-22.721861</td>
      <td>-0.591909</td>
      <td>14</td>
      <td>-22.721861</td>
      <td>16.762790</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Up</td>
      <td>34</td>
      <td>2025-04-02</td>
      <td>2025-05-08</td>
      <td>gradual</td>
      <td>73.687771</td>
      <td>3.944243</td>
      <td>36</td>
      <td>72.611833</td>
      <td>21.701162</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Down</td>
      <td>38</td>
      <td>2025-05-09</td>
      <td>2025-06-17</td>
      <td>gradual</td>
      <td>-73.253968</td>
      <td>-0.805442</td>
      <td>39</td>
      <td>-73.253968</td>
      <td>21.122099</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Flat</td>
      <td>13</td>
      <td>2025-06-18</td>
      <td>2025-06-29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11</td>
      <td>NaN</td>
      <td>19.039273</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>
</br>

As well as filter only for a specific direction.
```
results.filter_segments(direction='Up')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>direction</th>
      <th>segmenth_length</th>
      <th>start</th>
      <th>end</th>
      <th>trend_class</th>
      <th>change</th>
      <th>pct_change</th>
      <th>days</th>
      <th>total_change</th>
      <th>SNR</th>
      <th>change_rank</th>
    </tr>
    <tr>
      <th>time_index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Up</td>
      <td>34</td>
      <td>2025-04-02</td>
      <td>2025-05-08</td>
      <td>gradual</td>
      <td>73.687771</td>
      <td>3.944243</td>
      <td>36</td>
      <td>72.611833</td>
      <td>21.701162</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Up</td>
      <td>22</td>
      <td>2025-02-10</td>
      <td>2025-03-14</td>
      <td>gradual</td>
      <td>26.015512</td>
      <td>1.974942</td>
      <td>32</td>
      <td>24.632035</td>
      <td>18.871430</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Up</td>
      <td>17</td>
      <td>2025-01-02</td>
      <td>2025-01-24</td>
      <td>gradual</td>
      <td>14.013348</td>
      <td>1.044080</td>
      <td>22</td>
      <td>14.013348</td>
      <td>22.207980</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
</br>

## Upcoming
- More DEMO examples.
- Automated testing in CI/CD pipeline.
- Documentation, moving more verbose tutorials to there.