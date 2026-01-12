```python
import pandas as pd

df = pd.read_csv("bank.csv")
df.head()

```




<div>

    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30;"unemployed";"married";"primary";"no";1787;...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33;"services";"married";"secondary";"no";4789;...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35;"management";"single";"tertiary";"no";1350;...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30;"management";"married";"tertiary";"no";1476...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59;"blue-collar";"married";"secondary";"no";0;...</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Cell In[2], line 4
          2 import pandas as pd
          3 import matplotlib.pyplot as plt
    ----> 4 import seaborn as sns


    ModuleNotFoundError: No module named 'seaborn'



```python
df = pd.read_csv("bank.csv", sep = ';')
df.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>unemployed</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>1787</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>oct</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>4789</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1350</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>16</td>
      <td>apr</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1476</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>3</td>
      <td>jun</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4516</th>
      <td>33</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>-333</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>30</td>
      <td>jul</td>
      <td>329</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4517</th>
      <td>57</td>
      <td>self-employed</td>
      <td>married</td>
      <td>tertiary</td>
      <td>yes</td>
      <td>-3313</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>9</td>
      <td>may</td>
      <td>153</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4518</th>
      <td>57</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>295</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>aug</td>
      <td>151</td>
      <td>11</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4519</th>
      <td>28</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1137</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>6</td>
      <td>feb</td>
      <td>129</td>
      <td>4</td>
      <td>211</td>
      <td>3</td>
      <td>other</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4520</th>
      <td>44</td>
      <td>entrepreneur</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1136</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>3</td>
      <td>apr</td>
      <td>345</td>
      <td>2</td>
      <td>249</td>
      <td>7</td>
      <td>other</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
def replace_marital(val):
  if val == 'single':
    return 0
  else:
    return 1

df['marital'] = df['marital'].apply(replace_marital)
df.head()
```




<div>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>unemployed</td>
      <td>1</td>
      <td>primary</td>
      <td>no</td>
      <td>1787</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>oct</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>services</td>
      <td>1</td>
      <td>secondary</td>
      <td>no</td>
      <td>4789</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>management</td>
      <td>0</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1350</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>16</td>
      <td>apr</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>management</td>
      <td>1</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1476</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>3</td>
      <td>jun</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>blue-collar</td>
      <td>1</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["housing"] = df["housing"].map({"yes": 1, "no": 0}.get)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>unemployed</td>
      <td>1</td>
      <td>primary</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>oct</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>services</td>
      <td>1</td>
      <td>secondary</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>yes</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>management</td>
      <td>0</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>no</td>
      <td>cellular</td>
      <td>16</td>
      <td>apr</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>management</td>
      <td>1</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>yes</td>
      <td>unknown</td>
      <td>3</td>
      <td>jun</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>blue-collar</td>
      <td>1</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["loan"] = df["loan"].replace({"yes": 1, "no": 0})
df.head()
```

    /tmp/ipykernel_18086/1568368702.py:1: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df["loan"] = df["loan"].replace({"yes": 1, "no": 0})





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>unemployed</td>
      <td>1</td>
      <td>primary</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>oct</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>services</td>
      <td>1</td>
      <td>secondary</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>management</td>
      <td>0</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>apr</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>management</td>
      <td>1</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>jun</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>blue-collar</td>
      <td>1</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["loan"] = df["loan"].replace({
"yes": 1,
"no": 0})
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>NaN</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>NaN</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>NaN</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>NaN</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>NaN</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["job"].unique()
```




    array(['unemployed', 'services', 'management', 'blue-collar',
           'self-employed', 'technician', 'entrepreneur', 'admin.', 'student',
           'housemaid', 'retired', 'unknown'], dtype=object)




```python
df["job"] = df["job"].replace({'unemployed': 0,
                   'services': 0,
                   'management': 1,
                   'blue-collar': 0,
                   'self-employed': 0,
                   'technician': 1,
                   'entrepreneur': 1,
                   'admin.': 0,
                   'student': 1,
                   'housemaid': 0,
                   'retired': 0,
                   'unknown': np.nan})
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>10</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>5</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>4</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>6</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>5</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["month"].unique()
```




    array(['oct', 'may', 'apr', 'jun', 'feb', 'aug', 'jan', 'jul', 'nov',
           'sep', 'mar', 'dec'], dtype=object)




```python
df.month = df.month.map({
    'oct': 10,
    'may': 5,
    'apr': 4, 
    'jun': 6,
    'feb': 2,
    'aug': 8,
    'jan': 1,
    'jul': 7,
    'nov': 11,
    'sep': 9,
    'mar': 3,
    'dec': 12
})
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>NaN</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>NaN</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>NaN</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>NaN</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>NaN</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>747</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>23</td>
      <td>NaN</td>
      <td>141</td>
      <td>2</td>
      <td>176</td>
      <td>3</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>307</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>14</td>
      <td>NaN</td>
      <td>341</td>
      <td>1</td>
      <td>330</td>
      <td>2</td>
      <td>other</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>147</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>6</td>
      <td>NaN</td>
      <td>151</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>221</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>14</td>
      <td>NaN</td>
      <td>57</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>43</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>-88</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>17</td>
      <td>NaN</td>
      <td>313</td>
      <td>1</td>
      <td>147</td>
      <td>2</td>
      <td>failure</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["education"].unique()
```




    array(['primary', 'secondary', 'tertiary', 'unknown'], dtype=object)




```python
df.education = df.education.map({
    'primary': 1, 
    'secondary': 2,
    'tertiary': 3,
    'unknown': np.nan
})
```


```python
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>10</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>5</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>4</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>6</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>5</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>747</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>23</td>
      <td>2</td>
      <td>141</td>
      <td>2</td>
      <td>176</td>
      <td>3</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>307</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>14</td>
      <td>5</td>
      <td>341</td>
      <td>1</td>
      <td>330</td>
      <td>2</td>
      <td>other</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>147</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>6</td>
      <td>5</td>
      <td>151</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>221</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>14</td>
      <td>5</td>
      <td>57</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>43</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>-88</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>17</td>
      <td>4</td>
      <td>313</td>
      <td>1</td>
      <td>147</td>
      <td>2</td>
      <td>failure</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["poutcome"].unique()
```




    array(['unknown', 'failure', 'other', 'success'], dtype=object)




```python
df.poutcome = df.poutcome.map({
    'unknown': np.nan, 
    'failure': 1,
    'other': 2,
    'success': 3
})
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>1787</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>NaN</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>4789</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>NaN</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>1350</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>NaN</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>1476</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>NaN</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>NaN</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>747</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>23</td>
      <td>NaN</td>
      <td>141</td>
      <td>2</td>
      <td>176</td>
      <td>3</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>307</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>14</td>
      <td>NaN</td>
      <td>341</td>
      <td>1</td>
      <td>330</td>
      <td>2</td>
      <td>2.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>147</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>6</td>
      <td>NaN</td>
      <td>151</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>221</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>14</td>
      <td>NaN</td>
      <td>57</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>43</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>-88</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>17</td>
      <td>NaN</td>
      <td>313</td>
      <td>1</td>
      <td>147</td>
      <td>2</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["balance"] = df["balance"].apply(lambda v: (v - df["balance"].min())/ (df["balance"].max() - df["balance"].min()))
```


```python
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.068455</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>NaN</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.108750</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>NaN</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.062590</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>NaN</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.064281</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>NaN</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.044469</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>NaN</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.054496</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>23</td>
      <td>NaN</td>
      <td>141</td>
      <td>2</td>
      <td>176</td>
      <td>3</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.048590</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>14</td>
      <td>NaN</td>
      <td>341</td>
      <td>1</td>
      <td>330</td>
      <td>2</td>
      <td>2.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.046442</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>6</td>
      <td>NaN</td>
      <td>151</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.047436</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>14</td>
      <td>NaN</td>
      <td>57</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>43</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.043288</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>17</td>
      <td>NaN</td>
      <td>313</td>
      <td>1</td>
      <td>147</td>
      <td>2</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["pdays"] = df["pdays"].apply(lambda v: (v - df["pdays"].min())/ (df["pdays"].max() - df["pdays"].min()))
```


```python
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.068455</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>NaN</td>
      <td>79</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.108750</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>NaN</td>
      <td>220</td>
      <td>1</td>
      <td>0.389908</td>
      <td>4</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.062590</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>NaN</td>
      <td>185</td>
      <td>1</td>
      <td>0.379587</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.064281</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>NaN</td>
      <td>199</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.044469</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>NaN</td>
      <td>226</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.054496</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>23</td>
      <td>NaN</td>
      <td>141</td>
      <td>2</td>
      <td>0.202982</td>
      <td>3</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.048590</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>14</td>
      <td>NaN</td>
      <td>341</td>
      <td>1</td>
      <td>0.379587</td>
      <td>2</td>
      <td>2.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.046442</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>6</td>
      <td>NaN</td>
      <td>151</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.047436</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>14</td>
      <td>NaN</td>
      <td>57</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>43</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.043288</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>17</td>
      <td>NaN</td>
      <td>313</td>
      <td>1</td>
      <td>0.169725</td>
      <td>2</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
df["duration"] = scaler.fit_transform(df[["duration"]])
df["pdays"] = scaler.fit_transform(df[["pdays"]])
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
      <td>0.068455</td>
      <td>0</td>
      <td>0</td>
      <td>cellular</td>
      <td>19</td>
      <td>NaN</td>
      <td>0.024826</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.108750</td>
      <td>1</td>
      <td>1</td>
      <td>cellular</td>
      <td>11</td>
      <td>NaN</td>
      <td>0.071500</td>
      <td>1</td>
      <td>0.389908</td>
      <td>4</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>1.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.062590</td>
      <td>1</td>
      <td>0</td>
      <td>cellular</td>
      <td>16</td>
      <td>NaN</td>
      <td>0.059914</td>
      <td>1</td>
      <td>0.379587</td>
      <td>1</td>
      <td>1.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>1.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>no</td>
      <td>0.064281</td>
      <td>1</td>
      <td>1</td>
      <td>unknown</td>
      <td>3</td>
      <td>NaN</td>
      <td>0.064548</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.044469</td>
      <td>1</td>
      <td>0</td>
      <td>unknown</td>
      <td>5</td>
      <td>NaN</td>
      <td>0.073486</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
