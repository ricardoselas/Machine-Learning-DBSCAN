# Machine-Learning-DBSCAN
DBSCAN

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.cluster import DBSCAN
from collections import Counter
%matplotlib inline
rcParams['figure.figsize'] = 7.5, 6
sns.set_style('whitegrid')
df = sns.load_dataset("iris")
cols = ['comp da cepa', 'larg da cepa', 'comp da pétala', 'larg da pétala', 'espécie
df.columns = cols
X = df[cols[:4]].values
y = df['espécie'].values

modelo = DBSCAN(eps=0.8, min_samples=19).fit(X)
print(modelo)

outliers_df = pd.DataFrame(X)
print(Counter(modelo.labels_))

filtro = modelo.labels_ == -1
print(outliers_df[filtro])

cores = modelo.labels_
plt.scatter(X[:,2], X[:,1], c=cores, s=120)
plt.xlabel('Comprimento da Pétala')
plt.ylabel('largura da Cepa')
plt.title('DBScan para detecção de Outlier')
plt.show()
