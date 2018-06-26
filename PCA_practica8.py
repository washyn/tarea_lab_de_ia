
# coding: utf-8

# In[2]:


from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.mlab import PCA as mlabPCA
import pandas as pd


# # Actividad 1:
#  Carga de datos

# In[18]:


filename = "petrologydata.csv"
data = pd.read_csv(filename)


# # ACTIVIDAD 2:
#  Plotear las concentraciones de MgO versus SiO2
# - ¿Están correlacionados o no están correlacionados?  

# In[20]:


# MgO versus SiO2
y = data["MgO"]
x = data["SiO2"]

plt.scatter(x,y)
plt.grid(True)
plt.xlabel("MgO")
plt.ylabel("SiO2")
plt.title("SiO2 vs MgO")
# plt.savefig("svm.png")


# Si estan correlacionadas como se puede ver en el grafico.

# - ¿Qué es el coeficiente de correlación? 
# Es una medida de la rrealcion lineal entre dos variables cuantitativas aleatorias.
# Para este caso de MgO vs SiO2 es:

# In[27]:


correlacion_mgo_sio2 = data.corr(method="pearson")["MgO"]["SiO2"]


# ### Correlacion entre MgO y SiO2
# 
# -0.8335655524716729
# Hay un coheficiente de correlacion negativa.

# # Actividad 3
# ### Calcule los componentes principales de los datos de óxido.  
# 
# - Calcular el promedio en cada columna de datos 

# In[29]:


data.mean()


# - Calcule la matriz de covarianza 

# In[139]:


data.cov()


# In[140]:


data.corr()


# - Realice la descomposición de valores propios y asegúrese de que los valores propios estén en orden descendente. 

# In[88]:



# print(type(data.values))
# data.values
m_data = np.matrix(data.values)
# val_prop,vect_prop = np.linalg.eig(array_data)
m_data_t = m_data.T
m_cuadrada = m_data * m_data_t

val_prop, vect_prop = np.linalg.eig(np.matrix(data.cov()))

# val_prop


#  
# - Valores propios en orden descendente:
# 

# In[119]:


for i in val_prop:
    print(i)


# 
# - Calcule la fracción de la varianza explicada por cada componente principal 

# In[141]:


sumatoria = np.sum(val_prop)
for i in val_prop:
    print( i/sumatoria)


# - Haga un gráfico de la varianza porcentual acumulada contabilizada frente al índice del componente. 

# In[147]:


# crear la instancia de PCA
pca = PCA()
# ajuste en los datos
pca.fit(data.values)

kk = pca.fit_transform(data.values)
# acceso a valores y vectores
# print('PCA componentes')
# print(pca.components_)
# print('PCA varianza')
# print(pca.explained_variance_)

# plt.imshow(pca.inverse_transform(kk), cmap=plt.cm.Greys_r)

# a = pca.get_covariance()
# a.shape


pca = PCA()
pca.fit(data.values)
varianza = pca.explained_variance_ratio_
var_acum= np.cumsum(varianza)
plt.bar(range(len(varianza)), varianza, )
plt.grid(True)
plt.show()


# In[133]:


# print(1.651/3.448)


# In[143]:


plt.plot(range(len(varianza)), var_acum)
plt.grid(True)
plt.show()

