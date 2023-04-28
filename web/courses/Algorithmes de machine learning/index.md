# Algorithmes de machine learning

## **Les algorithmes les plus utilisés en machine learning**

!!! Info ""
    Les algorithmes les plus utilisés en machine learning sont les suivants:

!!! Info "Régression linéaire"
    C'est un algorithme simple qui est utilisé pour prédire la valeur d'une variable continue en utilisant une ou plusieurs variables explicatives.

!!! Info "Régression logistique"
    C'est un algorithme de classification utilisé pour prédire une variable de sortie binaire.

!!! Info "Forêt aléatoire (Random Forest)"
    C'est un algorithme de classification et de régression basé sur l'apprentissage ensembliste qui utilise plusieurs arbres de décision pour prédire la sortie.

!!! Info "K-Nearest Neighbors (K-NN)"
    C'est un algorithme de classification et de régression basé sur l'apprentissage par instance qui prédit la sortie en utilisant les données les plus proches de l'exemple à prédire.

!!! Info "Naive Bayes"
    C'est un algorithme de classification probabiliste qui prédit la classe d'une instance en utilisant les probabilités conditionnelles de chaque classe donnée les caractéristiques de l'instance.

!!! Info "Arbre de décision"
    C'est un algorithme de classification et de régression qui crée un arbre de décision pour représenter les relations entre les variables d'entrée et les sorties.

!!! Info "Support Vector Machine (SVM)"
    C'est un algorithme de classification qui définit une frontière de décision en utilisant les données d'entraînement les plus représentatives pour séparer les différentes classes.

!!! Info "Réseau de neurones artificiels (RNA)"
    C'est un algorithme de deep learning qui modélise les connexions entre les neurones pour effectuer des tâches complexes telles que la reconnaissance d'images ou la génération de langage.

???+ note

    Il est important de noter que le choix de l'algorithme dépend des données d'entrée, de la tâche de machine learning et de l'objectif de l'analyse. Il est donc souvent nécessaire d'essayer plusieurs algorithmes pour trouver le meilleur pour chaque cas d'utilisation.

## **Implémentation simple de la régression linéaire**

!!! Info ""
    Voici une implémentation simple de la régression linéaire en utilisant Python avec le module scikit-learn:

``` py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data[['Variable_1', 'Variable_2', 'Variable_3']]
y = data['Variable_cible']

# création d'un modèle de régression linéaire
reg = LinearRegression().fit(X, y)

# coefficients de régression
print("Coefficients de régression:", reg.coef_)

# intercept de la régression
print("Intercept de la régression:", reg.intercept_)

# prédiction sur les données d'entraînement
y_pred = reg.predict(X)

# calcul de la performance du modèle
r2_score = reg.score(X, y)
print("Score R^2:", r2_score)
```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle de régression linéaire est créé en utilisant la fonction `LinearRegression` du module scikit-learn. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable `y_pred`. Enfin, nous calculons le score R^2 pour évaluer la performance du modèle.


## **Implémentation simple de la régression logistique**

!!! Info ""
    Voici une implémentation simple de la régression logistique en utilisant Python avec le module scikit-learn:

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data[['Variable_1', 'Variable_2', 'Variable_3']]
y = data['Variable_cible']

# création d'un modèle de régression logistique
logreg = LogisticRegression().fit(X, y)

# coefficients de régression
print("Coefficients de régression:", logreg.coef_)

# intercept de la régression
print("Intercept de la régression:", logreg.intercept_)

# prédiction sur les données d'entraînement
y_pred = logreg.predict(X)

# calcul de la performance du modèle
accuracy = logreg.score(X, y)
print("Précision:", accuracy)

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle de régression logistique est créé en utilisant la fonction `LogisticRegression` du module scikit-learn. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable `y_pred`. Enfin, nous calculons la précision du modèle pour évaluer la performance. Notez que la régression logistique est un algorithme de classification binaire, donc la variable cible doit être binaire.

## **Implémentation simple de Forêt aléatoire (Random Forest)**

!!! Info ""
    Voici une implémentation simple de l'algorithme Random Forest en utilisant Python avec le module scikit-learn:

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data.drop('Variable_cible', axis=1)
y = data['Variable_cible']

# création d'un modèle de Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# prédiction sur les données d'entraînement
y_pred = clf.predict(X)

# calcul de la performance du modèle
accuracy = clf.score(X, y)
print("Précision:", accuracy)

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle de Random Forest est créé en utilisant la fonction `RandomForestClassifier` du module scikit-learn. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable `y_pred`. Enfin, nous calculons la précision du modèle pour évaluer la performance. Notez que ce code implémente un modèle de Random Forest pour une classification binaire, mais vous pouvez également utiliser l'algorithme pour des tâches de régression en utilisant la fonction `RandomForestRegressor` au lieu de `RandomForestClassifier`.

## **Implémentation simple de K-Nearest Neighbors (K-NN)**

!!! Info ""
    Voici une implémentation simple de l'algorithme K-Nearest Neighbors (K-NN) en utilisant Python avec le module scikit-learn:

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data.drop('Variable_cible', axis=1)
y = data['Variable_cible']

# création d'un modèle K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# prédiction sur les données d'entraînement
y_pred = knn.predict(X)

# calcul de la performance du modèle
accuracy = knn.score(X, y)
print("Précision:", accuracy)

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle K-NN est créé en utilisant la fonction `KNeighborsClassifier` du module scikit-learn. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable y_pred. Enfin, nous calculons la précision du modèle pour évaluer la performance. Notez que ce code implémente un modèle K-NN pour une classification binaire, mais vous pouvez également utiliser l'algorithme pour des tâches de régression en utilisant la fonction `KNeighborsRegressor` au lieu de `KNeighborsClassifier`.


## **Implémentation simple de Naive Bayes**

!!! Info ""
    Voici une implémentation simple de l'algorithme Naive Bayes en utilisant Python avec le module scikit-learn:

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data.drop('Variable_cible', axis=1)
y = data['Variable_cible']

# création d'un modèle Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)

# prédiction sur les données d'entraînement
y_pred = gnb.predict(X)

# calcul de la performance du modèle
accuracy = gnb.score(X, y)
print("Précision:", accuracy)

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle Naive Bayes est créé en utilisant la fonction `GaussianNB` du module scikit-learn. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable `y_pred`. Enfin, nous calculons la précision du modèle pour évaluer la performance. Notez que ce code implémente un modèle Naive Bayes pour une classification binaire, mais vous pouvez également utiliser l'algorithme pour des tâches de régression en utilisant la fonction `MultinomialNB` ou `BernoulliNB` selon le type de données.

## **Implémentation simple d'Arbre de décision**

!!! Info ""
    Voici une implémentation simple de l'algorithme d'Arbre de décision en utilisant Python avec le module scikit-learn:

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data.drop('Variable_cible', axis=1)
y = data['Variable_cible']

# création d'un modèle d'Arbre de décision
dt = DecisionTreeClassifier()
dt.fit(X, y)

# prédiction sur les données d'entraînement
y_pred = dt.predict(X)

# calcul de la performance du modèle
accuracy = dt.score(X, y)
print("Précision:", accuracy)

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle d'Arbre de décision est créé en utilisant la fonction `DecisionTreeClassifier` du module scikit-learn. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable `y_pred`. Enfin, nous calculons la précision du modèle pour évaluer la performance. Notez que ce code implémente un modèle d'Arbre de décision pour une classification binaire, mais vous pouvez également utiliser l'algorithme pour des tâches de régression en utilisant la fonction `DecisionTreeRegressor`.

## **Implémentation simple de Support Vector Machine (SVM)**

!!! Info ""
    Voici une implémentation simple de l'algorithme Support Vector Machine (SVM) en utilisant Python avec le module scikit-learn:

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn import svm

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data.drop('Variable_cible', axis=1)
y = data['Variable_cible']

# création d'un modèle SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

# prédiction sur les données d'entraînement
y_pred = clf.predict(X)

# calcul de la performance du modèle
accuracy = clf.score(X, y)
print("Précision:", accuracy)

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données. Le modèle SVM est créé en utilisant la classe `SVC` du module scikit-learn. Nous spécifions ici que nous souhaitons utiliser un noyau linéaire et un coefficient de régularisation `C` de 1. Ensuite, nous utilisons le modèle pour prédire la variable cible en utilisant les variables explicatives, en stockant les prédictions dans la variable `y_pred`. Enfin, nous calculons la précision du modèle pour évaluer la performance. Notez que vous pouvez également utiliser d'autres types de noyaux, tels que les noyaux polynomiaux et les noyaux Gaussiens, pour résoudre des tâches de classification et de régression.

## **Implémentation simple de Réseau de neurones artificiels (RNA)**

!!! Info ""
    Voici une implémentation simple de l'algorithme Réseau de neurones artificiels (RNA) en utilisant Python avec le module Keras :

```py linenums="1"
# importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# chargement des données
data = pd.read_csv("data.csv")

# sélection des variables explicatives et de la variable cible
X = data.drop('Variable_cible', axis=1)
y = data['Variable_cible']

# normalisation des données
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# création d'un modèle de RNA
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(1, activation='sigmoid'))

# compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# entraînement du modèle
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# évaluation du modèle
score = model.evaluate(X, y)
print("Précision:", score[1])

```

!!! Info ""
    Dans cet exemple, nous importons les bibliothèques numpy et pandas pour charger et manipuler les données, et nous utilisons Keras pour construire et entraîner le modèle de RNA. Nous normalisons d'abord les données pour améliorer la convergence de l'entraînement. Ensuite, nous définissons un modèle séquentiel avec deux couches cachées utilisant la fonction d'activation `relu`, ainsi qu'une couche de sortie utilisant la fonction d'activation `sigmoid`. Nous compilons le modèle en spécifiant l'optimiseur `adam`, la fonction de coût `binary_crossentropy` (car nous résolvons ici une tâche de classification binaire) et les métriques d'évaluation `accuracy`. Enfin, nous entraînons le modèle en utilisant les données d'entraînement, et nous évaluons la performance en utilisant la précision. Il est important de noter que ceci n'est qu'une implémentation simple, et qu'il est souvent nécessaire d'expérimenter avec différents architectures et hyperparamètres pour obtenir les meilleurs résultats sur les données réelles.


