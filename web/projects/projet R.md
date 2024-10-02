---
comments: true
---

# Profilage des chauffeurs

!!! Info "Membres du groupe"
    - [Abdellatif BELMADY](https://github.com/Abdellatif-belmady/)
    - [Fatine BOUSSATTINE](https://github.com/FatineDev/)
    - [Hamza HAJJINI](https://github.com/HAJJINIHamza/)
    - [Hamza Dribine](https://github.com/hamza-dri/)
    - [Mohamed Ait Hajjoub](https://github.com/)

## **Importer les packages**

```r linenums="1"
library(sf)           # manipulation des données spatiales
library(osmdata)      # extraction des données OpenStreetMap
library(leaflet)      # visualisation interactive avec leaflet
library(mapsf)        # cartographie statistique
library(lubridate)    # manipulation des dates
library(tidyverse)    # méta-package d'Hadley Wickham
```

`getwd()` est une fonction qui permet de récupérer le chemin absolu du répertoire de travail actuel.

```r linenums="7"
getwd()
```

??? success "Output"

    [1] "C:/Users/abdel/Documents"

## **Importer la data**

- Le premier fichier, `casabound.geojson`, est lu à l'aide de la fonction **st_read()** de la bibliothèque **sf**. Cette fonction est utilisée pour lire des fichiers de données spatiales tels que des fichiers shapefile, des fichiers GeoJSON, etc. Ici, il lit un fichier GeoJSON nommé "casabound.geojson" et stocke les données dans un objet nommé **casaBound**.

- Le deuxième fichier, `heetchmarchcrop.Rds`, est lu à l'aide de la fonction **readRDS()**. Cette fonction est utilisée pour lire des fichiers de données R sauvegardés en utilisant la fonction **saveRDS()**. Ici, il lit un fichier RDS nommé heetchmarchcrop.Rds et stocke les données dans un objet nommé **heetchPoints**.

- Le troisième fichier, `osmfeatures.Rds`, est également lu à l'aide de la fonction **readRDS()**. Comme le deuxième fichier, il s'agit d'un fichier RDS et est lu dans un objet nommé **osmFeatures**.

```r linenums="8"
casaBound <- st_read("DATA/casabound.geojson")
heetchPoints <- readRDS("DATA/heetchmarchcrop.Rds")
osmFeatures <- readRDS("DATA/osmfeatures.Rds")
```

??? success "Output"

    ![image](../../assets/images/img1.PNG)


## **Définir la problématique**

  A travers ce travail, nous cherchons à identifier les conducteurs qui respectent les règles de conduite et à évaluer leur sécurité sur la route, pour ce faire, nous nous concentrerons sur le calcul de la vitesse moyenne des conducteurs.

## **Résoudre la problématique**

  Ce code R `length(unique(heetchPoints$driver_id))` calcule le nombre de valeurs uniques dans la colonne **driver_id** de l'objet **heetchPoints**.

  La fonction **unique()** est utilisée pour extraire les valeurs uniques de la colonne driver_id. Ensuite, la fonction **length()** est utilisée pour renvoyer le nombre d'éléments dans le vecteur résultant.

```r linenums="11" title="Nombre de chauffeurs"
length(unique(heetchPoints$driver_id))
```
??? success "Output"
    [1] 1309

Le code R présenté ci-dessous est une fonction appelée `my_function`, qui prend un argument **id_driver**. La fonction effectue les opérations suivantes:

1. Initialise une variable **i** à zéro.

2. Affiche la valeur de **i**.

3. Filtre la table **heetchPoints** en fonction de la valeur **id_driver**.

4. Trier la table **driver** en fonction de la colonne **location_at_local_time**.

6. Effectue une projection de la table **driver_tri** dans une projection cartographique spécifique (**crs = 26191**).

7. Calcule les distances entre tous les points dans la table **driver_tri** à l'aide de la fonction **st_distance**.

8. Calcule la différence de temps entre chaque deux points consécutifs dans la table **driver_tri** à l'aide de la fonction **difftime**.

9. Filtre la table **driver_tri** pour conserver uniquement les points ayant une différence de temps entre 0.016 et 0.025 heures.

10. Calcule la vitesse entre chaque deux points successifs en divisant la distance sur le temps.

11. Filtre la table **driver_tri_2** pour ne conserver que les points ayant une vitesse entre 6 et 120 km/h.

12. Retourne la moyenne des vitesses de la table **driver_tri_3**.

```r linenums="12" title="Défenir la fonction qui calcul la moyenne des vitesses d'un chauffeur sur un jour"
i = 0
my_function <- function (id_driver){
  
  i=i+1
  print(i)
  driver <- heetchPoints %>% 
    filter(driver_id == id_driver) 
  
  # Prendre le premier jour + classer par location_at_local_time
#  jour <- driver %>% 
 #   filter(substr(driver$location_at_local_time, start = 9, stop = 10) == "01")
  
  #plot(driver$geometry, border = "red", lwd = 2)
  
#  time_tri <- order(jour$location_at_local_time)
  
 # jour_tri <- jour[time_tri,]
  
  #Triage temporel de la table driver 
  
  driver_tri_index <- order(driver$location_at_local_time)
  driver_tri <- driver[driver_tri_index,]
  
  # Projection des points
  driver_tri <- st_transform(x = driver_tri, crs = 26191)
  
  


  # Calculons les distances entres tous les points
  n <- nrow(driver_tri)  
  n
  list_distance <- list()
  for( i in 1:(n-1)){
    distance <- st_distance(x = driver_tri[i, ],
                            y = driver_tri[i+1, ],
                            by_element = TRUE)
    units(distance) <- "km"
    list_distance <- append (list_distance, list(distance))
  }
  length (list_distance)
  list_distance <- c(0, list_distance)
  
  driver_tri$distdiff <- list_distance
  
  
  
  
  list_time <- list()
  for( i in 1:(n-1)){
    date_point1 <- driver_tri$location_at_local_time[i]
    date_point2 <- driver_tri$location_at_local_time[i+1]
    diff?rence <- difftime(date_point2, date_point1, units = "hours")
    list_time <- append (list_time, list(diff?rence))
  }
  
  list_time <- c(0, list_time)
  driver_tri$timediff <- list_time
  #Calculons la liste des vitesse entre chaque deux points successifs en divisant la distance sur le temps
  
  driver_tri_2 <- driver_tri[driver_tri$timediff > 0.016 & driver_tri$timediff < 0.025, ]
  
  
  class(driver_tri_2$distdiff)
  class(driver_tri_2$timediff)
  
  driver_tri_2$distdiff <- as.numeric(driver_tri_2$distdiff)
  driver_tri_2$timediff <- as.numeric(driver_tri_2$timediff)
  
  driver_tri_2$vitesse <- driver_tri_2$distdiff / driver_tri_2$timediff
  driver_tri_3 <- driver_tri_2[driver_tri_2$vitesse >= 6 & driver_tri_2$vitesse <= 120, ]
  
  return (mean(driver_tri_3$vitesse))
}
```
Le code ci-dessous commence par créer un objet de type **data.frame** appelé **vitesse_table** à l'aide de la fonction **data.frame()**.

Ensuite, la boucle **for** est utilisée pour itérer sur une liste de trois valeurs de l'ID de conducteur **driver_id** comprises entre 10 et 12 inclusivement.

À chaque itération, le code crée une **liste driver_list** avec deux éléments : le premier est l'ID du conducteur et le deuxième est le résultat de la fonction **my_function()** avec l'ID du conducteur en argument.

Enfin, la fonction **rbind()** est utilisée pour ajouter la liste **driver_list** en tant que nouvelle ligne à la fin du **data.frame** ***vitesse_table**.

Ainsi, à la fin de la boucle **for**, **vitesse_table** contiendra une liste de conducteurs avec leurs ID et la valeur de la vitesse obtenue à l'aide de la fonction **my_function()**.

```r linenums="86" title="Calculons la moyenne des vitesse de tous les chauffeurs"
vitesse_table <- data.frame()

for (driver_id in unique(heetchPoints$driver_id)[10:12]){
  driver_list <- list(driver_id, my_function (driver_id))
  vitesse_table <- rbind(vitesse_table, driver_list)
  }
```







