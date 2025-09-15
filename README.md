# Proyecto SIMA

Para este proyecto, se veló mantener en un directorio de github todas las versiones que nos llevaron a la conclusión final para selección del modelo presentado en el reporte final. Los archivos finales que fueron los mostrados en el streamlit son "monterrey_air.py", el cuál se complementa con "monterrey_test.py". 
Nuestra herramienta interactiva ligada a este documento es la siguiente: https://simaprojectapp-ma2003b.streamlit.app/

A continuación se detalla que hace cada uno de los directiorios y archivos contenidos en el GitHub:

Archivos python:
1. Manova.py: con este archivo se probó MANOVA para comprobar nuestra hipótesis, que a lo largo del día, existen múltiples horarios con distintas combinaciones de concentraciones en gases.
2. SIMA_firstEDA.ipynb: nuestro primer acercamiento a comprender la base de datos se encuentra en este archivo.
3. Sima_databases.ipynb: este notebook contiene la limpieza final de los archivos, en donde se hace la selección de estaciones finales, relleno de valores nulos mediante imputación y finalmente, genera un archivo de excel que contiene nuestra selección de 5 estaciones, en conjunto a su análisis.
4. monterrey_air.py: este archivo es la base del streamlit, que es nuestar aplicación interactiva. Contiene tanto el contenido del análisis explroativo como también la segmentación por clusters de las bases de datos.
5. test_monterrey.py: este es complementorio a monterrey_air.py y sirve para los casos de prueba.

Directorios:
1. Bases_Datos: contiene las bases de datos brindadas por nuestro socioformador SIMA, en conjunto a nuestro f24_clean.xlsx que posee el archivo con las estaciones seleccionadas y con la imputación de los datos realizada.
2. Boxplots: resultados del análisis explorativo donde se encuentra los boxplots e histogramas realizados en los archivos.
3. manova_outputs: contiene los resultados de las salidas del análisis manova realizado.
4. models: conjunto de arhivos .pkl que poseen modelos generados para nuestro acercamiento por medio de kmeans, el cual fue descartado.
