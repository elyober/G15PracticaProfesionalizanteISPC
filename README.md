![image](https://user-images.githubusercontent.com/101228469/172445821-245dee9a-7c37-4f00-97b4-7c03965467f3.png)
# G15 Practica Profesionalizante ISPC 2023  TCDIA COHORTE 2022

## Predicción de sentimientos e impacto de acciones en redes sociales

### Integrantes

- Eliana Karina Steinbrecher
- Juan Alcaraz
- Roberto Schiaffino
- Hellmutt Popp
- Sergio Tamietto
- Alan Lovera
- CHACON, Claudio
- CONTRERAS, Carla
- CORDOBA, Marcelo
- CUESTAS, Natalia
  
1- Creación del repositorio remoto llamado G15 Practica Profesionalizante ISPC

2- Armado de Trello por Equipo: [https://trello.com/b/6X7uUnio/agile-board-template-trello](https://trello.com/invite/b/6X7uUnio/ATTI6abceef26fb024674de9bc66694f633883550417/practica-profesionalizante-equipo-15)

3- Definir tipo de problema y objetivo: El propósito del proyecto de Social Listening es implementar una estrategia de escucha activa en las redes sociales para comprender las percepciones, opiniones y sentimientos de clientes y la comunidad en línea..

(img\BRIEF SUMMARY GERENCIA.png)

4- Analisis de factibilidad legaly factibilidad técnica:

### Librerias Utilizadas

A lo largo de este proyecto, hemos empleado diversas librerias de Python para realizar análisis de datos, entrenar modelos y trabajar con datos. Algunas de las bibliotecas clave incluyen:

[scikit-learn](https://scikit-learn.org/stable/): Utilizada para el procesamiento de datos y la creación de modelos de aprendizaje automático.

[pandas](https://pandas.pydata.org/): Utilizada para la manipulación y análisis de datos tabulares.

[numpy](https://numpy.org/): Utilizada para realizar operaciones matriciales y numéricas.

[tweepy](https://www.tweepy.com/): Utilizada para interactuar con la API de Twitter y obtener datos relevantes.


- Facebook: se consiguio un archivo con xx registros asociados a 
- Instagram: no se valido a tiempo el uso de la api de toma de datos
- TikTok: no se valido a tiempo el uso de la api de toma de datos. Se realizo parte del codigo que instancia la misma. 
- Twitter: Actualmente, estamos experimentando restricciones en las funcionalidades proporcionadas por la API gratuita de Twitter. Hemos tomado la decisión de desestimar el uso de la API debido a las limitaciones de funcionalidades en su versión gratuita
- DataSet Amazon: se tomo un dataset de prueba de Amazon con Review de artículos de cocina y hogar con casi 7 millones de registros. Se etiqueto con sentimientos de pésimo, negativo, neutral, positiva y excelente. Se intento pasar a español para que sea fácil al momento de predecir pero resulto que las opciones para realizar esto estaban fuera de nuestro alcance.
Conclusion: dado los tiempos de respuesta para la obtencion de los accesos de autorización adicional de las API de las diferentes RRSS se concluye utilizar el archivo Meta.

5- Carga de datos: archivo resultante de la extraccion de datos de Meta

6- Fase de análisis exploratorio de datos

7- Análisis sobre los datos

8- Tratamiento de los datos anómalos.

9- Inspección gráfica de los estadísticos

10- Observar de manera gráfica si se aprecia relación entre "y" y las "X" (ekis).

11- Análisis de datos atípicos.

12- Distribución de probabilidad de las muestras.

13- Definir modelos a entrenar.

14- Al finalizar el ciclo, evaluar el costo computacional (o sea, poner un timer).

15- Calcular métricas

16- Calcular los estadísticos (test y sus pruebas para determinar la consistencia)

17- Evaluar modelos

18- Determinar qué tipo de problema tiene el modelo y valorar los errores (bias,varianza y error irreductible)

19- Conclusión y futuro plan de acción para iniciar otro ciclo

20- Evaluar hiper parámetros y estrategias de entrenamiento. 
-Repetir esto hasta lograr un modelo consistente y de performante
