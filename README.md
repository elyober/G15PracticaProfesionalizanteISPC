![image](https://user-images.githubusercontent.com/101228469/172445821-245dee9a-7c37-4f00-97b4-7c03965467f3.png)
# G15 Practica Profesionalizante ISPC 2023  TCDIA COHORTE 2022

## Social Media Listening :speech_balloon::ear:
***Es la práctica de monitorear y analizar activamente las conversaciones y discusiones en las redes sociales para comprender mejor las percepciones, opiniones y sentimientos de los usuarios y la comunidad en línea. Esta práctica es fundamental para las empresas y organizaciones que desean mantenerse al tanto de lo que se dice sobre sus productos, servicios y marca en las redes sociales.***

### Integrantes

- Eliana Karina Steinbrecher
- Juan Alcaraz
- Roberto Schiaffino
- Sergio Tamietto
- Alan Lovera
- Carla Contreras
- Natalia Cuestas

### Librerias Utilizadas :books:

A lo largo de este proyecto, hemos empleado diversas librerias de Python para realizar análisis de datos, entrenar modelos y trabajar con datos. Algunas de las bibliotecas clave incluyen:

[Pandas](https://pandas.pydata.org/): Utilizada para la manipulación y análisis de datos tabulares.

[Numpy](https://numpy.org/): Utilizada para realizar operaciones matriciales y numéricas.

[Tweepy](https://www.tweepy.com/): Utilizada para interactuar con la API de Twitter y obtener datos relevantes.

[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): Utilizada para limpiar y procesar texto HTML. 

[Scikit-Learn](https://scikit-learn.org/stable/): Utilizada para el procesamiento de datos y la creación de modelos de aprendizaje automático.

[Nltk](https://www.nltk.org/): Utilizada para el procesamiento de lenguaje natural, que incluye tokenización, lematización y otras tareas relacionadas con el texto.

[Joblib](https://joblib.readthedocs.io/en/latest/): Empleado para guardar y cargar modelos entrenados.

[MatplotLib](https://matplotlib.org/): Utilizada para la visualización de datos, como generar gráficos de WordCloud.

[Seaborn](https://seaborn.pydata.org/): Otra biblioteca para la visualización de datos, que proporciona gráficos más avanzados y atractivos que los ofrecidos por Matplotlib.

[TikTokApi](https://developers.tiktok.com/): Las API de contenido comercial permitirán al público y a los investigadores realizar búsquedas personalizadas basadas en nombres de anunciantes o palabras clave en anuncios y otros datos de contenido comercial almacenados en la biblioteca de contenido comercial.


1- **Creación del repositorio remoto llamado G15 Practica Profesionalizante ISPC**

2- **Armado de Trello por Equipo:**
[https://trello.com/b/6X7uUnio/agile-board-template-trello](https://trello.com/invite/b/6X7uUnio/ATTI6abceef26fb024674de9bc66694f633883550417/practica-profesionalizante-equipo-15)
<p align="center">
<img src="https://i.ibb.co/kgktYHG/Screenshot-11.jpg">
</p>

3- **Definir tipo de problema y objetivo:**
El propósito del proyecto de Social Listening es implementar una estrategia de escucha activa en las redes sociales para comprender las percepciones, opiniones y sentimientos de clientes y la comunidad en línea..

<p align="center">
  <img src="https://i.ibb.co/tKt32RT/BRIEF-SUMMARY-GERENCIA-1.png">
</p>

4- **Analisis de factibilidad legal y factibilidad técnica en RRSS:**

- **Facebook:** se consiguio un archivo con xx registros asociados a 
- **Instagram:** no se valido a tiempo el uso de la api de toma de datos
- **TikTok:** no se validó a tiempo el uso de la api de toma de datos.En espera de habilitación. Se realizó parte del código que instancia la misma. 
- **Twitter:** Actualmente, estamos experimentando restricciones en las funcionalidades proporcionadas por la API gratuita de Twitter. Hemos tomado la decisión de desestimar el uso de la API debido a las limitaciones de funcionalidades en su versión gratuita
- **DataSet Amazon:** Se utilizó un conjunto de datos de prueba de Amazon que contiene reseñas de productos relacionados con cocina y hogar, el cual consta de casi 7 millones de registros. Estos registros se etiquetaron con distintos sentimientos, incluyendo "pésimo," "negativo," "neutral," "positivo" y "excelente." Inicialmente, se intentó traducir las reseñas al español para facilitar el proceso de predicción. Sin embargo, se encontraron limitaciones técnicas que dificultaron esta tarea y estuvieron fuera de nuestro control para resolver en el tiempo requerido.

***Conclusion: Debido a la demora en la obtención de las credenciales de autorización adicional para acceder a las API de diferentes redes sociales, se tomó la decisión de utilizar el archivo "Meta" como fuente principal de datos. Esta elección se basó en consideraciones de tiempo y recursos disponibles.***

5- **Carga de datos:**

Este modelo se entrenará utilizando el conjunto de datos data_sentimientos.csv, el cual es una adaptación más reducida del conjunto de datos propuesto por Amazon Home_and_Kitchen_5.json. Este último contiene alrededor de 7 millones de reseñas que los usuarios de Amazon han realizado sobre productos de cocina.
El archivo destinado a aplicar el modelo entrenado es un conjunto de datos que presenta valoraciones sobre una marca de utensilios de cocina.

6- **Fase de análisis exploratorio de datos**

Una vez que se ha cargado el conjunto de datos, se procede a realizar un análisis para determinar cómo están representadas las diferentes etiquetas de sentimiento. Esto se logra agrupando y contabilizando la cantidad de cada etiqueta de sentimiento presente en el conjunto de datos. La idea es comprender la distribución y frecuencia de cada etiqueta, lo que permitirá visualizar si hay una representación desigual entre ellas.

7- **Análisis sobre los datos**

Tras esta agrupación, se examinan los resultados para evaluar la presencia de cada etiqueta en los datos. En el análisis particular proporcionado, se encontró que la etiqueta "Excelente" está considerablemente sobrerrepresentada en comparación con las otras etiquetas de sentimiento.

Este desequilibrio en la representación de las etiquetas puede influir en la capacidad del modelo para aprender de manera equitativa y precisa. Por ende, se decide igualar la cantidad de muestras para cada etiqueta de sentimiento seleccionando un número específico de muestras aleatorias de cada una. Este enfoque tiene como objetivo equilibrar la distribución de las etiquetas en el conjunto de datos, lo que podría mejorar la capacidad del modelo para generalizar y predecir con mayor precisión en todas las categorías de sentimiento.

8- **Tratamiento de los datos anómalos**
**Cambio de tipo de datos:** Las columnas "reviewText" y "sentimiento_marca" se convierten en cadenas (strings) para asegurar su compatibilidad y uniformidad.

**Redistribución de muestras:** Debido a la considerable disparidad en la cantidad de etiquetas "Excelente" en comparación con las demás, se realiza un muestreo aleatorio para tomar un número específico de instancias de cada etiqueta. En este caso, se selecciona un total de 90,000 instancias para cada etiqueta "Excelente", "Negativa", "Pesima" y "Positiva".

**Creación de un nuevo conjunto de datos equilibrado:** Las muestras seleccionadas aleatoriamente se combinan en un nuevo conjunto de datos llamado "combinado_df", donde cada etiqueta de sentimiento está igualmente representada. Este nuevo conjunto de datos está completamente aleatorizado para garantizar que no exista un sesgo en el orden de las instancias.

**Limpieza de HTML en las reseñas:** Se aplica una función que utiliza BeautifulSoup (biblioteca de Python) para limpiar posibles códigos HTML presentes en las reseñas, asegurando que solo se conserve el texto relevante y se elimine el marcado HTML.

9- **Inspección gráfica de los estadísticos**

10- **Observar de manera gráfica si se aprecia relación entre "y" y las "X" (ekis)**

11- **Análisis de datos atípicos**

12- **Distribución de probabilidad de las muestras**

13- **Definir modelos a entrenar**

14- **Al finalizar el ciclo, evaluar el costo computacional (o sea, poner un timer)**

15- **Calcular métricas**

16- **Calcular los estadísticos (test y sus pruebas para determinar la consistencia)**

17- **Evaluar modelos**

18- **Determinar qué tipo de problema tiene el modelo y valorar los errores (bias,varianza y error irreductible)**

19- **Conclusión y futuro plan de acción para iniciar otro ciclo**

20- **Evaluar hiper parámetros y estrategias de entrenamiento**
-***Repetir esto hasta lograr un modelo consistente y de performante***
