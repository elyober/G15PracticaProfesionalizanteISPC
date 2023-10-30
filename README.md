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

<p align="center">
<img src="IMG\GRAF1.jpg">
</p>

8- **Tratamiento de los datos anómalos**

**Cambio de tipo de datos:** Las columnas "reviewText" y "sentimiento_marca" se convierten en cadenas (strings) para asegurar su compatibilidad y uniformidad.

**Redistribución de muestras:** Debido a la considerable disparidad en la cantidad de etiquetas "Excelente" en comparación con las demás, se realiza un muestreo aleatorio para tomar un número específico de instancias de cada etiqueta. En este caso, se selecciona un total de 90,000 instancias para cada etiqueta "Excelente", "Negativa", "Pesima" y "Positiva".

**Creación de un nuevo conjunto de datos equilibrado:** Las muestras seleccionadas aleatoriamente se combinan en un nuevo conjunto de datos llamado "combinado_df", donde cada etiqueta de sentimiento está igualmente representada. Este nuevo conjunto de datos está completamente aleatorizado para garantizar que no exista un sesgo en el orden de las instancias.

**Limpieza de HTML en las reseñas:** Se aplica una función que utiliza BeautifulSoup (biblioteca de Python) para limpiar posibles códigos HTML presentes en las reseñas, asegurando que solo se conserve el texto relevante y se elimine el marcado HTML.

9- **Selección del Modelo**

La selección del modelo para llevar a cabo el análisis de Social Media Listening se centró en la elección de la Regresión Logística debido a su idoneidad en tareas de procesamiento de lenguaje natural (PLN) y, específicamente, en la clasificación de sentimientos y temáticas en textos procedentes de distintos medios.

La Regresión Logística ha demostrado ser un modelo adecuado para este escenario por diversas razones:

**Capacidad de Clasificación Efectiva:**

La naturaleza clasificatoria de la Regresión Logística se adapta perfectamente al análisis de sentimientos, permitiendo distinguir y clasificar textos en categorías de polaridad (positiva, negativa, neutral) con precisión.

**Interpretación Sencilla:**

La Regresión Logística, a pesar de su simplicidad, es altamente interpretable. Esta cualidad es esencial en la comprensión de la lógica detrás de las predicciones, brindando transparencia en el proceso de toma de decisiones.

**Eficacia en Datos Textuales:**

El modelo ha demostrado ser eficaz en la clasificación de textos provenientes de diversos medios, debido a su capacidad para analizar y procesar datos textuales, especialmente luego de un exhaustivo preprocesamiento y transformación del texto en vectores numéricos.

**Buena Generalización:**

Ajustando los parámetros, especialmente el parámetro de regularización (C), la Regresión Logística es capaz de generalizar bien a nuevos datos. Esto es fundamental, considerando la actualización constante de los medios y la necesidad de adaptarse a cambios en el contenido.

**Prediccion:** 

La Regresión Logística, utilizada en este análisis de Social Media Listening, ofrece la capacidad de predecir con precisión la polaridad de los textos extraídos de los medios sociales. Esta capacidad de predicción permite discernir y clasificar los textos en categorías como positiva, negativa o neutral, proporcionando una comprensión profunda de la actitud o sentimiento expresado en el contenido de los medios sociales.

10- **PREPARACION Y ENTRENAMIENTO DEL MODELO**

**División del conjunto de datos:**

- **Propósito:** Dividir los datos en conjuntos de entrenamiento y prueba.
- **Uso de la función train_test_split():** Esta función de la biblioteca sklearn se utiliza para dividir el conjunto de datos en dos partes: una para entrenar el modelo y otra para probar su desempeño.
- **Razón:** Al entrenar el modelo con un conjunto de datos y probarlo con otro independiente, podemos evaluar su rendimiento en datos no vistos y verificar su capacidad para generalizar.

**Configuración de Herramientas de Procesamiento de Lenguaje Natural (NLP):**

- **Propósito:** Preparar el texto para su procesamiento.
- **Uso de nltk (Natural Language Toolkit) y funciones de preprocesamiento:**
- **Instalación de nltk: Permite acceder a herramientas para el procesamiento de texto.** Descarga de recursos como tokenizadores y palabras vacías: Los tokenizadores dividen el texto en tokens (palabras o frases), y las palabras vacías son palabras comunes que generalmente no contribuyen al significado (como "y", "el", "un", etc.).
- **Funciones para preprocesamiento de texto:** Incluyen tokenización, eliminación de palabras vacías, lematización y uso de un tokenizador porter para devolver las palabras a su forma raíz.
- **Razón:** Estas operaciones de preprocesamiento son fundamentales para normalizar el texto y prepararlo para su posterior análisis. Eliminan el ruido y simplifican el texto para que el modelo pueda interpretarlo mejor.

**Vectorización de texto:**

- **Propósito:** Convertir el texto en características numéricas para su procesamiento en modelos de aprendizaje automático.
- **Uso de TfidfVectorizer:** Transforma el texto en una representación numérica utilizando la frecuencia de términos (TF) e inversa frecuencia del documento (IDF).
- **Razón:** Los modelos de aprendizaje automático requieren datos numéricos para su funcionamiento. La vectorización de texto es crucial para representar las palabras y oraciones en un formato que los modelos puedan comprender.

**Ajuste de un Modelo:**

- **Propósito:** Entrenar un modelo para clasificar las reseñas de productos en diferentes categorías de sentimientos.
- **Uso de Logistic Regression y GridSearchCV:**
Logistic Regression es un modelo de clasificación lineal.
GridSearchCV busca los mejores hiperparámetros para el modelo, como el valor de regularización.
- **Razón:** Se utiliza para encontrar el modelo que mejor se ajuste a los datos y optimice la capacidad del modelo para predecir los sentimientos de las reseñas de productos.

**Resultados y Evaluación del Modelo:**

- **Propósito:** Comprender el rendimiento del modelo entrenado.
Presentación de los mejores parámetros y métricas de precisión: Se muestran los hiperparámetros óptimos y métricas de evaluación como precisión, recall y F1-score.
- **Razón:** Evaluar la eficacia del modelo y comprender su capacidad para predecir correctamente los sentimientos de las reseñas.

**Prueba con Ejemplos de Oraciones en Inglés:**

- **Propósito:** Verificar la capacidad del modelo para clasificar correctamente nuevas entradas de texto.
- **Razón:** Evaluar cómo el modelo clasifica y responde a ejemplos de oraciones en inglés no utilizadas durante el entrenamiento, lo que muestra su capacidad para generalizar a datos no vistos.

**Almacenamiento del Mejor Modelo Entrenado**

Una etapa crucial en el proceso de modelado es preservar el mejor estimador derivado de la optimización del modelo. En nuestro caso, tras realizar la búsqueda de hiperparámetros mediante la técnica de Grid Search en un modelo de regresión logística con regularización L1 y TF-IDF (Term Frequency-Inverse Document Frequency), se ha identificado el modelo que ofrece el mejor rendimiento en base a las métricas predeterminadas.

Para garantizar la conservación de este modelo optimizado, hemos empleado la biblioteca Joblib en Python. Utilizamos la función joblib.dump para guardar este estimador seleccionado en un archivo específico. El archivo generado, denominado 'Regresion_en_l1.pkl', contiene todos los parámetros, la estructura y la configuración de este modelo que sobresale en términos de desempeño y exactitud.

**Carga del Modelo Guardado**

Este archivo, 'Regresion_en_l1.pkl', facilita la recuperación del modelo completo sin necesidad de reentrenamiento. La carga del modelo guardado se logra mediante la función joblib.load, permitiéndonos utilizar directamente el modelo preservado para predecir etiquetas de sentimiento en nuevos datos.

El procedimiento para cargar y emplear el modelo guardado implica utilizar el método predict sobre los nuevos datos de interés, obteniendo así las predicciones basadas en el modelo sin la necesidad de repetir el proceso de entrenamiento.

Este enfoque nos brinda la capacidad de aplicar directamente el modelo optimizado en datos nuevos sin la fase de entrenamiento, lo cual resulta fundamental para la eficiencia en la implementación y la predicción precisa en aplicaciones del mundo real.


11- **Análisis de Sentimientos sobre el Dataset de Tweets de la Marca H**

- Con el propósito de comprender la percepción del público sobre la marca H, hemos realizado un análisis de sentimientos sobre un conjunto de datos que incluye tweets relacionados con dicha marca. El enfoque adoptado para este análisis se basa en la utilización de un modelo previamente entrenado que ha demostrado ser efectivo en la identificación de la polaridad de opiniones en texto.

- El proceso se inicia con la carga del modelo de clasificación previamente entrenado, el cual se ha construido mediante un algoritmo de regresión logística con regularización L1. Este modelo se recupera del archivo 'Regresion_en_l1.pkl' mediante el uso de la biblioteca Joblib en Python.

- A continuación, se procede a la definición de una función que permitirá aplicar este modelo al conjunto de datos de tweets relacionados con la marca H. Esta función ha sido diseñada para predecir la polaridad del sentimiento en los textos proporcionados, otorgando etiquetas como "Excelente" o "Pésima" a cada tweet en función de su contenido.

- El dataset, que contiene información relevante como el ID del usuario, nombre, edad, ubicación y, lo más importante, los tweets, se carga desde el archivo 'H.csv'. Posteriormente, se realiza una limpieza inicial del conjunto de datos, eliminando columnas irrelevantes para nuestro análisis, como es el caso de 'Unnamed: 5'.

- El modelo de clasificación es aplicado a los tweets mediante la función definida, etiquetando cada uno con un sentimiento específico según su contenido. En el proceso de etiquetado, se ha identificado una advertencia relacionada con la consistencia de los stopwords utilizados para el preprocesamiento del texto, lo cual merece una revisión adicional para asegurar la coherencia en los resultados obtenidos.

- El resultado de esta aplicación del modelo muestra una nueva columna, 'Sentimiento_Marca', que refleja la evaluación del sentimiento asociado a cada tweet. Cada fila del conjunto de datos ahora incluye una etiqueta que describe el sentimiento expresado en el tweet, desde opiniones "Excelentes" hasta aquellas que son consideradas "Pésimas".

- Este análisis de sentimientos proporciona una visión general sobre la percepción de la marca H en el dominio de las redes sociales, permitiendo identificar tendencias y valoraciones predominantes entre los usuarios, lo cual resulta fundamental para la comprensión de la reputación de la marca en el mercado digital.

12- **Análisis de Sentimientos en Tweets Relacionados con la Marca H a Nivel Nacional**

El análisis se enfocó en comprender la percepción de los usuarios en relación a la marca H, examinando una colección de Tweets de alcance nacional. Los resultados, detallados a continuación, revelan aspectos clave sobre los sentimientos expresados en esta plataforma.

**Distribución de Sentimientos**

**Análisis Cuantitativo de Sentimientos:**
- A través de gráficos, se evidenció que más del 51% de los Tweets evaluados sobre la marca H se catalogan como "Excelentes".

**Geolocalización y Análisis Demográfico**
  
**Regionalización de Tweets:**

- Se llevó a cabo un análisis geoespacial para determinar la distribución de los Tweets por provincias, con el fin de identificar regiones en las que la marca es más mencionada.

**Perfil de Usuarios por Edad:**

- Se priorizó la evaluación de la participación de usuarios de 31 años, detectando una representación significativa en la emisión de Tweets relacionados con la marca H. Este análisis desglosó los sentimientos asociados con esta cohorte, destacando la recepción positiva de la marca entre estos usuarios.

**Análisis Temático**

**Temas Latentes en Tweets Excelentes:**

- A través de la técnica de LatentDirichletAllocation, se extrajeron tópicos predominantes de los Tweets catalogados como "Excelentes". Los resultados se tradujeron en gráficos de nube de palabras, visualizando los temas más recurrentes.

**Temas en Tweets Negativos:**

- De manera similar, se examinaron y presentaron los tópicos sobresalientes extraídos de los Tweets con calificaciones negativas, generando representaciones visuales mediante gráficos de nube de palabras.

**Análisis de Temas en Tweets Positivos y Pesimos:**

- Se procedió a analizar los Tweets considerados "Positivos" y "Pesimos", identificando y presentando los temas predominantes a través de gráficos de nube de palabras.

13- **Informe sobre el Costo Computacional de la Ejecución:**

Para medir el costo computacional de un segmento específico de código, se utilizó una metodología para capturar el tiempo de inicio y finalización de su ejecución. Esta metodología es esencial para evaluar la eficiencia y el tiempo requerido para llevar a cabo operaciones computacionales.
El código implementado comienza con la obtención del tiempo en el inicio de la ejecución, utilizando la función time.time() de la librería time. A continuación, tras la ejecución del bloque de código bajo medición, se registra el tiempo nuevamente al finalizar su procesamiento.
El tiempo transcurrido se calcula restando el tiempo final al tiempo de inicio, lo que resulta en la duración total de la ejecución del código.
La duración obtenida se convierte a una forma legible, expresada en horas, minutos y segundos, empleando la función divmod para realizar las divisiones y obtener los valores respectivos.
El tiempo total de ejecución se presenta en un formato comprensible en el que se indican horas, minutos y segundos con dos decimales, aportando información clara acerca del costo computacional del bloque de código evaluado.
En el caso específico analizado, el tiempo transcurrido fue de 2 horas, 41 minutos y 46.21 segundos. Este enfoque de medición de tiempo resulta fundamental para evaluar la eficiencia y la carga computacional de segmentos específicos de código.

14- **CONCLUSIONES**

El análisis detallado de los posteos reveló que más del 51% de las publicaciones asociadas con la marca 'H' en Twitter reflejaban sentimientos positivos, mientras que aproximadamente un 30% se clasificaron como negativos. Estos datos demuestran una prevalencia notable de opiniones favorables hacia la marca, lo que podría sugerir una percepción general positiva entre los usuarios de la plataforma.
Además, al analizar los posteos por ubicación geográfica, se identificó una distribución dispar en la percepción de la marca. Por ejemplo, se observó una alta prevalencia de sentimientos excelentes en provincias como La Pampa, Mendoza y Jujuy. También se notó que los usuarios de 31 años realizaron la mayor cantidad de publicaciones, y la mayoría de ellas reflejaban una percepción positiva hacia la marca 'H'.
Los análisis temáticos a través de modelos de aprendizaje no supervisado revelaron tópicos comunes en los posteos con sentimientos positivos, como referencias a la calidad de los productos, la facilidad de uso y la satisfacción en la experiencia culinaria. Mientras que en los posteos con sentimientos negativos, se mencionaron temas como problemas con la durabilidad, la percepción de sobrevaloración y descontento general en la experiencia de uso.
En conclusión, estos hallazgos proporcionan una visión integral de la percepción de la marca 'H', lo que puede ser de gran utilidad para futuras estrategias de marketing y mejora de productos para satisfacer las necesidades y expectativas de los usuarios.

15- **OBJETIVOS ALCANZADOS**

Los objetivos alcanzados en el análisis de sentimientos sobre la marca 'H' a través de los posteos son los siguientes:

- **Clasificación de Sentimientos:** El análisis permitió clasificar los sentimientos expresados en los posteos, identificando la prevalencia de opiniones positivas, negativas y neutrales relacionadas con la marca 'H'. Se logró cuantificar y visualizar la distribución de estas opiniones.
- **Análisis Geográfico:** Se examinaron los sentimientos asociados con la marca 'H' por ubicación geográfica, identificando variaciones en la percepción de la marca en diferentes provincias.
- **Análisis por Edad:** Se estudiaron los sentimientos expresados por diferentes grupos de edad, lo que reveló la predominancia de publicaciones de usuarios de 31 años y la inclinación general de estos usuarios hacia opiniones positivas.
- **Modelado de Temas:** Se aplicaron técnicas de aprendizaje no supervisado para extraer temas comunes de los posteos con sentimientos positivos y negativos, lo que proporcionó una comprensión más profunda de los temas relevantes y las preocupaciones de los usuarios.
- **Aportes para la Estrategia de Marketing:** Los resultados del análisis brindan información valiosa que podría guiar estrategias de marketing y mejoras de productos para alinearlos con las expectativas y preferencias de los usuarios.

En resumen, el análisis permitió comprender la percepción general de la marca 'H' en la plataforma de Twitter, identificando patrones y áreas de oportunidad que podrían aprovecharse para fortalecer la imagen de la marca y mejorar su relación con los consumidores.

16- **PROBLEMAS ALCANZADOS DEL ANALISIS**

A lo largo del análisis de sentimientos sobre la marca 'H' en los posteos, se enfrentaron varios desafíos y obstáculos. Algunos de los problemas alcanzados incluyen:

- **Limpieza de Datos:** El proceso de limpieza de datos involucró la eliminación de registros duplicados, valores nulos y caracteres especiales, lo que a menudo requería tiempo y esfuerzo para garantizar la calidad de los datos.
- **Variabilidad de Opiniones:** La variedad de opiniones expresadas en los posteos dificultó la tarea de clasificar los sentimientos de manera precisa, ya que algunas publicaciones eran ambiguas o sarcásticas.
- **Desbalance en los Datos:** La cantidad de posteos con opiniones positivas era significativamente mayor que los negativos o neutrales, lo que llevó a un desbalance en los datos y requirió considerar estrategias para abordar este problema.
- **Tamaño del Conjunto de Datos:** La cantidad de posteos recopilados podría no ser representativa de la totalidad de la conversación en línea sobre la marca 'H', lo que limita la generalización de los resultados.
- **Interpretación de Temas:** La identificación de temas en los posteos requirió técnicas de aprendizaje no supervisado, lo que implicó cierta subjetividad en la interpretación de los resultados.

A pesar de estos desafíos, se lograron los objetivos del análisis y se proporcionaron conclusiones significativas que pueden beneficiar a la marca 'H' en términos de comprender la percepción del público y guiar estrategias futuras.

17- **TRABAJO A FUTURO**

- *Refinamiento de Modelos:** Se puede explorar el uso de técnicas más avanzadas de procesamiento de lenguaje natural (NLP) y modelos de aprendizaje automático más complejos para mejorar la precisión en la clasificación de sentimientos.

- *Inclusión de Modelos Contextuales:** La adopción de modelos pre-entrenados de lenguaje como BERT, GPT, o XLNet podría mejorar el entendimiento del contexto y las sutilezas lingüísticas en los Tweets.

- *Aumento de Datos:** La adquisición de un conjunto de datos más grande y equilibrado con opiniones variadas permitiría mejorar la capacidad del modelo para generalizar y clasificar sentimientos con mayor precisión.

- *Análisis Temporal:** Considerar la variación de sentimientos a lo largo del tiempo podría proporcionar una visión más profunda de la evolución de la percepción de la marca 'H' en respuesta a diferentes eventos o campañas.

- *Optimización del Rendimiento:** Dependiendo del tamaño del conjunto de datos, se podría trabajar en la optimización de las operaciones, especialmente al cargar y manipular conjuntos de datos grandes.

***Estas mejoras y consideraciones futuras pueden potenciar la capacidad del algoritmo para analizar sentimientos con mayor precisión y profundidad, proporcionando información valiosa para la marca 'H'.***

