![](./references/branding/Recurso%201-.png)


Este repositorio es para explorar y desarrollar un modelo de inteligencia artificial
basado en aprendizaje supervisado para detecta sequías con suficiente tiempo de anticipación.

Organización del proyecto
------------

    ├── LICENSE
    ├── README.md          <- El README.md para el entendimiento del proyecto.
    │
    ├── config             <- Folder to save configuration files. 
    │
    ├── data
    │   ├── external       <- Datos de fuentes externas.
    │   ├── interim        <- Datos intermedios transformados.
    │   ├── processed      <- Datos canónicos para entrenamiento/evaluación del modelo.
    │   └── raw            <- Datos originales.
    │
    ├── models             <- Modelos entrenados y serializados. 
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── references         <- Material de apoyo para el proyecto en general. 
    │
    ├── relevant_documents <- Papers y documentos relevantes para el proyecto.
    │
    ├── reports            <- Analisis generados como PDF, LaTeX, etc.
    │    └── figures       <- Figuras y gráficos. 
    │
    ├── requirements.txt   <- Archivo de requerimientos para instalar dependencias, replicando el ambiente.
    │
    ├── setup.py           <- Hace el proyecto instalable por pip (pip install -e .), src puede ser importado.
    │
    └── src                <- Código fuente para este proyecto.
        ├── __init__.py    <- Hace a src un módulo de python
        │
        ├── data           <- Scripts para descargar datos.
        │
        ├── demo           <- Scripts para crear la demo/mvp.
        │
        ├── features       <- Scripts para procesar los datos y crear features.
        │
        ├── models         <- Scripts para entrenar modelos. 
        │
        └── visualization  <- Scripts para crear visualizaciones. 
---------
## Problemática

El calentamiento global está incrementando la incidencia e intensificación de las sequías, afectando directamente la producción de alimentos y la disponibilidad del agua para distintas actividades humanas debido a la escasez hídrica (IPCC, 2023; Means, 2023). Esto a su vez ocasiona que este fenómeno dure más tiempo y abarque territorios más amplios, causando una aridificación generalizada de los climas; es decir los climas humedos son menos húmedos y los secos son más secos (BBVA, 2023.a).  

En la actualidad se calcula que 55 millones de personas a nivel mundial se ven afectadas de forma directa cada año (WHO, 2023), y hay escenarios donde el 75% de la población podría verse afectada para el 2050 (BBVA, 2023.b).

El impacto de las sequías se manifiesta en tres dimensiones interconectadas: económica, ambiental y social. En el ámbito económico, las sequías provocan daños severos a la agricultura y la ganadería, ocasionando desequilibrios en los precios de los alimentos. Esto deriva, de forma directa, en inseguridad alimentaria y, de forma indirecta, en desempleo, pobreza e inflación. A nivel ambiental, las sequías generan un impacto considerable en los ecosistemas, afectando tanto a la flora como a la fauna. Además, provocan la pérdida de la calidad del suelo y el aumento de otros fenómenos de alto impacto, como los incendios forestales. Por último, las sequías tienen un profundo impacto social, reflejado en cambios en el estilo de vida debido a la escasez de agua. Además, ponen en riesgo la seguridad y la salud de las personas por la malnutrición y el aumento del riesgo de contraer enfermedades (National Drought Mitigation Center, 2023).

Si bien en México las medidas para enfrentar las sequías han sido principalmente reactivas (Arreguín-Cortés, et.al. 2016), como la distribución de agua potable mediante pipas o la creación de nuevos pozos (CONAGUA, 2014), se ha buscado avanzar hacia un enfoque preventivo. Para ello se crearon programas denominados Programas de Medidas Preventivas y de Mitigación de la Sequía (PMPMS), donde la Comisión Nacional del Agua (CONAGUA) juega un papel central a través del Programa Nacional Contra la Sequía (PRONACOSE) (Ortega-Gaucin, 2018). Los PMPMS permiten determinar los niveles de riesgo en cada sistema hídrico y priorizar acciones para prevenir el desarrollo o el empeoramiento de las sequías (CONAGUA, 2015). Sin embargo, a pesar de este avance, las acciones implementadas siguen teniendo un marcado carácter reactivo, con un enfoque predominante en lo inmediato.

Aunado a esto, se cuenta con el Monitor de Sequía en México (MSM), el cual permite determinar la presencia de sequía a nivel municipal con una actualización de cada 15 días. Esto, siendo un sistema de vigilancia, sigue teniendo enfoque reactivo pero con un alcance más cercano a lo preventivo ya que es muy parecido a tener la disponibilidad de información en tiempo real (solo con 15 días de atraso). 

## Solución

La implementación de un sistema completamente preventivo para la gestión de sequías requiere de mecanismos de predicción precisos. Es por esto que presentamos SeGuía, un modelo de inteligencia artificial basado en aprendizaje supervisado, que aprende de datos históricos para realizar predicciones de la presencia/ausencia de la sequía contemplando a su vez la severidad de las mismas (vease figura 1).

![Figura 1: Diagrama para ilustrar la generación de la Inteligencia Artificial: SeGuía
](./references/Seguia-modelo.png)


## Viabilidad – Factibilidad: 

El proyecto SeGuía presenta una alta viabilidad técnica, sustentada en los siguientes aspectos:

- Disponibilidad de datos: Se cuenta con información histórica proporcionada por el MSM como base para el entrenamiento del modelo. Además, se contempla la integración de datos abiertos de diversas fuentes, incluyendo variables climatológicas (Administración Nacional de Aeronáutica y el Espacio [NASA]), geológicas (La Comisión Nacional para el Conocimiento y Uso de la Biodiversidad [CONABIO]) y demográficas (datos abiertos de México), para enriquecer las predicciones.
- Recursos tecnológicos: La tecnología actual permite la creación e implementación de modelos predictivos de manera sencilla. Se utilizará Python para el desarrollo del modelo, un lenguaje de programación ampliamente utilizado y con amplia comunidad de desarrolladores.
- Capacidad del equipo: El equipo cuenta con la experiencia y las habilidades necesarias para desarrollar el sistema. Se tiene la capacidad de abordar la solución de manera programática y de integrar las diferentes fuentes de datos.

### Consideraciones sobre la disponibilidad de datos:

Si bien la disponibilidad de datos es un punto fuerte del proyecto, es importante reconocer que la cantidad y calidad de los datos disponibles podrían ser un factor limitante para la precisión de las predicciones. En este sentido, el equipo de SeGuía se compromete a:
- Implementar técnicas de aprendizaje automático que sean robustas a la falta de datos.
- Explorar estrategias para la obtención de datos adicionales de alta calidad.
- Evaluar continuamente el rendimiento del modelo y realizar ajustes según sea necesario. 
 
### Viabilidad legal:
  No se prevén barreras legales significativas para el desarrollo del proyecto SeGuía, dado que:
  - Los datos utilizados son de carácter abierto: Se utilizarán datos abiertos provenientes de fuentes confiables como la NASA, CONABIO y datos abiertos de México. Estos datos no contienen información sensible o personal.
  - Los datos se encuentran agregados a nivel municipal: La agregación de datos a nivel municipal garantiza la protección de la privacidad individual y evita la identificación de personas o grupos específicos.

  En resumen, el proyecto SeGuía presenta una alta viabilidad, lo que lo convierte en una propuesta factible para su implementación. A pesar de los riesgos potenciales, el equipo confía en que el proyecto SeGuía puede ser una herramienta valiosa para la gestión de sequías en México.
  
## Innovación – Disrupción: 
  El proyecto SeGuía se presenta como una solución innovadora y disruptiva en el ámbito de la gestión de sequías, ofreciendo las siguientes ventajas:
  - Predicción anticipada de sequías: A diferencia de los enfoques reactivos tradicionales, SeGuía permitirá predecir la presencia y severidad de las sequías con anticipación, posibilitando la toma de medidas preventivas y la reducción de su impacto.
  - Cuantificación del error de predicción: El modelo de SeGuía no solo predicirá la ocurrencia de sequías, sino que también cuantifica el error de la predicción. Esta información es vital para evaluar la confiabilidad de las predicciones y tomar decisiones informadas.
  - Mejora de las prácticas actuales: La implementación de SeGuía puede transformar las prácticas actuales de gestión de sequías, proporcionando una herramienta accesible y efectiva para diversos actores, desde agricultores hasta organismos gubernamentales.
  - Accesibilidad y escalabilidad: SeGuía está diseñado para ser una herramienta accesible y escalable, permitiendo su implementación en diferentes contextos y regiones.
  - Impacto potencial: SeGuía tiene el potencial de generar un impacto significativo en la reducción de las pérdidas económicas, sociales y ambientales asociadas a las sequías.
  Sin embargo, es importante reconocer que el desarrollo de modelos de predicción conlleva ciertos desafíos:
  - Disponibilidad y calidad de datos: La precisión de las predicciones del modelo depende en gran medida de la calidad y disponibilidad de los datos utilizados para su entrenamiento. Es importante contar con datos confiables y representativos para garantizar la efectividad del modelo.
  - Sesgos en el modelo: Los modelos de aprendizaje automático pueden presentar sesgos, lo que significa que pueden generar predicciones discriminatorias o injustas. Es importante implementar técnicas de explicabilidad para identificar y mitigar los sesgos en el modelo.
  - Interpretabilidad del modelo: La complejidad de los modelos de aprendizaje automático puede dificultar su interpretación, lo que puede generar dudas sobre la confiabilidad de las predicciones. Es necesario desarrollar estrategias para comunicar de manera efectiva el funcionamiento del modelo.
  Es posible mitigar los desafíos antes descritos, por lo cual esta herramienta tiene potencial para transformar la gestión de sequías.
## Escalabilidad – Ampliación del Potencial: 
  SeGuía no solo se presenta como una herramienta viable y factible para la gestión de sequías en México, sino que también poseerá un alto potencial de escalabilidad y adaptación, permitiendo su implementación a nivel nacional e incluso internacional. A continuación, se detallan los aspectos clave:
- Escalabilidad nacional:
  - Cobertura nacional: SeGuía estará diseñado para ser aplicado en todo el territorio nacional, abarcando las diversas regiones y climas que conforman el país.
  - Adaptación a diferentes contextos: El modelo puede ser adaptado a las características específicas de cada región, tomando en cuenta las variables climáticas, geológicas y demográficas particulares.
  - Integración con sistemas existentes: SeGuía puede integrarse con sistemas de alerta temprana y planes de gestión de riesgos existentes, fortaleciendo la capacidad de respuesta ante las sequías a nivel nacional.
  - Mejora en el alcance:  Al ser un modelo de uso general respecto a la predicción de sequías, es posible que en fases avanzadas el proyecto pueda generar pronósticos con diferentes horizontes temporales. De esta manera, los usuarios pueden obtener predicciones anticipadas de 3 meses o 6 meses, según sus requerimientos particulares.
- Potencial para la aplicación internacional:
  - Adaptación a otros países: El modelo SeGuía puede ser adaptado a otros países siempre y cuando se cuenten con datos históricos relevantes o se realice una evaluación y validación de su desempeño en esas geografías.
  - Requisitos de datos: Para realizar la inferencia en otros países, se requiere información previa sobre las variables climáticas, geológicas y demográficas del territorio.
  - Evaluación y validación: En caso de no contar con datos históricos de sequías, es posible realizar una evaluación y validación del modelo utilizando únicamente información climatológica, geológica y demográfica.
  - Flexibilidad en la predicción: SeGuía puede ser configurado para realizar predicciones con diferentes horizontes temporales, adaptándose a las necesidades específicas de cada contexto.

SeGuía tiene el potencial de ser escalable y adaptable para transformar la gestión de sequías a nivel nacional e internacional. Su capacidad para predecir sequías con anticipación, cuantificar el error de predicción, mejorar las prácticas actuales y adaptarse a diversos contextos la convierte en una propuesta con un impacto significativo en la reducción de las pérdidas asociadas a las sequías.

### Modelo de Negocio 
El modelo de negocio para SeGuía pretender formar alianzas con entidades públicas y privadas. 
- Alianzas con gobiernos:
  - Colaboración con agencias gubernamentales
  - Participación en programas gubernamentales
  - Financiamiento y apoyo a la investigación
Se busca colaborar con entidades gubernamentales para obtener financiamiento para investigación y desarrollo continuo de SeGuía, así como para estudios sobre el impacto de las sequías en México.
- Colaboración con organizaciones internacionales:
  - Participación en proyectos internacionales
  - Difusión y sensibilización sobre las sequías
Colaborar con organizaciones internacionales para difundir información sobre sequías, sus impactos y medidas de mitigación, promoviendo el uso responsable del agua. Así mismo participar en proyectos de investigación y desarrollo financiados por organizaciones internacionales, intercambio de conocimientos con expertos globales en gestión de sequías.
Mecanismos para generar ingresos con el modelo de negocio de SeGuía:
- Financiamiento para la implementación:
  - Solicitar financiamiento a través de fondos destinados a proyectos de innovación tecnológica y gestión del agua.
  - Buscar apoyo de inversionistas interesados en proyectos con impacto social y ambiental positivo.
  - Establecer acuerdos con entidades públicas para la implementación conjunta de SeGuía, a cambio de una tarifa.
- Monetización de datos:
  - Crear APIs que permitan a otras empresas acceder a datos y funcionalidades de SeGuía por una tarifa.
  - Servicios de análisis de datos: Ofrecer análisis de datos personalizados a entidades públicas y privadas.
- Licenciamiento de la tecnología:
  - Licenciar la tecnología de SeGuía para su implementación en otros proyectos.


### Impacto

SeGuía se presenta como una herramienta innovadora y disruptiva con el potencial de transformar la gestión de sequías en México y contribuir a un futuro más resiliente. Su capacidad la convierte en una propuesta viable, factible y escalable, con un impacto significativo en los ámbitos económico, social y ambiental.
A pesar de los desafíos potenciales, el equipo de SeGuía está comprometido a implementar estrategias para mitigarlos y garantizar la confiabilidad y la equidad de las predicciones.
La implementación exitosa de SeGuía requiere de un compromiso continuo con la investigación y el desarrollo, la colaboración entre diferentes sectores y la creación de políticas públicas que fomenten la adopción de esta herramienta innovadora. SeGuía representa una oportunidad única para transformar la forma en que México enfrenta las sequías y construir un futuro con agua para las generaciones venideras.

### Referencias:
1) IPCC, 2023: Sections. In: Climate Change 2023: Synthesis Report. Contribution of Working Groups I, II and III to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Core Writing Team, H. Lee and J. Romero (eds.)]. IPCC, Geneva, Switzerland, pp. 35-115, doi: 10.59327/IPCC/AR6-9789291691647 [Recuperado el 23 de junio del 2024 de: https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf]
2) Means, T. (2023) La conexión entre el cambio climático y las sequias. Yale Climate Connections. [Recuperado el 23 de junio del 2024 de https://yaleclimateconnections.org/2023/05/la-conexion-entre-el-cambio-climatico-y-las-sequias/]
3) BBVA (2023.a) Del cambio climático a la mala gestión del agua: causas y consecuencias de la sequía. BBVA Sostenibilidad. [Recuperado el 23 de juno del 2024 de: https://www.bbva.com/es/sostenibilidad/del-cambio-climatico-a-la-mala-gestion-del-agua-causas-y-consecuencias-de-la-sequia/]
4) World Health Organization (2023) Drought. [Recuperado el 23 de junio del 2024 de: https://www.who.int/health-topics/drought]5)
5) BBVA (2023.b) Cuando Falta el agua. Monografias sostenibilidad. [Recuperado el 23 de junio del 2024 de: https://www.bbva.com/wp-content/uploads/2023/07/monografico-sequia.pdf]
6) National Drought Mitigation Center (2023). How does drought affect our lives? [Recuperado el 23 de junio del 2024 de: https://drought.unl.edu/Education/DroughtforKids/DroughtEffects.aspx]
7) Arreguín-Cortés, F., López-Pérez, M., Ortega-Gaucin, D. y Ibañez-Hernández O. (2016) La política pública contra la sequía en México: avances, necesidades y perspectivas. Tecnología y Ciencias del Agua, 7(5): 63-76. [Recuperado el 23 de junio del 2024 de: http://repositorio.imta.mx/handle/20.500.12013/1705 ]
8) CONAGUA (2014) Política Pública Nacional para la Sequía Documento rector. [Recuperado el 23 de junio del 2024 de: https://www.conagua.gob.mx/CONAGUA07/Contenido/Documentos/Pol%C3%ADtica%20P%C3%BAblica%20Nacional%20para%20la%20Sequ%C3%ADa%20Documento%20Rector.pdf]
9) Ortega-Gaucin, D. (2018) Medidas para afrontar la sequía en México: Una visión retrospectiva. Revista Col. San Luis vol.8 no.15 San Luis Potosí ene./abr. 2018. [Recuperado el 23 de junio del 2024 de: https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1665-899X2018000100077]
10) CONAGUA (2015) PMPMS Para usuarios urbanos de agua potable y saneamiento. Consejo de cuencia del río santiago. [Recuperado el 23 de junio del 2024 de: https://www.gob.mx/cms/uploads/attachment/file/99855/PMPMS_ZM_Guadalajara_Jal.pdf]

## Información de Sequías:

La información de sequías fue obtenida de la [Conagua](https://smn.conagua.gob.mx/es/climatologia/monitor-de-sequia/monitor-de-sequia-en-mexico)
utilizando el 
[`Monitor de Sequía de México (MSM)`](https://smn.conagua.gob.mx/es/climatologia/monitor-de-sequia/monitor-de-sequia-en-mexico). 

![](https://smn.conagua.gob.mx/tools/DATA/Climatolog%C3%ADa/Sequ%C3%ADa/Monitor%20de%20sequ%C3%ADa%20en%20M%C3%A9xico/Seguimiento%20de%20Sequ%C3%ADa/MSM20231130.png)
La información corresponde a los municipios de México que han sido afectados por una de las siguientes condiciones:

- NaN: No se registró sequía
- D0: Anormalmente seco
- D1: Sequía moderada
- D2: Sequía severa
- D3: Sequía extrema
- D4: Sequía excepcional

La información relevante sobre sequías proviene después de 2016 por dos factores principales.
El primer factor es que, desde 2014, los datos adquirieron un alcance nacional, donde la información se obtuvo de manera quincenal (cada 15 días). El segundo factor es que en 2016 los criterios de asignación del índice de sequía cambiaron drásticamente. Para más información, visite el siguiente
[enlace](https://smn.conagua.gob.mx/es/climatologia/monitor-de-sequia/monitor-de-sequia-en-mexico).

## Información Geoespacial

El archivo de forma de los municipios fue obtenido de 
[CONABIO](http://geoportal.conabio.gob.mx/metadatos/doc/html/muni_2018gw.html)
utilizando los municipios a principios de 2018. Corresponde a los municipios de cada estado federal. La información denota los límites geoestadísticos de cada municipio, con una resolución de 1:250000. Cada municipio contiene un código único compuesto por tres dígitos, comenzando con 001 para cada entidad.

## Información Meteorológica

La información meteorológica fue de acceso libre de la NASA. 

------







![](./references/branding/Logo%20seguia.png)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
