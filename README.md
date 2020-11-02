# neural_network
Implementación de una red neuronal con modelamiento matricial

El código fue hecho en Spyder y Python 3.7, por lo que son recomendados para usar el programa correctamente. Para ejecutar el código, ejecutar el archivo "neural_network_matrix.py". El entrenamiento puede demorar hasta 10 minutos, pero se puede ajustar según número de epochs. Sin embargo, se recomienda dejar en un número bastante alto, o hasta que el error cuadrático medio rondee 0.3. Los archivos dentro de la carpeta fails no tienen ninguna utilidad, solo son la muestra del fracaso en los primeros intentos de implementación xd. 

En esta implementación se utilizó un modelamiento tipo matricial para representar las capas y los pesos de la red, aprovechando la eficiencia de numpy para operaciones matriciales. El dataset que se utilizó es un dataset de ropa clasificada por su tipo, de un tamaño de sobre 30.000 instancias de 784 atributos relacionados a la imagen y un atributo llamado label que representa el tipo de ropa de la imagen. La red funciona en términos generales de la siguiente manera:

1) Entrenamiento con un una porción del dataset: 

    a) Se normaliza la data(todo el dataset) y se separa los datos de la etiqueta
    
    b) Se crean dos matrices de pesos de tamaños (784, 20) y (20,10). La razón de estos números es, en el caso del 784, la cantidad de atributos que tiene una instancia de entrenamiento, 20 es un número arbitrario entre el número de atributos y el número de outputs (10), se elige este número relativamente bajo porque la complejidad de las operaciones aumenta considerablemente con números más altos. 10 Es el número de outputs, es decir el vector con el resultado final. Ya que hay 10 clases distintas de clasificación, el vector que arrojará es un vector de tamaño 10 con números entre 0 y 1, que gracias a la función softmax suman 1, dándole sentido probabilístico a la predicción.
    
    c) Se pasa una instancia aleatoria a la red un número arbitrario de veces (por defecto 5000). Esta instancia es pasada entre capas mediante operaciones de multiplicación entre los valores (o valores activados por la función sigmoide en capas intermedias) por sus pesos correspondientes, hasta llegar a la capa final o de output, que es activado por la función softmax. * Normalmente cada capa, excepto la última, tienen un valor adicional llamado "bias" que se agrega, pero en este caso los mejores resultados se consiguieron sin el bias, ya que probablemente hubo un error de implementación del delta de bias.
    
    d) Una vez que se llega al final, se calcula el error cuadrático medio entre la solución esperada y la solución arrojada, y se compara también el vector de salida con el vector esperado, para utilizar la diferencia y propagarla hacia la red, actualizando la matriz de pesos de manera acorde al error que tuvo cada elemento de esta. Las operaciones acá utilizan álgebra lineal y cálculo, ya que se requieren las derivadas y los deltas para calcularlos. 
    
    e) Se realiza esto un número arbitrario de epochs o épocas, tendiendo a disminuir el error cuadrático medio a medida que se van ajustando los pesos mientras la red "aprende" de los valores dados.
    
 2) Cálculo de precisión con el resto del dataset:
 
    a) Con las instancias sobrantes que no ha visto la red previamente se intenta predecir qué tipo de ropa es, para esto, el vector que arroja la predicción de una instancia busca el valor máximo en esta (el resultado que tendría más probabilidades de ser el correcto según la red) y lo compara con el vector de output esperado, retornando 1 si es correcto y 0 si es incorrecto.
    
    b) Una vez que itera por todo el dataset, calculamos la precisión promedio con la división entre el puntaje total de predicción (la cantidad de aciertos en el dataset) por sobre el número de instancias del dataset de testeo, entregando un número entre 0 y 1 que representa el porcentaje de precisión del algoritmo.
