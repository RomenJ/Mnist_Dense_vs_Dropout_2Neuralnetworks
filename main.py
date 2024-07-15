# Importaciones necesarias para el proyecto
# model2 from :  https://www.youtube.com/watch?v=aFZEvQDTSyA&ab_channel=RingaTech
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuración del logger para evitar mensajes de advertencia innecesarios
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Cargar el conjunto de datos MNIST
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Nombres de las clases para la visualización
class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]

# Número de ejemplos en los conjuntos de entrenamiento y prueba
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# Función para normalizar las imágenes (de 0-255 a 0-1)
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# Aplicar la normalización a los conjuntos de datos
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Definición del modelo usando Keras (model2)
model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # Transformar la imagen en un vector
    tf.keras.layers.Dense(64, activation=tf.nn.relu), # Capa oculta con 64 neuronas y activación ReLU
    tf.keras.layers.Dense(64, activation=tf.nn.relu), # Otra capa oculta con 64 neuronas y activación ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Capa de salida con 10 neuronas y activación softmax para clasificación
])

# Compilación de model2
model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Definición del modelo usando Keras (model)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compilación del modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Configuración del entrenamiento por lotes
BATCHSIZE = 64
train_dataset = train_dataset.shuffle(num_train_examples).batch(BATCHSIZE).repeat()
test_dataset = test_dataset.batch(BATCHSIZE).repeat()

# Entrenamiento de model
history = model.fit(
    train_dataset, epochs=15,
    steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE),
    validation_data=test_dataset,
    validation_steps=math.ceil(num_test_examples/BATCHSIZE)
)

# Entrenamiento de model2
history2 = model2.fit(
    train_dataset, epochs=15,
    steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE),
    validation_data=test_dataset,
    validation_steps=math.ceil(num_test_examples/BATCHSIZE)
)

# Evaluación de model en el conjunto de prueba
eval_steps = math.ceil(num_test_examples / BATCHSIZE)
test_loss, test_accuracy = model.evaluate(test_dataset, steps=eval_steps)
print("Resultado en las pruebas de model: ", test_accuracy)

# Evaluación de model2 en el conjunto de prueba
test_loss2, test_accuracy2 = model2.evaluate(test_dataset, steps=eval_steps)
print("Resultado en las pruebas de model2: ", test_accuracy2)

# Gráfico de precisión y pérdida de ambos modelos
def plot_training_histories(history, history2):
    # Gráfico de precisión
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento model')
    plt.plot(history2.history['accuracy'], label='Precisión de entrenamiento model2')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Precisión durante el entrenamiento (Train)')
    
    # Gráfico de pérdida
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento model')
    plt.plot(history2.history['loss'], label='Pérdida de entrenamiento model2')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdida durante el entrenamiento (Train)')
    plt.savefig('Precision_y_perdida_Trainning_model_vs_model2.jpg')
    plt.show()

# Gráfico de precisión y pérdida de test
def plot_test_histories(history, history2):
    # Gráfico de precisión
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(history.history['val_accuracy'], label='Precisión de prueba model')
    plt.plot(history2.history['val_accuracy'], label='Precisión de prueba model2')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Precisión durante la prueba (Test)')
    
    # Gráfico de pérdida
    plt.subplot(2, 1, 2)
    plt.plot(history.history['val_loss'], label='Pérdida de prueba model')
    plt.plot(history2.history['val_loss'], label='Pérdida de prueba model2')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdida durante la prueba (Test)')
    plt.savefig('Precision_y_perdida_Test_model_vs_model2.jpg')
    plt.show()

plot_training_histories(history, history2)
plot_test_histories(history, history2)

# Generar predicciones sobre el conjunto de prueba para model
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    predictions2 = model2.predict(test_images)

# Función para visualizar una imagen con su predicción
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Prediccion: {} ({})".format(class_names[predicted_label], class_names[true_label]), color=color)

# Función para visualizar las probabilidades de predicción para una imagen
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#888888")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Visualización de algunas imágenes con sus predicciones para model
def plot_model_predictions(test_labels, test_images, predictions, model_name):
    numrows = 5
    numcols = 3
    numimages = numrows * numcols

    plt.figure(figsize=(2 * 2 * numcols, 2 * numrows))
    for i in range(numimages):
        plt.subplot(numrows, 2 * numcols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(numrows, 2 * numcols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.suptitle(f'Predicciones del {model_name}')
    plt.show()

# Mostrar predicciones de model
plot_model_predictions(test_labels, test_images, predictions, 'model')

# Mostrar predicciones de model2
plot_model_predictions(test_labels, test_images, predictions2, 'model2')

# Generar matriz de confusión para model
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de confusión para model')
plt.show()

# Generar matriz de confusión para model2
y_pred2 = np.argmax(predictions2, axis=1)
cm2 = confusion_matrix(test_labels, y_pred2)

plt.figure(figsize=(10, 8))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de confusión para model2')
plt.show()
