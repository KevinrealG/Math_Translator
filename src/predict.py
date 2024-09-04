import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np
import pickle
import os
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model_1_de.h5')

# Modify the model to remove the time_major argument
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.LSTM):
        config = layer.get_config()
        if 'time_major' in config:
            del config['time_major']
        new_layer = tf.keras.layers.LSTM.from_config(config)
        model._layers[layer.name] = new_layer

# Save the modified model
model.save('model_1_de_fixed.h5')
# Oración de prueba
# Check current working directory
print("Current working directory:", os.getcwd())

# Oración de prueba
# Carga el modelo
try:
    model = tf.keras.models.load_model('model_1_de.h5')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the file 'model_1_de.h5' exists in the directory:", os.getcwd())
    raise
with open('tokenizer_es_def.pickle', 'rb') as handle:
    tokenizer_es = pickle.load(handle)
with open('tokenizer_mathml_def.pickle', 'rb') as handle:
    tokenizer_mathml = pickle.load(handle)
print("Loaded tokenizer from disk")
max_length_seq = 70


#test_sentences = ["<START> un número multiplicado por 4  <END>"]
def inferencia(sentence: str):
    """
    Función que realiza la inferencia de una oración en lenguaje natural a una ecuación en MathML.
    
    Args:
        sentence (str): Oración en lenguaje natural.
        
    Returns:
        str: Ecuación en MathML.
        
        """
    test_sentences = ["<START> "+sentence+" <END>"]
    # Tokeniza la oración de entrada
    test_sequences = tokenizer_es.texts_to_sequences(test_sentences)
    test_sequences_padded = pad_sequences(test_sequences, maxlen=max_length_seq, padding='post')
    # Inicializa la secuencia de salida con el token de inicio
    start_token = tokenizer_mathml.word_index['<start>']
    output_sequence = np.zeros((1, max_length_seq), dtype=int)
    output_sequence[0, 0] = start_token

    for i in range(1, max_length_seq):
        # no imprimir la ejecución del modelo predict
        #usa el modelo para predecir la distribución de probabilidad 
        # sobre el vocabulario para cada posición de la secuencia de salida.

        pred = model.predict([test_sequences_padded, output_sequence], verbose=0)
        
        #selecciona el token más probable para la posición actual i
        pred_token = np.argmax(pred[0],axis=-1)
        # agrega el token predicho a la secuencia de salida en la posición i.
        output_sequence[0, :i+1] = pred_token[:i+1]


        
        if pred_token[-1] == tokenizer_mathml.word_index['<end>'] :
            break
        

    # Convierte la secuencia predicha a texto
    predicted_output_text = []
    for token_index in output_sequence[0]:
        word = tokenizer_mathml.index_word.get(token_index, '')
        if word and word != '<start>' and word != '<end>':
            predicted_output_text.append(word)

    mathtml_equation = ' '.join(predicted_output_text)
    return mathtml_equation





