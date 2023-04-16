import numpy as np
import keras
import librosa
import tensorflow as tf

def build_wavenet_model(input_shape, num_filters, num_layers, num_blocks, kernel_size):
    inputs = keras.Input(input_shape)

    x = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal')(inputs)
    skip_connections = []
    # Add the residual blocks
    for i in range(num_blocks):
        # Add the dilation layers
        for j in range(num_layers):
            dilation_rate = 2 ** j
            x1 = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
            x2 = keras.layers.Conv1D(filters=num_filters, kernel_size=1)(x1)
            x3 = keras.layers.Conv1D(filters=num_filters, kernel_size=1)(x)
            x4 = keras.layers.Add()([x2, x3])
            x5 = keras.layers.Activation('relu')(x4)
            x6 = keras.layers.Conv1D(filters=num_filters, kernel_size=1)(x5)
            x7 = keras.layers.Multiply()([x, x6])
            x = keras.layers.Add()([x, x7])

        # Add the skip connection
        skip = keras.layers.Conv1D(filters=num_filters, kernel_size=1)(x)
        skip = keras.layers.Activation('relu')(skip)
        skip_connections.append(skip)

    # Add the output layers
    x = keras.layers.Add()(skip_connections)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv1D(filters=num_filters, kernel_size=1)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1)(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model

def u_law(x):
    return np.sign(x)* np.log(1+255*np.abs(x))/np.log(1+255)
def read_audio(path):
    audio, sr = librosa.load(path)
    audio = audio / np.abs(audio).max()
    z = u_law(audio)
    print(z[:100])
    audio = np.asarray( (z+1)/2 * 255, dtype=int)
    print(audio[:100],audio.max())
    num_classes = 256
    audio_data_categorical = keras.utils.to_categorical((audio + 1) , num_classes=num_classes)

    return audio_data_categorical

audio_data_categorical = read_audio("data/20230415 Edit 1 Export 2.wav")
input_shape = (None, audio_data_categorical.shape[-1])
num_filters = 128
num_layers = 10
num_blocks = 3
kernel_size = 2
model = build_wavenet_model(input_shape, num_filters, num_layers, num_blocks, kernel_size)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001))

# Train the model
model.fit(audio_data_categorical[:-1], audio_data_categorical[1:], batch_size=256, epochs=100)

# Generate new audio
seed = np.zeros((1, 1, audio_data_categorical.shape[-1]))
for i in range(1000):
    pred = model.predict(seed)
    new_sample = np.argmax(pred[:, -1, :], axis=-1)
    new_sample_categorical = keras.utils.to_categorical(new_sample, num_classes=256).reshape(1, 1, -1)
    seed = np.concatenate([seed, new_sample_categorical], axis=1)

# Convert the generated audio back to waveform format
generated_audio = (np.argmax(seed[0], axis=-1) / (256 - 1) * 2)