""" Module returns FFNN model based on the type of model requested either Hero or Team. """
import keras
from keras.layers import Input, Dense, Add
from keras.models import Model

def ffnn_hero_model():
    """ Returns model and batch size for training and testing Hero data. """
    initializer = keras.initializers.glorot_normal()
    h_1 = Input(shape=(1,))
    hero_1 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_1)
    h_2 = Input(shape=(1,))
    hero_2 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_2)
    h_3 = Input(shape=(1,))
    hero_3 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_3)
    h_4 = Input(shape=(1,))
    hero_4 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_4)
    h_5 = Input(shape=(1,))
    hero_5 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_5)
    h_6 = Input(shape=(1,))
    hero_6 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_6)
    h_7 = Input(shape=(1,))
    hero_7 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_7)
    h_8 = Input(shape=(1,))
    hero_8 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_8)
    h_9 = Input(shape=(1,))
    hero_9 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_9)
    h_10 = Input(shape=(1,))
    hero_10 = Dense(16, activation='tanh', kernel_initializer=initializer)(h_10)

    team_1 = Add()([hero_1, hero_2, hero_3, hero_4, hero_5])
    team_2 = Add()([hero_6, hero_7, hero_8, hero_9, hero_10])

    merged = Add()([team_1, team_2])
    layer_1 = Dense(16, activation='tanh', kernel_initializer=initializer)(merged)

    output = Dense(1, activation='sigmoid', kernel_initializer=initializer)(layer_1)
    model_t = Model(inputs=[h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9, h_10], outputs=output)
    batch_size = 256
    model_t.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_t, batch_size

def ffnn_team_model():
    """ Returns model and batch size for training and testing Team data. """
    initializer = keras.initializers.glorot_normal()
    t_1 = Input(shape=(5,))
    team_1 = Dense(64, activation='relu', kernel_initializer=initializer)(t_1)
    t_2 = Input(shape=(5,))
    team_2 = Dense(64, activation='relu', kernel_initializer=initializer)(t_2)

    merged = Add()([team_1, team_2])
    layer_1 = Dense(32, activation='sigmoid', kernel_initializer=initializer)(merged)

    output = Dense(1, activation='sigmoid')(layer_1)
    model_t = Model(inputs=[t_1, t_2], outputs=output)
    batch_size = 128
    model_t.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_t, batch_size
