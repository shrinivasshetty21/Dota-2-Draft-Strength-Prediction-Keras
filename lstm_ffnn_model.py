""" Module returns LSTM FFNN model based on the type of model requested either Hero or Team. """
import keras
from keras.layers import Input, Dense, Add, LSTM, Embedding
from keras.models import Model

def lstm_ffnn_hero_model():
    """ Returns LSTM FFNN model and batch size for training and testing Hero data. """
    initializer = keras.initializers.glorot_normal()

    hero_1_ = Input(shape=(1,), dtype='int32', name='hero_1')
    hero_2_ = Input(shape=(1,), dtype='int32', name='hero_2')
    hero_3_ = Input(shape=(1,), dtype='int32', name='hero_3')
    hero_4_ = Input(shape=(1,), dtype='int32', name='hero_4')
    hero_5_ = Input(shape=(1,), dtype='int32', name='hero_5')
    hero_6_ = Input(shape=(1,), dtype='int32', name='hero_6')
    hero_7_ = Input(shape=(1,), dtype='int32', name='hero_7')
    hero_8_ = Input(shape=(1,), dtype='int32', name='hero_8')
    hero_9_ = Input(shape=(1,), dtype='int32', name='hero_9')
    hero_10_ = Input(shape=(1,), dtype='int32', name='hero_10')

    emb = Embedding(input_dim=125, output_dim=16, input_length=1)
    emb_h1 = emb(hero_1_)
    emb_h2 = emb(hero_2_)
    emb_h3 = emb(hero_3_)
    emb_h4 = emb(hero_4_)
    emb_h5 = emb(hero_5_)
    emb_h6 = emb(hero_6_)
    emb_h7 = emb(hero_7_)
    emb_h8 = emb(hero_8_)
    emb_h9 = emb(hero_9_)
    emb_h10 = emb(hero_10_)

    lstm_heroes = LSTM(16, input_shape=(1, 16))

    emb_lstm_1 = lstm_heroes(emb_h1)
    emb_lstm_2 = lstm_heroes(emb_h2)
    emb_lstm_3 = lstm_heroes(emb_h3)
    emb_lstm_4 = lstm_heroes(emb_h4)
    emb_lstm_5 = lstm_heroes(emb_h5)
    emb_lstm_6 = lstm_heroes(emb_h6)
    emb_lstm_7 = lstm_heroes(emb_h7)
    emb_lstm_8 = lstm_heroes(emb_h8)
    emb_lstm_9 = lstm_heroes(emb_h9)
    emb_lstm_10 = lstm_heroes(emb_h10)

    team_1 = Add()([emb_lstm_1, emb_lstm_2, emb_lstm_3, emb_lstm_4, emb_lstm_5])
    team_1_ffnn = Dense(32, activation='tanh', kernel_initializer=initializer)(team_1)

    team_2 = Add()([emb_lstm_6, emb_lstm_7, emb_lstm_8, emb_lstm_9, emb_lstm_10])
    team_2_ffnn = Dense(32, activation='tanh', kernel_initializer=initializer)(team_2)

    merged = Add()([team_1_ffnn, team_2_ffnn])
    layer_1 = Dense(16, activation='tanh', kernel_initializer=initializer)(merged)

    output = Dense(1, activation='sigmoid')(layer_1)
    model_t = Model(inputs=[hero_1_, hero_2_, hero_3_, hero_4_, hero_5_, hero_6_, hero_7_, hero_8_, hero_9_, hero_10_], outputs=[output])
    batch_size = 64
    model_t.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_t, batch_size

def lstm_ffnn_team_model():
    """ Returns LSTM FFNN model and batch size for training and testing Team data. """
    initializer = keras.initializers.glorot_normal()
    t_1 = Input(shape=(5,), dtype='int32', name='Team_1')
    t_2 = Input(shape=(5,), dtype='int32', name='Team_2')

    emb = Embedding(input_dim=125, output_dim=32, input_length=5)
    emb_t1 = emb(t_1)
    emb_t2 = emb(t_2)

    lstm_team = LSTM(32, input_shape=(1, 32))

    emb_lstm_1 = lstm_team(emb_t1)
    team_1_ffnn = Dense(32, activation='tanh', kernel_initializer=initializer)(emb_lstm_1)

    emb_lstm_2 = lstm_team(emb_t2)
    team_2_ffnn = Dense(32, activation='tanh', kernel_initializer=initializer)(emb_lstm_2)

    merged = Add()([team_1_ffnn, team_2_ffnn])
    layer_1 = Dense(32, activation='tanh', kernel_initializer=initializer)(merged)

    output = Dense(1, activation='sigmoid')(layer_1)
    model_t = Model(inputs=[t_1, t_2], outputs=output)
    batch_size = 64
    model_t.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_t, batch_size
