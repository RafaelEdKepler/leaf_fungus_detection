from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# definir o modelo de CNN
modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Conv2D(128, (3, 3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(256, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(3, activation='softmax'))

# compilar o modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# pré-processar as imagens
gerador_treinamento = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
gerador_teste = ImageDataGenerator(rescale=1./255)

# carregar as imagens
conjunto_treinamento = gerador_treinamento.flow_from_directory('./treinamento/CF3', target_size=(128, 128), batch_size=32, class_mode='categorical')
conjunto_teste = gerador_teste.flow_from_directory('./treinamento/Sem fungicida', target_size=(128, 128), batch_size=32, class_mode='categorical')

# treinar o modelo
modelo.fit_generator(conjunto_treinamento, steps_per_epoch=100, epochs=10, validation_data=10)