from tensorflow import keras


def factorize(
	N: int, 
	M: int, 
	D: int, 
	reg: float,
) -> keras.Model:

	user_input = keras.layers.Input(shape=(1,))
	movie_input = keras.layers.Input(shape=(1,))

	U = keras.layers.Embedding(N, D, 
		embeddings_regularizer=keras.regularizers.l2(reg))(user_input)
	W = keras.layers.Embedding(M, D, 
		embeddings_regularizer=keras.regularizers.l2(reg))(movie_input)
	B = keras.layers.Embedding(N, 1, 
		embeddings_regularizer=keras.regularizers.l2(reg))(user_input)
	C = keras.layers.Embedding(M, 1,
	 	embeddings_regularizer=keras.regularizers.l2(reg))(movie_input)

	UW = keras.layers.Dot(axes=2)([U, W])
	R = keras.layers.Add()([UW, B, C])
	R = keras.layers.Flatten()(R)

	return keras.Model([user_input, movie_input], R)

"""
mu = df_train.rating.mean()

model.compile(
  loss='mse',
  optimizer=keras.optimizers.SGD(lr=0.08, momentum=0.9),
  metrics=['mse'])

history = model.fit(
  [df_train.userId.values, df_train.movieId.values],
  df_train.rating.values - mu,
  epochs=15,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movieId.values],
    df_test.rating.values - mu
  )
)
"""
