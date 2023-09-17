import tensorflow as tf

nilai = 20
tf_model = tf.keras.models.load_model('model/')
prediction = tf_model.predict([nilai])
print('predictions =',float(prediction))