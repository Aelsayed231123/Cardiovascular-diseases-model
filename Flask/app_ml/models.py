from joblib import load
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

print("Current Working Directory:", os.getcwd())
dl_model_path = 'model.keras'
knn_path = 'Lreg_model.joblib'
lreg_path = 'knn_model.joblib'
preprocessor_path = 'preprocessor.joblib'
print("DL File Exists:", os.path.isfile(dl_model_path))
print("KNN File Exists:", os.path.isfile(knn_path))
print("Lreg File Exists:", os.path.isfile(lreg_path))





modeltf = tf.keras.models.load_model(filepath = dl_model_path)
knn = load(knn_path)
lreg = load(lreg_path)
loadedpp = load(preprocessor_path)




