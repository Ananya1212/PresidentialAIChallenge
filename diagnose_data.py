print("DIAGNOSTIC SCRIPT IS RUNNING")

import joblib
import pandas as pd

print("Loading preprocessor...")
preprocessor = joblib.load("preprocessor.pkl")

print("Preprocessor expects columns:")
print(preprocessor.feature_names_in_)

print("DONE")
