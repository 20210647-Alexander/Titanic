from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo y los transformadores
clf, pca, encoder, selected_features, scaler = joblib.load('modelo_titanic.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Obtener datos del formulario
        data = {
            'Pclass': int(request.form['Pclass']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': float(request.form['Fare']),
            'Sex': request.form['Sex'],
            'Embarked': request.form['Embarked'],
            'Ticket': request.form['Ticket'],
            'Cabin': request.form['Cabin'],
            'Age': float(request.form['Age'])
        }

        df_input = pd.DataFrame([data])
        
        # Transformar variables categóricas
        df_input[['Sex', 'Embarked', 'Ticket', 'Cabin']] = encoder.transform(df_input[['Sex', 'Embarked', 'Ticket', 'Cabin']])

        # Seleccionar características
        x_input = df_input[selected_features]

        # Escalar
        x_input_scaled = scaler.transform(x_input)

        # Aplicar PCA
        x_input_pca = pca.transform(x_input_scaled)

        # Predecir
        resultado = clf.predict(x_input_pca)
        prediction = "Sobrevivió" if resultado[0] == 1 else "No sobrevivió"

    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
