from flask import Flask, request, render_template
import joblib
#Chargement du modele
modele=joblib.load('modeleDT.pkl')
#Instanciation de Flask
app=Flask(__name__)
#Debut des APIs
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	mesFeatures=[float(x) for x in request.form.values()]
	import numpy as np

	mesFeatures=[np.array(mesFeatures)]
	resultat=round(modele.predict(mesFeatures)[0],2)
	res=f"La charge d'assurance à payer vaut: {resultat}$"
	return render_template('index.html', 
		resultat_prediction=res)



if __name__=="__main__":
	app.run(debug=False)
