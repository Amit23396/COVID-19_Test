import pickle
from flask import Flask, render_template, request

app = Flask(__name__)


file = open('templates/model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello():
	if request.method == 'POST':
		myDict = request.form
		fever = int(myDict['fever'])
		bodyPain = int(myDict['bodyPain'])
		age = int(myDict['age'])
		runnyNose = int(myDict['runnyNose'])
		diffBreath = int(myDict['diffBreath'])

		inputFeatures = [fever, bodyPain, age, runnyNose, diffBreath]
		infProb = clf.predict_proba([inputFeatures])[0][1]
		print(infProb)
		return render_template('show.html', inf = round(infProb*100))
	return render_template('index.html')
#	return 'Hello _'+str(infProb[0])

if __name__ == '__main__':
	app.run(debug=True)