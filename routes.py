from flask import Flask, render_template
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor

regr = DecisionTreeRegressor(random_state=0)

app = Flask(__name__)

@app.route("/")
def index():
    model = joblib.load('lab2_soos.pkl')
    prediction = model.predict([[4, 2.5, 3005, 15, 17903.0]]).round(1)
    prediction.tostring()
    return render_template("index.html", prediction=prediction)


if __name__ == '__main__':

    app.run(debug=True)