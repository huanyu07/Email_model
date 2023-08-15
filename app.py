from flask import Flask, render_template, request, session, flash, redirect, url_for
import config
import os
from model import bayes_predict, random_forest_predict, svm_predict
from hybrid_model import predict as hybrid_predict
from hybrid_model import Hybrid

app = Flask(__name__, static_folder='templates/Stack/assets')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
app.config.from_object(config)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        bayes_result = -1
        random_forest_result = -1
        svm_result = -1
        hybrid_result = -1
    else:
        text = request.form['text']
        bayes_result = bayes_predict(text)
        random_forest_result = random_forest_predict(text)
        svm_result = svm_predict(text)
        hybrid_result = hybrid_predict(text)
    return render_template('Stack/tables.html', bayes_result=bayes_result,
                           random_forest_result=random_forest_result, svm_result=svm_result, hybrid_result=hybrid_result)
@app.route("/new-url")
def new_url():
    return render_template('Stack/feature.html')

if __name__ == '__main__':
    app.run()

# if __name__ == '__main__':
#
#     db.drop_all()
#     db.create_all()


