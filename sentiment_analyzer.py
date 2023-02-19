from flask import Flask, request,jsonify,render_template
import joblib
import pandas as pd
import pickle


app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        content = request.form['review']
        r = pd.Series(content)
        vectorizer = joblib.load('vectorizer/vectorizer1.pkl')
        review = vectorizer.transform(r)
        print(review)
        classifier = joblib.load('model/model_2.pkl')
        prediction = classifier.predict(review)
        if prediction == 1.0:
            sentiment = 'Positive'
            return render_template('index.html',message = "Positive😁😁")
        elif prediction == -1.0:
            sentiment = 'negative'
            return render_template('index.html',message = "Negative😡😡")
        else:
            sentiment = 'Neutral'
            return render_template('index.html',message = "Neutral🙂🙂")
    return render_template('index.html')



if __name__ =='__main__':
    app.run(debug=True,port=5000)






# def predict():
#     if request.method == 'POST':
#         content = request.form['review']
#         sid = SentimentIntensityAnalyzer()
#         score = sid.polarity_scores(content)
#         if (score['compound'] >= 0.0):
#             render_template('index.html',message = "Positive😁😁")
#         else:
#             render_template('index.html', message="Negative😡😡")
#
#     return render_template('index.html')

