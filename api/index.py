from flask import Flask, render_template

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def home():
    metrics = {
        "accuracy": "0.93",  
        "report": """
               precision    recall  f1-score   support

         Low       0.98      0.93      0.96       613
      Medium       0.69      0.92      0.79       107
        High       0.00      0.00      0.00         3

    accuracy                           0.93       723
   macro avg       0.56      0.62      0.58       723
weighted avg       0.94      0.93      0.93       723
        """  
    }
    return render_template('index.html', metrics=metrics)

app.index = app

# Add this at the very bottom of api/index.py to test locally
if __name__ == '__main__':
    app.run(debug=True)