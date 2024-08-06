import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox

data = {
    'review': [
        'The food was great and the service was fantastic!',
        'Horrible experience, will not come again.',
        'Loved the ambiance and the food was delicious.',
        'Service was slow and food was cold.',
        'Best restaurant in town!',
        'Terrible, absolutely terrible.',
        'Amazing food and wonderful staff.',
        'Not worth the money, very disappointing.',
        'Excellent food and a wonderful atmosphere.',
        'The worst dining experience I have ever had.'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Model Performance on Test Data:")
print(classification_report(y_test, y_pred))

def predict_sentiment(review):
    input_vector = vectorizer.transform([review])
    prediction = model.predict(input_vector)[0]
    return "Positive" if prediction == 1 else "Negative"

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis")

        self.label = tk.Label(root, text="Enter your review:")
        self.label.pack()

        self.text_input = tk.Entry(root, width=50)
        self.text_input.pack()

        self.predict_button = tk.Button(root, text="Predict Sentiment", command=self.predict)
        self.predict_button.pack()

        sample_reviews = [
            "The food was delicious and the service was excellent!",
            "Terrible experience, food was cold and service was slow."
        ]
        
        for review in sample_reviews:
            button = tk.Button(root, text=review, command=lambda r=review: self.set_sample(r))
            button.pack()

    def set_sample(self, sample):
        self.text_input.delete(0, tk.END)
        self.text_input.insert(0, sample)

    def predict(self):
        review = self.text_input.get()
        sentiment = predict_sentiment(review)
        messagebox.showinfo("Prediction", f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()