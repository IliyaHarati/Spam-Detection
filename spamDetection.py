import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tkinter import *

# Load data
data = pd.read_csv("spam.csv")

# Separate target variables
X = data["Message"]
y = data["Category"]

# Preprocess email text
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X)

# Train Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y)


# Define GUI
def classify_message():
    message = entry_text.get()
    message_vector = vectorizer.transform([message])
    prediction = clf.predict(message_vector)[0]
    if prediction == "spam":
        result_text.set("This message is spam.")
    else:
        result_text.set("This message is not spam.")


# Create GUI
root = Tk()
root.title("Spam Classifier")

mainframe = Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack(pady=50, padx=50)

entry_label = Label(mainframe, text="Enter a message:")
entry_label.grid(column=1, row=1, sticky=W)

entry_text = Entry(mainframe, width=50)
entry_text.grid(column=2, row=1, sticky=W)

classify_button = Button(mainframe, text="Classify", command=classify_message)
classify_button.grid(column=3, row=1, sticky=W)

result_text = StringVar()
result_label = Label(mainframe, textvariable=result_text)
result_label.grid(column=2, row=2, sticky=W)

root.mainloop()
