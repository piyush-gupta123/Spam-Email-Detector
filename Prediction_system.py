from Spam_Email_detector import model,feature_extraction
input_mail=["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]
#Convert input mail using TFidVectorizer
test_data = feature_extraction.transform(input_mail)
predicted_value = model.predict(test_data)

if predicted_value[0] ==1:
    print("Not A Spam Mail")
else:
    print("Spam Mail")