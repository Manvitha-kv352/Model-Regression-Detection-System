from classifier import classify_email

email = "I was changed my password still its showing incorrect."

result = classify_email(email)

print("Category:", result["category"])
print("Summary:", result["summary"])