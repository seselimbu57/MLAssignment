import random

news_data = [
    ("A new AI technology is revolutionizing the tech industry.", "Technology"),
    ("Cybersecurity threats are increasing rapidly.", "Technology"),
    ("Quantum computing is becoming a reality.", "Technology"),
    ("Tech companies are investing heavily in renewable energy.", "Technology"),
    ("The latest smartphone model has been released.", "Technology"),
    ("A popular movie has broken box office records.", "Entertainment"),
    ("A renowned actor won an award for their latest film.", "Entertainment"),
    ("A new streaming service is launching next month.", "Entertainment"),
    ("Music festivals are attracting larger crowds than ever.", "Entertainment"),
    ("A bestselling author is releasing a new book series.", "Entertainment"),
    ("A new startup is disrupting the food delivery market.", "Business"),
    ("Stock markets are experiencing unprecedented volatility.", "Business"),
    ("A major corporation is undergoing a significant merger.", "Business"),
    ("Entrepreneurs are finding innovative ways to fund their projects.", "Business"),
    ("A new financial regulation is impacting businesses.", "Business"),
    ("The city is implementing new sustainability initiatives.", "Environment"),
    ("Wildlife conservation efforts are gaining momentum.", "Environment"),
    ("A new national park is being established.", "Environment"),
    ("Climate change policies are being strengthened.", "Environment"),
    ("Renewable energy sources are becoming more widespread.", "Environment"),
]

classes = ["Technology", "Entertainment", "Business", "Environment"]


def random_predict():
    return random.choice(classes)


correct_predictions = 0


for news, true_class in news_data:
    predicted_class = random_predict()
    if predicted_class == true_class:
        correct_predictions += 1


accuracy = correct_predictions / len(news_data)
accuracy_percentage = accuracy * 100


print(f"Accuracy of the random model: {accuracy_percentage:.2f}%")
