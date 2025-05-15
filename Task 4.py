import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'ID': [1,2,3,4,5,6,7,8,9,10],
    'Date': ['2025-05-01', '2025-05-02', '2025-05-03', '2025-05-04', '2025-05-05', '2025-05-06', '2025-05-07', '2025-05-08', '2025-05-09', '2025-05-10'],
    'Tweet': [
        "I love the new features of Brand X! It's amazing!",
        "Brand X is terrible! Their customer support is awful!",
        "I think Brand X could improve their app. Needs work.",
        "The new update from Brand X is super smooth. Great job!",
        "I had to wait too long for my order from Brand X.",
        "Brand X is okay, but I expected more.",
        "I can't get enough of Brand X's products. Best ever!",
        "The new features are confusing. Not happy with it.",
        "Brand X's design is pretty good but could be improved.",
        "I'm absolutely in love with Brand X. Highly recommend!"
    ]
}

df = pd.DataFrame(data)

# Perform sentiment analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Tweet'].apply(get_sentiment)

# Count sentiment distribution
sentiment_counts = df['Sentiment'].value_counts()

# Plot pie chart
plt.figure(figsize=(6,6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
plt.title('Sentiment Distribution (Offline Sample)')
plt.axis('equal')
plt.show()
