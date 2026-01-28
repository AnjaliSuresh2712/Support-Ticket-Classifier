import pandas as pd
import random
import os

#data data folder to save csv file inside
os.makedirs('data', exist_ok=True)

categories = ['billing', 'technical', 'account', 'feature_request']
urgencies = ['low', 'medium', 'high']
tickets = []

#synthetic ticket data
ticket_templates = {
    'billing': [
        "I was charged twice for my subscription",
        "My credit card was declined but I have funds",
        "I need a refund for last month",
        "The pricing doesn't match what I was quoted"
    ],
    'technical': [
        "The app keeps crashing on startup",
        "I can't log into my account",
        "Features are not loading properly",
        "Getting error code 500 when I try to save"
    ],
    'account': [
        "I need to update my email address",
        "How do I delete my account",
        "I forgot my password",
        "My account was locked"
    ],
    'feature_request': [
        "Can you add dark mode",
        "I'd like to export my data as CSV",
        "Please add two-factor authentication",
        "Integration with Slack would be helpful"
    ]
}

for i in range(150):
    category = random.choice(categories)
    template = random.choice(ticket_templates[category])
    tickets.append({
        'id': i,
        'text': template,
        'category': category,
        'urgency': random.choice(urgencies)
    })

df = pd.DataFrame(tickets)
df.to_csv('data/tickets.csv', index=False)
print(f"✓ Generated {len(df)} support tickets")
print(f"✓ Saved to data/tickets.csv")
print(f"\nCategory distribution:")
print(df['category'].value_counts())
