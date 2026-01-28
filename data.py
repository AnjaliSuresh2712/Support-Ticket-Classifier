import pandas as pd
import random
import os

os.makedirs('data', exist_ok=True)

categories = ['billing', 'technical', 'account', 'feature_request']
urgencies = ['low', 'medium', 'high']

ticket_templates = {
    'billing': [
        "I was charged twice for my subscription",
        "My credit card was declined but I have funds",
        "I need a refund for last month",
        "The pricing doesn't match what I was quoted",
        "Double billing issue on my account",
        "Can't update my payment method",
        "Incorrect invoice amount",
        "Need to cancel my subscription",
        "Proration charges seem wrong",
        "Annual plan pricing confusion",
        "Why was I charged for a feature I didn't use",
        "Payment failed but I see a pending charge",
        "Need to upgrade my subscription plan",
        "Downgrade billing didn't process correctly",
        "Tax calculation seems incorrect on invoice"
    ],
    'technical': [
        "The app keeps crashing on startup",
        "I can't log into my account",
        "Features are not loading properly",
        "Getting error code 500 when I try to save",
        "Dashboard won't load",
        "API integration is broken",
        "Mobile app freezes constantly",
        "Can't upload files",
        "Getting timeout errors",
        "Data sync is failing",
        "Search function not working",
        "Export to PDF fails every time",
        "Notifications aren't being sent",
        "Form submission returns error",
        "Page keeps redirecting me"
    ],
    'account': [
        "I need to update my email address",
        "How do I delete my account",
        "I forgot my password",
        "My account was locked",
        "Need to change my username",
        "Two-factor authentication issues",
        "Can't verify my email",
        "Account suspended notification",
        "Profile settings won't save",
        "Need to merge duplicate accounts",
        "How do I add team members",
        "Can't access my old account",
        "Need to transfer account ownership",
        "Profile picture upload fails",
        "Account security concerns"
    ],
    'feature_request': [
        "Can you add dark mode",
        "I'd like to export my data as CSV",
        "Please add two-factor authentication",
        "Integration with Slack would be helpful",
        "Need bulk upload functionality",
        "API documentation is needed",
        "Mobile app for iOS please",
        "Add calendar integration",
        "Need custom reporting features",
        "Would like email notifications",
        "Please add keyboard shortcuts",
        "Ability to customize dashboard layout",
        "Support for multiple languages",
        "Add webhook support",
        "Need better filtering options"
    ]
}

# Prefixes and suffixes for variety
prefixes = ["", "Help: ", "Issue: ", "Urgent: ", "Question: "]
suffixes = ["", " please help", " urgently", " ASAP", " thanks", " any update?", "!!!"]

tickets = []

# Generate 10,000 tickets
for _ in range(10000):
    category = random.choice(categories)
    template = random.choice(ticket_templates[category])
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    
    text = prefix + template + suffix
    text = text.strip()
    
    # Random case variations
    if random.random() < 0.1:
        text = text.upper()
    elif random.random() < 0.1:
        text = text.lower()
    
    tickets.append({
        'id': len(tickets),
        'text': text,
        'category': category,
        'urgency': random.choice(urgencies)
    })

df = pd.DataFrame(tickets)
df.to_csv('data/tickets.csv', index=False)

print(f"Generated {len(df)} support tickets")
print(f"Saved to data/tickets.csv")
print(f"\nCategory distribution:")
print(df['category'].value_counts())
print(f"\nUnique tickets: {df['text'].nunique()}")
print(f"Duplicate rate: {(1 - df['text'].nunique() / len(df)) * 100:.1f}%")