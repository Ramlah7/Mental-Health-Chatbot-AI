import pandas as pd
from tagging import detect_intent

# Load the dataset
df = pd.read_csv("mental_health.csv")

# Add an intent column
df["intent"] = df["user_input"].apply(detect_intent)

# Show a sample for checking
print(df.head(10))

# Save the updated CSV
df.to_csv("mental_health_tagged.csv", index=False)
print("✅ Tagged CSV saved as 'mental_health_tagged.csv'")
