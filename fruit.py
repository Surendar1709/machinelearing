import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np # For numerical operations, especially with one-hot encoding

# 1. Create a Simple Dataset
# Let's define some fruit characteristics and their labels
data = {
    'Color': ['Red', 'Green', 'Yellow', 'Yellow', 'Orange', 'Red', 'Green'],
    'Shape': ['Round', 'Round', 'Curved', 'Round', 'Round', 'Heart', 'Oblong'],
    'FruitType': ['Apple', 'Apple', 'Banana', 'Lemon', 'Orange', 'Strawberry', 'Watermelon']
}
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)





# For this simplified example, let's restrict to just three fruit types
# to keep the tree very small and easy to trace.
df_filtered = df[df['FruitType'].isin(['Apple', 'Banana', 'Orange'])].copy()

# 2. Encode Categorical Features
# Decision trees work with numbers, so we need to convert 'Color' and 'Shape'
# One-hot encoding is suitable for features like these.
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df_filtered[['Color', 'Shape']])
encoded_feature_names = encoder.get_feature_names_out(['Color', 'Shape'])
X = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Encode the target variable (FruitType) to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_filtered['FruitType'])

print("\nFeatures (X) after One-Hot Encoding:")
print(X.head())
print("\nTarget (y) after Label Encoding (0: Apple, 1: Banana, 2: Orange):")
print(y)
print("\nMapping of labels:", list(label_encoder.classes_))


print("\n--- Data Preparation ---")

# 3. Split Data into Training and Testing Sets
# Small dataset, so we'll use a small test size or even no test set for demo
# Let's use no test set for this very small, illustrative example to simplify
# In a real scenario, you MUST use train/test split.
X_train, y_train = X, y
# If you wanted a split, it would be:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# print(f"Training data size: {len(X_train)} samples")
# print(f"Testing data size: {len(X_test)} samples")


print("\n--- Model Training ---")

# 4. Create and Train the Decision Tree Classifier
# We'll set a very small max_depth to make the tree easy to understand.
# criterion='gini' is default, let's keep it that way for simplicity.
model = DecisionTreeClassifier(random_state=42, max_depth=3) # Max depth of 3 for simple visualization

# Train the model
model.fit(X_train, y_train)

print("Decision Tree Model Trained!")

print("\n--- Model Evaluation (on training data for this small example) ---")

# 5. Make Predictions and Evaluate (on training data for simplicity)
y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Model Accuracy on Training Data: {accuracy_train:.2f}")
# If you used a test set:
# y_pred_test = model.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# print(f"Model Accuracy on Test Data: {accuracy_test:.2f}")


print("\n--- Visualizing the Decision Tree ---")

# 6. Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model,
          feature_names=encoded_feature_names, # Use the one-hot encoded feature names
          class_names=label_encoder.classes_, # Use the original fruit names
          filled=True,
          rounded=True,
          proportion=False, # Show counts, not proportions
          fontsize=10)
plt.title("Decision Tree for Fruit Classification (Simple Example)")
plt.show()


print("\n--- Making a New Prediction ---")
# 7. Make a prediction for a new, unseen fruit
new_fruit_data = pd.DataFrame([['Yellow', 'Curved']], columns=['Color', 'Shape'])

# Need to apply the same one-hot encoding as training data
# Ensure all possible columns from training are present, even if 0
new_fruit_encoded = encoder.transform(new_fruit_data)
new_fruit_df = pd.DataFrame(new_fruit_encoded, columns=encoded_feature_names)

predicted_label = model.predict(new_fruit_df)[0]
predicted_fruit_type = label_encoder.inverse_transform([predicted_label])[0]

print(f"New fruit: Color='{new_fruit_data['Color'].iloc[0]}', Shape='{new_fruit_data['Shape'].iloc[0]}'")
print(f"Predicted Fruit Type: {predicted_fruit_type}")

new_fruit_data_2 = pd.DataFrame([['Red', 'Round']], columns=['Color', 'Shape'])
new_fruit_encoded_2 = encoder.transform(new_fruit_data_2)
new_fruit_df_2 = pd.DataFrame(new_fruit_encoded_2, columns=encoded_feature_names)
predicted_label_2 = model.predict(new_fruit_df_2)[0]
predicted_fruit_type_2 = label_encoder.inverse_transform([predicted_label_2])[0]
print(f"New fruit: Color='{new_fruit_data_2['Color'].iloc[0]}', Shape='{new_fruit_data_2['Shape'].iloc[0]}'")
print(f"Predicted Fruit Type: {predicted_fruit_type_2}")

