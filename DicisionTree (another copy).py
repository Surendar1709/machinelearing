import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the Dataset
# The Iris dataset is built into scikit-learn, making it easy to access.
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) # Features (e.g., sepal length, petal width)
y = pd.Series(iris.target) # Target variable (species: 0, 1, 2)

# Display the first few rows of the data and target names
print("Features (X) head:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())
print("\nIris Species Names:", iris.target_names)

print("\n--- Data Preparation ---")

# 2. Split Data into Training and Testing Sets
# We split the data to evaluate our model's performance on unseen data.
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)} samples")
print(f"Testing data size: {len(X_test)} samples")

print("\n--- Model Training ---")

# 3. Create and Train the Decision Tree Classifier
# We initialize the DecisionTreeClassifier.
# random_state ensures reproducibility of the results.
# criterion='gini' is the default, but you can also use 'entropy'.
model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=3)

# Train the model using the training data
model.fit(X_train, y_train)

print("Decision Tree Model Trained!")

print("\n--- Model Evaluation ---")

# 4. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 5. Evaluate the Model's Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

print("\n--- Visualizing the Decision Tree ---")

# 6. Visualize the Decision Tree
# This step helps us understand the rules the tree learned.
plt.figure(figsize=(15, 10))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, # Color nodes to indicate majority class
          rounded=True, # Round node corners
          proportion=True, # Show proportion of samples in each class
          fontsize=10)
plt.title("Decision Tree for Iris Classification")
plt.show()

# 7. Understanding a Single Prediction (Optional)
# Let's take one sample from the test set and see its predicted class
sample_index = 0 # You can change this index
sample_features = X_test.iloc[[sample_index]]
true_species_code = y_test.iloc[sample_index]
predicted_species_code = model.predict(sample_features)[0]

print(f"\n--- Single Prediction Example ---")
print(f"Sample Features:\n{sample_features}")
print(f"True Species: {iris.target_names[true_species_code]}")
print(f"Predicted Species: {iris.target_names[predicted_species_code]}")