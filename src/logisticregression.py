# Build Logistic Regression model
logreg = LogisticRegression(random_state = 42)

# Train the model
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Compute accuracy score
acc_logreg = accuracy_score(y_test, y_pred)

# Print evaluation results
print("=== Logistic Regression Evaluation Results ===")
print("Accuracy:", acc_logreg)
print(classification_report(y_test, y_pred))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()