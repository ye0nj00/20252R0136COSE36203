# Build Logistic Regression model using VADER
logreg_vader = LogisticRegression(random_state = 42)

# Train the model
logreg_vader.fit(X_train_vader, y_train_vader)

# Predict on test set
y_pred_vader = logreg_vader.predict(X_test_vader)

# Compute accuracy score
acc_logreg_vader = accuracy_score(y_test_vader, y_pred_vader)

# Print evaluation results
print("=== Logistic Regression using VADER Evaluation Results ===")
print("Accuracy:", acc_logreg_vader)
print(classification_report(y_test_vader, y_pred_vader))

# Plot confusion matrix as heatmap
cm = confusion_matrix(y_test_vader, y_pred_vader)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression using VADER Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()