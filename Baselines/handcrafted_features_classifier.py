import pandas as pd
import numpy as np
import re
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')


def calculate_entropy(string):
    """Calculate Shannon entropy of a string"""
    if not string or len(string) == 0:
        return 0
    
    # Count frequency of each character
    freq = {}
    for char in string:
        freq[char] = freq.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0
    length = len(string)
    for count in freq.values():
        p = count / length
        entropy -= p * log2(p)
    
    return entropy


def extract_handcrafted_features(candidate_string, context_line=None):
    """
    Extract handcrafted features from candidate string
    
    Features:
    1) Has parentheses (Possible function call)
    2) Has brackets (Possible variable declaration)
    3) Has periods (Possible function call)
    4) Begins with $ sign (Possible variable)
    5) Has word 'Password' in it (Possible variable initialization)
    6) Has spaces (Possible sentence/loops)
    7) Has HTML tags (HTML file)
    8) Starts with #, *, /* (Possible comment)
    9) Has arrow -> or => (Possible pointer variable)
    10) Has keywords null/nil/undefined/None/true/false (Programming initialization)
    11) Is numerical value
    12) Entropy bins [0,1), [1,2), [2,3), [3,4), >=4
    13) Has BEGIN PRIVATE KEY tag
    """
    
    features = {}
    
    # Handle None or empty strings
    if candidate_string is None or pd.isna(candidate_string):
        candidate_string = ""
    else:
        candidate_string = str(candidate_string)
    
    if context_line is None or pd.isna(context_line):
        context_line = ""
    else:
        context_line = str(context_line)
    
    # 1) Has parentheses
    features['has_parentheses'] = 1 if '(' in candidate_string or ')' in candidate_string else 0
    
    # 2) Has brackets
    features['has_brackets'] = 1 if '[' in candidate_string or ']' in candidate_string or '{' in candidate_string or '}' in candidate_string else 0
    
    # 3) Has periods
    features['has_periods'] = 1 if '.' in candidate_string else 0
    
    # 4) Begins with $ sign
    features['starts_with_dollar'] = 1 if candidate_string.strip().startswith('$') else 0
    
    # 5) Has word 'Password'
    features['has_password_word'] = 1 if 'password' in candidate_string.lower() else 0
    
    # 6) Has spaces
    features['has_spaces'] = 1 if ' ' in candidate_string else 0
    
    # 7) Has HTML tags
    html_pattern = r'<[^>]+>'
    features['has_html_tags'] = 1 if re.search(html_pattern, context_line) else 0
    
    # 8) Starts with comment characters
    stripped = candidate_string.strip()
    features['starts_with_comment'] = 1 if (stripped.startswith('#') or 
                                             stripped.startswith('*') or 
                                             stripped.startswith('/*') or
                                             stripped.startswith('//')) else 0
    
    # 9) Has arrow
    features['has_arrow'] = 1 if '->' in candidate_string or '=>' in candidate_string else 0
    
    # 10) Has programming keywords
    keywords = ['null', 'nil', 'undefined', 'none', 'true', 'false']
    lower_candidate = candidate_string.lower()
    features['has_prog_keywords'] = 1 if any(keyword in lower_candidate for keyword in keywords) else 0
    
    # 11) Is numerical value
    try:
        float(candidate_string.strip())
        features['is_numerical'] = 1
    except (ValueError, AttributeError):
        features['is_numerical'] = 0
    
    # 12) Entropy bins
    entropy = calculate_entropy(candidate_string)
    features['entropy_0_1'] = 1 if 0 <= entropy < 1 else 0
    features['entropy_1_2'] = 1 if 1 <= entropy < 2 else 0
    features['entropy_2_3'] = 1 if 2 <= entropy < 3 else 0
    features['entropy_3_4'] = 1 if 3 <= entropy < 4 else 0
    features['entropy_4_plus'] = 1 if entropy >= 4 else 0
    features['entropy_value'] = entropy
    
    # 13) Has BEGIN PRIVATE KEY tag
    features['has_private_key_tag'] = 1 if 'BEGIN PRIVATE KEY' in candidate_string or 'BEGIN RSA PRIVATE KEY' in candidate_string else 0
    
    # Additional useful features
    # features['length'] = len(candidate_string)
    # features['num_digits'] = sum(c.isdigit() for c in candidate_string)
    # features['num_upper'] = sum(c.isupper() for c in candidate_string)
    # features['num_lower'] = sum(c.islower() for c in candidate_string)
    # features['num_special'] = sum(not c.isalnum() and not c.isspace() for c in candidate_string)
    
    return features


def load_and_prepare_data(train_path, test_path, sample_size=None):
    """Load data and extract features"""
    print("Loading data...")
    
    # Load datasets
    if sample_size:
        train_df = pd.read_csv(train_path, nrows=sample_size)
        test_df = pd.read_csv(test_path, nrows=sample_size)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Label distribution in train: {train_df['label'].value_counts().to_dict()}")
    
    # Extract features
    print("\nExtracting features from training data...")
    train_features = []
    for idx, row in train_df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(train_df)} training samples...")
        features = extract_handcrafted_features(row['candidate_string'], row.get('context_window', ''))
        train_features.append(features)
    
    print("\nExtracting features from test data...")
    test_features = []
    for idx, row in test_df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(test_df)} test samples...")
        features = extract_handcrafted_features(row['candidate_string'], row.get('context_window', ''))
        test_features.append(features)
    
    # Convert to DataFrames
    X_train = pd.DataFrame(train_features)
    X_test = pd.DataFrame(test_features)
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Feature names: {list(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classifiers"""
    
    results = {}
    
    # Define classifiers
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Naive Bayes': GaussianNB()
    }
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING INDIVIDUAL CLASSIFIERS")
    print("="*80)
    
    trained_classifiers = {}
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train
        print("  Training...")
        clf.fit(X_train, y_train)
        trained_classifiers[name] = clf
        
        # Predict
        print("  Predicting...")
        y_pred = clf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"  {cm}")
        
        # Detailed classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Secret', 'Secret']))
    
    # Voting Classifier
    print("\n" + "="*80)
    print("VOTING CLASSIFIER (Ensemble)")
    print("="*80)
    
    # Use the pre-trained classifiers for voting
    voting_clf = VotingClassifier(
        estimators=[
            # ('dt', trained_classifiers['Decision Tree']),
            # ('rf', trained_classifiers['Random Forest']),
            # ('knn', trained_classifiers['K-Nearest Neighbors']),
            ('lr', trained_classifiers['Logistic Regression']),
            ('svm', trained_classifiers['SVM']),
            ('nb', trained_classifiers['Naive Bayes'])
        ],
        voting='hard'
    )
    
    print("  Creating ensemble predictions...")
    # Since classifiers are already trained, we just need to get predictions
    # The VotingClassifier needs to be "fit" but we'll use a workaround
    voting_clf.fit(X_train[:100], y_train[:100])  # Dummy fit with small sample
    y_pred_voting = voting_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_voting)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_voting, average='binary', pos_label=1)
    
    results['Voting Classifier'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred_voting
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_voting)
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_voting, target_names=['Not Secret', 'Secret']))
    
    return results


def save_results(results, output_file):
    """Save results to file"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HANDCRAFTED FEATURES CLASSIFICATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Features Used:\n")
        f.write("1. Has parentheses (function call)\n")
        f.write("2. Has brackets (variable declaration)\n")
        f.write("3. Has periods (function call)\n")
        f.write("4. Begins with $ sign (variable)\n")
        f.write("5. Has word 'Password'\n")
        f.write("6. Has spaces (sentence/loops)\n")
        f.write("7. Has HTML tags\n")
        f.write("8. Starts with comment characters (#, *, /*)\n")
        f.write("9. Has arrow (-> or =>)\n")
        f.write("10. Has programming keywords (null/nil/undefined/None/true/false)\n")
        f.write("11. Is numerical value\n")
        f.write("12. Entropy bins [0,1), [1,2), [2,3), [3,4), >=4\n")
        f.write("13. Has BEGIN PRIVATE KEY tag\n")
        f.write("+ Additional features: length, digit count, case, special chars\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Create summary table
        f.write(f"{'Classifier':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-"*80 + "\n")
        
        for name, metrics in results.items():
            f.write(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        f.write(f"\nBest Model (by F1-Score): {best_model[0]}\n")
        f.write(f"F1-Score: {best_model[1]['f1']:.4f}\n")
        f.write(f"Accuracy: {best_model[1]['accuracy']:.4f}\n")
        f.write(f"Precision: {best_model[1]['precision']:.4f}\n")
        f.write(f"Recall: {best_model[1]['recall']:.4f}\n")
    
    print(f"\nResults saved to {output_file}")


def main():
    print("="*80)
    print("HANDCRAFTED FEATURES SECRET DETECTION")
    print("="*80)
    
    # Paths
    train_path = '/home/nafiu/baselines/train.csv'
    test_path = '/home/nafiu/baselines/test.csv'
    output_file = '/home/nafiu/baselines/handcrafted_features_results.txt'
    
    # Load and prepare data (use sample_size parameter if datasets are too large)
    # For full dataset, set sample_size=None
    # For testing, use sample_size=10000
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        train_path, 
        test_path, 
        sample_size=None  # Set to None for full dataset, or a number like 50000 for sampling
    )
    
    # Train and evaluate
    results = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)
    
    # Save results
    save_results(results, output_file)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
