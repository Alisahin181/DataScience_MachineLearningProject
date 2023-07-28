
class DecisionTreeClassifier:
    def __init__(self, max_depth= None):
        self.max_depth= max_depth
        self.tree=None

    def fit(self, X, y):
        self.tree= self.build_tree(X,y)


    def build_tree(self, X, y, depth=0):
        classes= list(set(y))

        if self.max_depth is not None and depth >= self.max_depth or len(classes)== 1:
           return {'class': classes[0]}

        num_features = len(X[0])
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        # En iyi bölme kriterini seçme
        for feature in range(num_features):
            thresholds = list(set(X[i][feature] for i in range(len(X))))
            for threshold in thresholds:
                left_indices = [i for i in range(len(X)) if X[i][feature] <= threshold]
                right_indices = [i for i in range(len(X)) if X[i][feature] > threshold]
                gini = self.gini_impurity(y, left_indices, right_indices)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        # Düğümü oluşturma
        node = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': None,
            'right': None
        }

        # Sol ve sağ alt ağaçları oluşturma
        left_indices = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
        node['left'] = self.build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1)
        node['right'] = self.build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1)

        return node


    def gini_impurity(self, y, left_indices, right_indices):
        left_classes = [y[i] for i in left_indices]
        right_classes = [y[i] for i in right_indices]

        left_counts = {}
        right_counts = {}

        for class_label in left_classes:
            if class_label in left_counts:
                left_counts[class_label] += 1
            else:
                left_counts[class_label] = 1

        for class_label in right_classes:
            if class_label in right_counts:
                right_counts[class_label] += 1
            else:
                right_counts[class_label] = 1

        left_probs = [count / len(left_classes) for count in left_counts.values()]
        right_probs = [count / len(right_classes) for count in right_counts.values()]

        left_gini = 1 - sum([prob**2 for prob in left_probs])
        right_gini = 1 - sum([prob**2 for prob in right_probs])

        gini = (sum(left_counts.values()) / len(y)) * left_gini + (sum(right_counts.values()) / len(y)) * right_gini
        return gini

    def traverse_tree(self, sample, node):
        if 'class' in node:
            return node['class']

        if sample[node['feature']] <= node['threshold']:
            return self.traverse_tree(sample, node['left'])
        else:
            return self.traverse_tree(sample, node['right'])

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.traverse_tree(sample, self.tree)
            predictions.append(prediction)
        return predictions
    
    def predict_proba(self, X):
        proba = []
        for sample in X:
            probabilities = self.traverse_proba(sample, self.tree)
            proba.append(probabilities)
        return np.array(proba)

    def traverse_proba(self, sample, node):
        if 'class' in node:
            return {c: 1.0 if c == node['class'] else 0.0 for c in np.unique(y)}

        if sample[node['feature']] <= node['threshold']:
            return self.traverse_proba(sample, node['left'])
        else:
            return self.traverse_proba(sample, node['right'])





