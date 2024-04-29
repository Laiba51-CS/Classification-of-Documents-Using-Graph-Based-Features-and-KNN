import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix
import networkx as nx
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# nltk.download('punkt')
# nltk.download('stopwords')


disease_file_path  = 'Articles/diseases.csv'
travel_file_path  = 'Articles/travel.csv'
science_education_file_path  = 'Articles/sci_edu.csv'

# Read in the data

#encoding="latin1"
travel_data = pd.read_csv(travel_file_path , encoding="latin1")
disease_data = pd.read_csv(disease_file_path,  encoding="latin1")
scie_edu_data = pd.read_csv(science_education_file_path ,encoding="latin1" )
# Read data from CSV files

# Display the shape of the data
#print(travel_data.shape)
#print(disease_data.shape)
#rint(scie_edu_data.shape)

# Display the columns of the data
#print(travel_data.columns)

# Display the first few rows of the data
#print(travel_data.head())




# Dividing the articles into data set and test set
train_travel= travel_data[:10]
test_travel= travel_data[10:15]

train_disease = disease_data[:10]
test_disease = disease_data[10:15]

train_science = scie_edu_data[:10]
test_science = scie_edu_data[10:15]

train_set = pd.concat([train_travel, train_disease, train_science], ignore_index=True)
test_set = pd.concat([test_travel, test_disease, test_science], ignore_index=True)



#checkng the length  30 sets of training and 15 sets of training half of it  
#print("Training set size:", len(train_set))
#print("Test set size:", len(test_set))

def tokenize(text):
    return nltk.word_tokenize(str(text).lower())

# Stop-word removal chlao
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Stemming
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

###########################################################################################################################################
###################################processing training set #################################################################################
############################################################################################################################################


def preprocess(data):
    preprocessed_data = []
    for index, row in data.iterrows():  # Iterate over each row in the training set DataFrame
    
        body_tokens = tokenize(row['body'])    # Access the 'body' column of the current row

        # title_tokens = remove_stopwords(title_tokens)
        body_tokens = remove_stopwords(body_tokens)

        # title_tokens = stem_tokens(title_tokens)
        body_tokens = stem_tokens(body_tokens)

        words_count = len(body_tokens)
        
        preprocessed_data.append({'label': row['label'], 'title': row['title'], 'body_tokens': body_tokens, 'words_count': words_count})
    return preprocessed_data




# Make a Directed Graph according to the paper
def make_Graph(text):
    # Split the string into words
    words = text
    # Create a directed graph
    G = nx.DiGraph()
    # Add nodes for each unique word
    for chunk in set(words):
        G.add_node(chunk)
    # Add edges between adjacent words
    for i in range(len(words) - 1):
        G.add_edge(words[i], words[i + 1])
        # nx.draw(G, with_labels=True)
        # plt.show()
    return G




##################################################################################################################################
#######################################  Making KNN class ###################################################################################
####################################################################################################################


class GraphKNN:
    def __init__(self, k: int):
        self.k = k
        self.train_graphs = []
        self.train_labels = []

    def fit(self, train_graphs, train_labels):
        self.train_graphs = train_graphs
        self.train_labels = train_labels

    def predict(self, graph):
        distances = []
        for train_graph in self.train_graphs:
            distance = Distance(graph, train_graph)
            distances.append(distance)
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[
            : self.k
        ]
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        # print("Prediction:", prediction)
        return prediction



def Distance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())

#####################################################################################################################################################
###############################################Making confusion matrix##############################################################################
####################################################################################################################################################
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# def processor(text):
#     if isinstance(text, str):
#         # Tokenization
#         tokens = re.findall(r"\b\w+\b", text.lower())
#         # Stop-word removal and stemming can be added here if needed
#         return " ".join(tokens)
#     else:
#         return "nothing"

# # Preprocess text data
#  train_set["text"] = train_set["body"].apply(processor)
preprocessed_train_set = preprocess(train_set)
trainGraphs = []
for index, article in enumerate(preprocessed_train_set):
    # Build the directed graph
    print("Article : ", index+1, "Graph built")
    graph = make_Graph(article['body_tokens'])
    trainGraphs.append(graph)

# Prepare training data
# trainTexts = preprocessed_train_set["body_tokens"].tolist()
# # trainTexts = train_set["text"].tolist()
# # trainLabels = train_set["label"].tolist()
trainLabels = [ row['label'] for  row in preprocessed_train_set]
# # trainGraphs = [make_Graph(text) for text in trainTexts]
# trainGraphs = [make_Graph(text) for text['body_tokens'] in preprocessed_train_set]

# Train the model
graphClassifier = GraphKNN(k=3)
graphClassifier.fit(trainGraphs, trainLabels)

# Test data 
preprocessed_test_set = preprocess(test_set)
testLabels = [ row['label'] for  row in preprocessed_test_set]
testText = [ row['body_tokens'] for  row in preprocessed_test_set]



testGraphs = [make_Graph(text) for text in testText]

# Predict
predictions = [graphClassifier.predict(graph) for graph in testGraphs]

# Evaluate 
# testLabels = ["diseases", "science and education", "travel"] 
# Calculate accuracy
accuracy = accuracy_score(testLabels, predictions)
# Calculate accuracy
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Calculate F1 Score for the second class
f1Scores = f1_score(testLabels, predictions, average=None)
# print("F1 Scores:", f1_scores)
f1ScorePercentage = f1Scores[0] * 100
print("F1 Score:", "{:.2f}%".format(f1ScorePercentage))

# Calculate Jaccard similarity for the second class
jaccard = jaccard_score(testLabels, predictions, average=None)
# print("jaccard:",jaccard)
jaccard_percentage = jaccard[0] * 100
print("Jaccard Similarity:", "{:.2f}%".format(jaccard_percentage))


# Plot confusion matrix
confMatrix = confusion_matrix(
    list(testLabels), list(predictions), labels=list(set(testLabels))
)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(
    confMatrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=set(testLabels),
    yticklabels=set(testLabels),
)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()




