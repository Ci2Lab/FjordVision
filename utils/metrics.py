from anytree.search import find
from anytree.walker import Walker

def hierarchical_similarity(node1, node2, tree):
    walker = Walker()
    # Assuming node1 and node2 are already the correct nodes from the tree
    upwards, _, down = walker.walk(node1, node2)
    distance = len(upwards) + len(down)


    return 1 / (1 + distance)

def calculate_hierarchical_precision_recall(Y, Yhat, tree, species_names):
    weighted_true_positives = 0
    weighted_false_positives = 0
    weighted_false_negatives = 0

    for true_label, predicted_label in zip(Y, Yhat):

        if predicted_label is None:  # Handle negative prediction as complete miss
            weighted_false_negatives += 1  # Might need to adjust based on how you want to treat negative predictions
            continue
        if true_label is None:  # Handle missing ground truth as complete miss
            weighted_false_positives += 1
            continue

        node1 = find(tree, lambda node: node.name == species_names[true_label])
        node2 = find(tree, lambda node: node.name == species_names[predicted_label])
        similarity_weight = hierarchical_similarity(node1, node2, tree)

        if true_label == predicted_label:
            weighted_true_positives += similarity_weight
        else:
            weighted_false_positives += (1 - similarity_weight)  # This assumes you want to penalize based on dissimilarity

    precision = weighted_true_positives / (weighted_true_positives + weighted_false_positives) if (weighted_true_positives + weighted_false_positives) > 0 else 0
    recall = weighted_true_positives / (weighted_true_positives + weighted_false_negatives) if (weighted_true_positives + weighted_false_negatives) > 0 else 0
    
    return precision, recall

def calculate_weighted_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
