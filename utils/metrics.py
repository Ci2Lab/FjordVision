from anytree.search import find
from anytree.walker import Walker

def hierarchical_similarity(node1, node2):
    walker = Walker()
    upwards, common, down = walker.walk(node1, node2)
    
    # Since 'common' is not a list but a single node in anytree's Walker output,
    # we directly use it to find the depth of the least common ancestor.
    # If 'common' is None (which shouldn't normally happen if node1 and node2 are in the same tree),
    # we handle it by setting depth_lca to 0. Otherwise, we calculate its depth.
    if common:
        depth_lca = common.depth  # Use the .depth attribute of the common ancestor
    else:
        depth_lca = 0  # Fallback in case common is None, which is unusual for connected nodes
    
    # Adjust the calculation of the total depth and the WUP similarity accordingly
    total_depth = len(upwards) + len(down) + 2 * depth_lca
    similarity = (2.0 * depth_lca) / total_depth if total_depth > 0 else 1.0
    
    return similarity

def calculate_hierarchical_precision_recall(Y, Yhat, confidences, taxonomies, tree, threshold=0.3):
    weighted_true_positives = 0
    weighted_false_positives = 0
    weighted_false_negatives = 0
    
    for true_label, predicted_label, conf in zip(Y, Yhat, confidences):
        if predicted_label is None:  # Handle negative prediction as complete miss
            weighted_false_negatives += 1
            continue
        if true_label is None:  # Handle missing ground truth as complete miss
            weighted_false_positives += 1
            continue

        current_taxonomy = 0
        node = find(tree.root, lambda node: node.name == taxonomies[current_taxonomy][predicted_label])
        while conf < threshold and current_taxonomy < len(taxonomies) - 1:
            # Move up the taxonomy if the confidence is below the threshold
            if node.parent is not None:
                node = node.parent
                current_taxonomy += 1
                # Attempt to find the new predicted_label index in the parent taxonomy
                try:
                    predicted_label = taxonomies[current_taxonomy].index(node.name)
                except ValueError:
                    # If the node's name is not in the taxonomy, break from the loop
                    break
                conf += node.probability
            else:
                break  # If there's no parent, we're at the root and cannot go up further

        # At this point, node represents the current predicted label node
        node1 = node
        node2 = find(tree.root, lambda node: node.name == taxonomies[0][true_label])
        similarity_weight = hierarchical_similarity(node1, node2)

        if true_label == predicted_label:
            weighted_true_positives += similarity_weight
        else:
            weighted_false_positives += (1 - similarity_weight)  # Penalize based on dissimilarity

    precision = weighted_true_positives / (weighted_true_positives + weighted_false_positives) if (weighted_true_positives + weighted_false_positives) > 0 else 0
    recall = weighted_true_positives / (weighted_true_positives + weighted_false_negatives) if (weighted_true_positives + weighted_false_negatives) > 0 else 0
    
    return precision, recall

def calculate_weighted_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
