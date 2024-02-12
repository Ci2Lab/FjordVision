from anytree import PreOrderIter, RenderTree
from anytree.importer import JsonImporter
from collections import defaultdict
from anytree import findall_by_attr

class ProbabilityTree:
    def __init__(self, ontology_path):
        self.importer = JsonImporter()
        with open(ontology_path, 'r') as f:
            self.root = self.importer.read(f)
        self.assign_uniform_probabilities_to_leaves_and_propagate_up()

    def assign_uniform_probabilities_to_leaves_and_propagate_up(self):
        # Step 1: Assign uniform probabilities to leaves
        leaves = [node for node in PreOrderIter(self.root) if node.is_leaf]
        uniform_probability = 1 / len(leaves) if leaves else 0
        for leaf in leaves:
            leaf.probability = uniform_probability
        
        # Step 2: Propagate probabilities up the tree
        for node in reversed(list(PreOrderIter(self.root))): # Process nodes from bottom to top
            if not node.is_leaf:
                node.probability = sum(child.probability for child in node.children if hasattr(child, 'probability'))

    def print_tree(self):
        for pre, _, node in RenderTree(self.root):
            treestr = u"%s%s" % (pre, node.name)
            print(f"{treestr} (Rank: {getattr(node, 'rank', 'N/A')}, Probability: {getattr(node, 'probability', 'N/A')})")

    def sum_siblings_probabilities(self, node_name):
            # Find the node by name
            nodes = findall_by_attr(self.root, name="name", value=node_name)
            if not nodes:
                print(f"No node found with the name {node_name}.")
                return 0
            node = nodes[0]  # Assuming unique names, take the first match

            # If the node is the root, it has no siblings
            if not node.parent:
                print(f"Node {node_name} is the root and has no siblings.")
                return node.probability if hasattr(node, 'probability') else 0

            # Calculate the sum of probabilities of all siblings (including the node itself)
            siblings_probability_sum = sum(sibling.probability for sibling in node.parent.children if hasattr(sibling, 'probability'))

            return siblings_probability_sum

    def update_probabilities_with_instance_counts(self, instance_counts):
        # instance_counts is a dictionary where keys are node names and values are the instance counts
        
        # Step 1: Update leaf node probabilities based on instance counts
        for node in PreOrderIter(self.root):
            if node.is_leaf:
                if node.name in instance_counts:
                    node.instance_count = instance_counts[node.name]
                else:
                    node.instance_count = 0  # Default to 0 if not specified

        # Calculate total instances for normalization
        total_instances = sum(node.instance_count for node in PreOrderIter(self.root) if node.is_leaf)

        # Assign probabilities to leaf nodes based on their instance count
        for node in PreOrderIter(self.root):
            if node.is_leaf:
                node.probability = node.instance_count / total_instances if total_instances else 0
        
        # Step 2: Propagate probabilities upwards
        for node in PreOrderIter(self.root):
            if not node.is_leaf:
                # Sum the probabilities of child nodes
                node.probability = sum(child.probability for child in node.children)

        # Note: This assumes the instance counts fully define the distribution at the lowest level
        # and that each non-leaf node's probability is the sum of its children's probabilities.
