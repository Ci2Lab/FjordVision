from anytree import PreOrderIter, RenderTree
from anytree.importer import JsonImporter
from collections import defaultdict
from anytree import findall_by_attr

class ProbabilityTree:
    def __init__(self, ontology_path):
        self.importer = JsonImporter()
        with open(ontology_path, 'r') as f:
            self.root = self.importer.read(f)
        self.assign_uniform_probabilities()

    def assign_uniform_probabilities(self):
        # Calculate the number of nodes per rank
        nodes_per_rank = defaultdict(int)
        for node in PreOrderIter(self.root):
            if hasattr(node, 'rank'):
                nodes_per_rank[node.rank] += 1

        # Assign uniform probability based on rank
        for node in PreOrderIter(self.root):
            if hasattr(node, 'rank'):
                total_nodes = nodes_per_rank[node.rank]
                node.probability = 1 / total_nodes if total_nodes else 0

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