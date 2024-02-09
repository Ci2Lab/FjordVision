from anytree import Node, PreOrderIter, RenderTree
from anytree.importer import JsonImporter
from collections import defaultdict

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