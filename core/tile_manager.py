import math

from core.config_manager import THUNDER_CONFIG

class ThunderTileNode:
    """Represents a node in the Thunder Parallel Tree."""
    def __init__(self, start, end, depth, id_prefix=""):
        self.range = (start, end)
        self.depth = depth
        self.id = f"{id_prefix}{depth}_{start}_{end}"
        self.children = []
        self.is_leaf = False

class FractalTiler:
    """
    Implements recursive tree-based tiling for extreme parallel processing.
    Segments context into a massively parallel tree structure.
    """
    
    def __init__(self, meso_size=None, micro_size=None):
        self.max_depth = THUNDER_CONFIG["engine"]["max_tree_depth"]
        self.sub_factor = THUNDER_CONFIG["engine"]["subdivision_factor"]
        self.micro_size = THUNDER_CONFIG["engine"]["micro_size"]

    def build_tile_tree(self, input_length):
        """Builds a recursive tree of context tiles."""
        root = ThunderTileNode(0, input_length, depth=0, id_prefix="root_")
        self._subdivide(root)
        return root

    def _subdivide(self, node):
        """Recursively splits a tile based on subdivision_factor."""
        start, end = node.range
        length = end - start
        
        # Stop if we hit depth limit or minimum parallelizable size
        if node.depth >= self.max_depth or length <= self.micro_size * 1.5:
            node.is_leaf = True
            return

        chunk_size = math.ceil(length / self.sub_factor)
        overlap = THUNDER_CONFIG["engine"]["overlap_size"]

        for i in range(self.sub_factor):
            c_start = start + i * chunk_size
            c_end = min(c_start + chunk_size + overlap, end)
            
            if c_start < end:
                child = ThunderTileNode(c_start, c_end, node.depth + 1, id_prefix=f"sub_{i}_")
                node.children.append(child)
                self._subdivide(child)

    def get_tiling_plan(self, input_length):
        """Legacy wrapper returning the tree root."""
        return self.build_tile_tree(input_length)
