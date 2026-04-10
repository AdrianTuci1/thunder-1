class ContextShaper:
    """
    Segments external data into the hierarchical Macro/Meso/Micro structure
    required by the Thunder engine.
    """
    
    def __init__(self):
        pass

    def shape_context(self, external_data, tokenizer=None):
        """
        Formats raw external data into simple fixed-size blocks compatible with
        the current 2048-token project scope.
        """
        print("⚡ Thunder: Shaping external data into packed blocks...")

        estimated_tokens = int(len(external_data.split()) * 1.3)
        target_block_size = 2048
        estimated_blocks = max(1, (estimated_tokens + target_block_size - 1) // target_block_size)

        return {
            "raw_length": len(external_data),
            "estimated_tokens": estimated_tokens,
            "packing_plan": {
                "target_block_size": target_block_size,
                "estimated_blocks": estimated_blocks,
            },
            "metadata": {"source": "rag_search_agent"}
        }
