class ContextShaper:
    """
    Segments external data into the hierarchical Macro/Meso/Micro structure
    required by the Thunder engine.
    """
    
    def __init__(self):
        pass

    def shape_context(self, external_data, tokenizer=None):
        """
        Formats raw external data into the fractal tile hierarchy.
        Connects external RAG data to the core inference engine.
        """
        print("⚡ Thunder: Shaping external data into fractal tiles...")
        
        # We leverage the core FractalTiler for consistency
        from core.tile_manager import FractalTiler
        tiler = FractalTiler()
        
        # Estimate token length (visualized for skeleton)
        token_count = len(external_data.split()) * 1.3 
        plan = tiler.get_tiling_plan(int(token_count))
        
        return {
            "raw_length": len(external_data),
            "estimated_tokens": int(token_count),
            "tiling_plan": plan,
            "metadata": {"source": "rag_search_agent"}
        }
