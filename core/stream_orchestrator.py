import torch

class StreamOrchestrator:
    """
    Manages asynchronous CUDA streams for parallel processing of micro-tiles.
    Goal: Saturate the RTX 4090's Tensor cores through massive parallelism.
    """
    
    def __init__(self, stream_count=64):
        self.stream_count = stream_count
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(stream_count)]
        else:
            self.streams = [None] * stream_count
        self.stream_pool = list(range(stream_count))

    def execute_tree_parallel(self, root_node, func):
        """
        Executes a function across the entire tile tree in parallel.
        Uses a breadth-first pool to saturate CUDA streams.
        """
        from collections import deque
        queue = deque([root_node])
        results = {}
        
        while queue:
            # Process current level in parallel branches
            level_nodes = []
            while queue:
                level_nodes.append(queue.popleft())
            
            # Saturate streams with current level branches
            level_results = self.execute_parallel(level_nodes, func)
            
            for node, res in zip(level_nodes, level_results):
                results[node.id] = res
                # Add children to queue for next level processing
                if hasattr(node, 'children'):
                    queue.extend(node.children)
                    
        return results

    def execute_parallel(self, tiles, func):
        """
        Executes a function across multiple tiles using available CUDA streams.
        Ensures maximum saturation of RTX 4090.
        """
        if not torch.cuda.is_available():
            return [func(tile) for tile in tiles]
            
        results = []
        events = []
        
        for i, tile in enumerate(tiles):
            stream = self.streams[i % self.stream_count]
            
            with torch.cuda.stream(stream):
                result = func(tile)
                results.append(result)
                
                # Record event for fine-grained sync
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
        
        # Wait for all scheduled work to complete
        for event in events:
            event.wait()
            
        return results

    def get_stream(self, index):
        """Returns the specific stream for a given hardware block."""
        return self.streams[index % self.stream_count]

    def crystallize_global_field(self, tiles_data, diffusion_engine, total_steps=50):
        """
        Executes the global synchronized reverse diffusion loop across all tiles.
        tiles_data is a list of dicts: {'id': id, 'latent': tensor, 'macro_context': tensor, 'anchor_data': tensor}
        """
        print(f"⚡ Thunder: Starting Global Field Crystallization ({len(tiles_data)} tiles, {total_steps} steps)...")
        
        for step_idx in range(total_steps):
            
            # 1. Define the single-step computation for a tile
            def step_func(tile):
                return diffusion_engine.process_single_step(
                    x=tile['latent'],
                    step_idx=step_idx,
                    total_steps=total_steps,
                    macro_context=tile.get('macro_context'),
                    anchor_data=tile.get('anchor_data')
                )
            
            # 2. Execute one step in parallel across all streams
            results = self.execute_parallel(tiles_data, step_func)
            
            # 3. Update latent fields with the new partially denoised state
            for i, result_latent in enumerate(results):
                tiles_data[i]['latent'] = result_latent
                
            # 4. Perform Global Field Fusion (Overlap Synchronization)
            # This extracts the boundary of tile i and sets it as the anchor for tile i+1 in the next step
            if step_idx < total_steps - 1:
                self._synchronize_overlaps(tiles_data, diffusion_engine.fuser)
            
        print("⚡ Thunder: Global Field Crystallization Complete.")
        return [t['latent'] for t in tiles_data]

    def _synchronize_overlaps(self, tiles_data, fuser):
        """
        Extracts boundaries after a step and sets them as anchors for the next step.
        Ensures perfect continuity across the 16k latent field.
        """
        overlap_size = fuser.overlap_size
        
        # Propagate boundary i -> anchor for i+1
        for i in range(len(tiles_data) - 1):
            left_tile_latent = tiles_data[i]['latent']
            
            # The boundary is the last 'overlap_size' elements of left_tile's sequence length
            # latent shape: [B, L, D]
            boundary_latent = left_tile_latent[:, -overlap_size:, :].detach().clone()
            
            # Set this as the anchor for the right tile (i+1) for the next step
            tiles_data[i+1]['anchor_data'] = boundary_latent
