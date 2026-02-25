from core.config_manager import THUNDER_CONFIG

class ThunderRouter:
    """
    SMART GATING: Decides whether to handle a query internally.
    """
    
    def __init__(self, internal_threshold=None):
        self.internal_threshold = internal_threshold or THUNDER_CONFIG["logic"]["internal_threshold"]

    def route_query(self, query, forced_mode=None):
        """
        Analyzes the query and routes it to the appropriate engine.
        Now supports forced modes: instant, fast, thinking.
        """
        confidence = self._estimate_internal_capability(query)
        
        # Heuristic for intensity and length
        is_complex = any(k in query.lower() for k in ["how", "why", "analyze", "explain", "compare"])
        is_short = any(k in query.lower() for k in ["who", "what is", "where", "list", "direct"])
        
        intensity = 0.9 if is_complex else 0.4
        predicted_length = 500 if is_complex else (100 if is_short else 250)
        
        # Determine mode if not forced
        if not forced_mode:
            if intensity > 0.8:
                suggested_mode = "thinking"
            elif intensity > 0.5:
                suggested_mode = "fast"
            else:
                suggested_mode = "instant"
        else:
            suggested_mode = forced_mode

        route = {
            "target": "INTERNAL_THUNDER" if confidence >= self.internal_threshold else "EXTERNAL_SEARCH",
            "confidence": confidence,
            "mode": suggested_mode,
            "intensity": intensity,
            "predicted_length": predicted_length
        }
        
        return route

    def _estimate_internal_capability(self, query):
        """
        Uses early-exit or fast-pass model to check if the context suffices.
        """
        # Mock confidence check
        return 0.9
