class ThunderPersonality:
    """
    Handles system prompt framing and Gemini-style alignment.
    Ensures the model maintains a premium, helpful, and concise persona.
    """
    
    def __init__(self, style="GEMINI"):
        self.style = style

    def get_system_prompt(self):
        """
        Returns the framing prompt based on the engine's core capabilities.
        """
        return {
            "role": "system",
            "content": (
                "You are Thunder, a high-performance inference engine by Static Labs. "
                "Utilizing Hierarchical Parallel Diffusion, you provide instant, "
                "coherent answers across massive context windows. "
                "Be direct, expert-level, and aesthetically sophisticated in your output."
            )
        }

    def apply_formatting(self, response):
        """
        Applies stylistic formatting to the output.
        """
        return response.strip()
