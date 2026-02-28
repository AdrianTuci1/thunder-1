# Connecting Thunder to Open WebUI

Thunder provides an OpenAI-compatible API that makes integration with Open WebUI seamless.

## 1. Start the Thunder Server
Ensure your server is running on your VPS or RunPod:
```bash
# From the project root
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 2. Configure Open WebUI
1. Log in to your **Open WebUI** interface.
2. Navigate to **Settings** > **Connections**.
3. Under the **OpenAI API** section:
   - **Base URL**: Set this to `http://<YOUR_VPS_IP>:8000/v1`
   - **API Key**: You can enter any string (e.g., `thunder-secret`), as the current engine doesn't enforce key validation.
4. Click **Save** (Refresh if needed).

## 3. Usage
- In the chat interface, select the **thunder-v1** model from the dropdown.
- Thunder will now handle your queries using its hierarchical crystallization logic.

## 4. Advanced: Glitch Effect via Filters
Since Thunder sends the final response directly to Open WebUI, you can simulate the "crystallization" effect using an Open WebUI **Filter** (Functions).
- Create a new Function that targets `thunder-v1`.
- Add a CSS animation that applies a brief `blur` or `character-scramble` effect to new message chunks.
