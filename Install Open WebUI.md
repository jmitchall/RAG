# Install Open WebUI

https://robert-mcdermott.medium.com/performance-vs-practicality-a-comparison-of-vllm-and-ollama-104acad250fd

0. Open WSL Ubuntu `"C:\Program Files\WSL\wsl.exe" --distribution-id {439a78b3-9763-443a-9526-5ec80fb9bee7} --cd ~`  
1. mkdir open-webui-srv
2. cd open-webui-srv
3. uv venv --python 3.12 --seed
4. uv pip install open-webui

# Start Open WebUI


0. Open WSL Ubuntu `"C:\Program Files\WSL\wsl.exe" --distribution-id {439a78b3-9763-443a-9526-5ec80fb9bee7} --cd ~`  
1. cd open-webui-srv
2. ⚠️ Note: The WEBUI_AUTH=False part of the above command sets an environment variable that tells Open WebUI to disable user authentication. By default, Open WebUI is a multi-user web application that requires user accounts and authentication, but we are just setting it up for personal use, so we are disabling the user authentication layer.

        WEBUI_AUTH=False uv run open-webui serve 
3. Open Browser to http://127.0.0.1:8080
4. Lower Left Hand Corner goto User -> Admin Panel -> Settings -> Connections -> OPENAI API
5. Add http://localhost:8000/v1 with a random api key
