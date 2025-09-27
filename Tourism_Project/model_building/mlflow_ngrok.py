
from pyngrok import ngrok, conf
import subprocess
import os

# Configure ngrok binary (adjust path if needed)
conf.get_default().ngrok_path = r"C:\Program Files\WindowsApps\ngrok.ngrok_3.24.0.0_x64__1g87z0zv29zzc\ngrok.exe"

# Set your ngrok auth token (comes from GitHub secret)
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# Start MLflow UI on port 5000
process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Expose MLflow UI via ngrok
public_url = ngrok.connect(5000).public_url
print("MLflow UI is available at:", public_url)

# Save the URL to a file so GitHub Actions can use it
with open("mlflow_url.txt", "w") as f:
    f.write(public_url)
from pyngrok import ngrok, conf
import subprocess
import os

# Configure ngrok binary (adjust path if needed)
conf.get_default().ngrok_path = r"C:\Program Files\WindowsApps\ngrok.ngrok_3.24.0.0_x64__1g87z0zv29zzc\ngrok.exe"

# Set your ngrok auth token (comes from GitHub secret)
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# Start MLflow UI on port 5000
process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Expose MLflow UI via ngrok
public_url = ngrok.connect(5000).public_url
print("MLflow UI is available at:", public_url)

# Save the URL to a file so GitHub Actions can use it
with open("mlflow_url.txt", "w") as f:
    f.write(public_url)

