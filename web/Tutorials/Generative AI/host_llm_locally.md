---
comments: true
---

# Host LLM Locally

## Installing WSL

To get started, you'll need to install WSL. Run the following command in your Windows terminal:

```bash
wsl --install
```

This will install WSL and launch a new window with the Ubuntu distribution.

## Connecting to a WSL Instance

To connect to an existing WSL instance, run the following command:

```bash
wsl -d Ubuntu
```

## Installing Ollama

[Ollama](https://ollama.com/download) is a popular tool for working with local LLMs.

## Monitoring GPU Performance

Run the following command to view your GPU's performance:

```bash
watch -n 0.5 nvidia-smi
```

This will display an update every 0.5 seconds.

## Installing Docker

- Update your package list:

```bash
sudo apt-get update
```

- Install the necessary packages:

```bash
sudo apt-get install ca-certificates curl
```

- Install Docker's official GPG key:

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
```

- Add the Docker repository to your package list:

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list
```

- Update your package list:

```bash
sudo apt-get update
```

- Install Docker:

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## Running Open WebUI

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

## Accessing Open WebUI running on a WSL instance from outside the host computer

- Port forwarding using Powershell in Admin mode:

```bash
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=[WSL_IP]
```

- Retrieve **WSL_IP**, in a ubuntu terminal run this command:

```bash
ip addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}'
```

- Punch a hole in Firewall to allow this port to be reached from outside:

```bash
New-NetFirewallRule -DisplayName "Allow WSL2 Port" -Description "To allow Open WebUI through the firewall." -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080
```

## Installing Stable Diffusion

### Prerequisites

* **Pyenv**: Install Pyenv and its prerequisites using the following command:

```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git
```

* **Pyenv**: Run the following command to install Pyenv:

```bash
curl https://pyenv.run | bash
```

- Add to the Path:

```bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init --path)"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bashrc
```

- **Python 3.10**: Install Python 3.10 using Pyenv:

```bash
pyenv install 3.10
```

* **Make it global**: Make sure to use the global shell:

```bash
pyenv global 3.10
```

### Installing Stable Diffusion

1. Download the webui.sh script from [here](https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh), or by this command:

```
wget https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
```

4. Run the following command to make the script executable:

```bash
chmod +x webui.sh
```

4. Run the following command to start Stable Diffusion:

```bash
./webui.sh --listen --api
```

Note: This is just a basic setup guide and may require additional steps or configurations depending on your specific use case.

