# Oracle Cloud Free Tier Deployment Guide

This guide provides a step-by-step process for deploying your `ml-tradingbot` to an Oracle Cloud **Always Free** instance. Using Oracle Cloud will allow your bot to run 24/7 with zero hosting costs.

## Step 1: Create an Oracle Cloud Account

1. Visit the [Oracle Cloud Free Tier sign-up page](https://www.oracle.com/cloud/free/).
2. Create an account. You will need to provide a credit/debit card for verification (you will not be charged unless you manually upgrade to a paid account).
3. Once your account is verified and ready, log in to your Oracle Cloud Console.

## Step 2: Provision the VM Instance

1. From the Oracle Cloud Console dashboard, click **Create a VM instance**.
2. **Name your instance**: e.g., `ml-tradingbot-vm`.
3. **Placement & Shape**:
   - **Image**: Click "Edit" and change the image to **Ubuntu 22.04** or **Ubuntu 24.04** (Always Free Eligible).
   - **Shape**: Change the shape. Go to "Virtual machine" > "Ampere" and select **VM.Standard.A1.Flex**.
     > [!TIP]
     > The ARM A1.Flex shape allows up to **4 OCPUs and 24 GB of RAM** for free. This is highly recommended over the Micro instance because your bot uses machine learning (Random Forest, pandas), which benefits heavily from the extra CPU power and memory. Choose 4 OCPUs and 24 GB RAM.
4. **Networking**: Leave the default VCN settings (it should create a new Virtual Cloud Network automatically with a public subnet). Keep "Assign a public IPv4 address" **checked**.
5. **Add SSH keys**:
   - Select **"Generate a key pair for me"** and click **"Save private key"**. (Keep this `ssh-key-*.key` file in a safe place, you need it to log in).
6. **Boot volume**: Leave default settings (usually 50 GB is standard for Always Free).
7. Click **Create** at the bottom. Wait a few minutes until the instance state changes from _Provisioning_ to **Running**.
8. Note down the **Public IP Address** displayed on the instance details page.

## Step 3: Connect to your Instance

Open your computer's terminal (since you are on a Mac, just open the native `Terminal` app) and perform the following:

1. **Change permissions of your downloaded private key** (SSH requires keys to be secure):
   ```bash
   chmod 400 ~/Downloads/ssh-key-*.key
   ```
2. **Connect to the server via SSH**:
   ```bash
   ssh -i ~/Downloads/ssh-key-*.key ubuntu@<YOUR_PUBLIC_IP_ADDRESS>
   ```
   _(Type `yes` when asked if you want to continue connecting)._

---

## Step 4: Install System Dependencies

Once logged into the server, update the packages, add the Python repository to get **Python 3.11**, and install essential build tools (important for `scipy` and `scikit-learn` on ARM instances):

```bash
sudo apt update && sudo apt upgrade -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install build-essential python3.11 python3.11-venv python3.11-dev python3-pip git tmux htop -y
```

## Step 5: Clone Your Repository

Your code is now on GitHub, so you can easily pull it down to your VM:

```bash
# Clone the repository
git clone https://github.com/kingym88/ml-tradingbot.git

# Enter the directory
cd ml-tradingbot
```

## Step 6: Set Up Python Environment & the Bot

Avoid installing packages globally by creating a virtual environment.

1. **Create and activate a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
2. **Run your setup script**:
   ```bash
   bash setup.sh
   ```
   _Note: This script will install Python dependencies ([requirements.txt](file:///Users/kingmurekio/Documents/ml-tradingbot/requirements.txt)), create the required folders (`data`, `models`, `logs`), create a template `.env` file, and collect the initial historical data._

## Step 7: Configure Environment Variables

You need to pass your Hyperliquid keys and specific environment configurations from your local Mac over to the Ubuntu server.

1. Open the `.env` file to edit it:
   ```bash
   nano config/.env
   ```
2. Paste in your actual `HYPERLIQUID_WALLET_ADDRESS` and `HYPERLIQUID_PRIVATE_KEY` (and any other keys you were using locally).
3. **Save and Exit** `nano` by pressing `Ctrl + X`, then type `Y`, and press `Enter`.

## Step 8: Run the Bot 24/7 (Using tmux)

If you run the bot normally and close your SSH connection (close your laptop), the bot will stop. To keep it running securely in the background, we use `tmux`.

1. **Start a new background session**:
   ```bash
   tmux new -s tradingbot
   ```
2. **Activate your virtual environment and start the bot**:
   ```bash
   source venv/bin/activate
   python main.py
   ```
   _(If you want to just train first without trading, use `python main.py --train-only` first)._
3. **Detach from the session** (Leave it running in the background):
   Press `Ctrl + B`, let go of both keys, then press `D`.
4. You can now safely close your terminal or log out of SSH.

### How to Monitor Your Bot Later

When you want to check on your bot again:

1. Reconnect via SSH to your server.
2. Reattach to the tmux session:
   ```bash
   tmux attach -t tradingbot
   ```
3. To detach again, press `Ctrl + B` then `D`.

---

> [!TIP]
> **Helpful Commands for maintenance:**
>
> - To check system memory/CPU usage: `htop`
> - To fetch latest code updates from GitHub in the future: `git pull origin main`
> - To read the latest bot logs (even while running): `tail -f logs/trading_bot.log`
