# Cloud Deploy

This bot now supports both:

- a UI-less long-running mode
- a `run once and exit` mode for scheduled execution

## Best Option on Google Cloud

If you only need the bot to work once right after each 30m candle closes, then a **scheduled run** is usually better than a permanently running process.

For your current strategy, that is a valid approach because:

- the model decision only changes on closed 30m candles
- TP/SL orders are placed on Binance itself
- the process does not need to watch ticks continuously between candles

So the most practical setup is:

- a small **Compute Engine Ubuntu VM**
- a cron job that runs every 30 minutes
- the bot starts, checks the latest closed bar, acts once, then exits

For this project, the simplest and most reliable option is a standard **Compute Engine Ubuntu VM**.

Do not use Cloud Run for this bot:

- this bot is a long-running background process
- it needs to stay alive between requests
- it should keep polling and trading even when nobody is visiting a page

Do not use Spot VM for your main bot:

- Spot VMs can be preempted by Google Cloud at any time
- that is fine for batch jobs, but not ideal for a trading bot that should stay online 24/7

Recommended starting point:

- machine type: `e2-standard-2`
- boot disk: `Ubuntu 24.04 LTS`
- disk: `30 GB`
- provisioning model: `Standard`
- allow HTTP/HTTPS: not required unless you want the dashboard remotely

## Google Cloud Step By Step

### 1. Create the VM

In Google Cloud Console:

1. Open **Compute Engine** -> **VM instances**
2. Click **Create instance**
3. Choose:
   - name: `btc-trade-bot`
   - region: a region close to you
   - machine: `e2-standard-2`
   - boot disk: `Ubuntu 24.04 LTS`
   - boot disk size: `30 GB`
   - provisioning model: `Standard`
4. Leave a public IP enabled
5. Create the VM

### 2. Connect to the VM

Use the **SSH** button in Google Cloud Console on the VM row.

### 3. Install Python and venv

Run:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### 4. Upload the project

Either:

- push this repo to GitHub and clone it on the VM
- or copy the project folder with `gcloud compute scp`

Example with git:

```bash
cd /opt
sudo git clone YOUR_REPO_URL btc_trade_bot
sudo chown -R $USER:$USER /opt/btc_trade_bot
cd /opt/btc_trade_bot/BTC_trade_bot
```

### 5. Create the virtual environment

```bash
cd /opt/btc_trade_bot
python3 -m venv venv
source venv/bin/activate
cd /opt/btc_trade_bot/BTC_trade_bot
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Create the env file

```bash
cp deploy/systemd/btc_trade_bot.env.example /opt/btc_trade_bot/.env
nano /opt/btc_trade_bot/.env
```

Fill in at least:

```bash
BINANCE_DEMO_API_KEY=YOUR_KEY
BINANCE_DEMO_API_SECRET=YOUR_SECRET
BTC_BOT_SYMBOL=BTCUSDT
BTC_BOT_LEVERAGE=5
BTC_BOT_ACCOUNT_RISK_PER_TRADE=0.03
BTC_BOT_STOP_LOSS_FACTOR=0.75
BTC_BOT_POLL_INTERVAL_SECONDS=15
BTC_BOT_KLINE_HISTORY_BARS=3400
```

### 7. Test once manually

```bash
cd /opt/btc_trade_bot
source venv/bin/activate
cd /opt/btc_trade_bot/BTC_trade_bot
python live_trading_bot.py --smoke-test
python live_trading_bot.py --headless --risk low
```

Stop it with `Ctrl+C` after confirming it starts.

### 8. Install as a service

```bash
sudo cp deploy/systemd/btc_trade_bot.service.example /etc/systemd/system/btc_trade_bot.service
sudo nano /etc/systemd/system/btc_trade_bot.service
```

Make sure these lines match your VM paths:

```ini
WorkingDirectory=/opt/btc_trade_bot/BTC_trade_bot
EnvironmentFile=/opt/btc_trade_bot/.env
ExecStart=/opt/btc_trade_bot/venv/bin/python /opt/btc_trade_bot/BTC_trade_bot/live_trading_bot.py --headless --risk low
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable btc_trade_bot
sudo systemctl start btc_trade_bot
```

### 9. Verify it survives logout and local PC shutdown

Check status:

```bash
sudo systemctl status btc_trade_bot
sudo journalctl -u btc_trade_bot -f
```

If the service is `active (running)`, then it continues running on the VM even if:

- you close SSH
- you shut down your own computer
- you disconnect from the internet

The bot only stops if:

- the VM is stopped
- the service crashes and cannot restart
- you stop the service manually

### 10. Optional: open the dashboard remotely

If you want the dashboard from your browser:

```bash
cd /opt/btc_trade_bot
source venv/bin/activate
cd /opt/btc_trade_bot/BTC_trade_bot
python live_trading_bot.py --host 0.0.0.0 --port 8080 --no-browser
```

Then create a firewall rule in Google Cloud for TCP `8080`, and open:

```text
http://YOUR_VM_EXTERNAL_IP:8080
```

## Headless Run

From the project directory:

```bash
python live_trading_bot.py --headless --risk low
```

## Run Once And Exit

This mode is ideal for cron or a scheduler:

```bash
python live_trading_bot.py --run-once --risk low
```

The bot writes the last processed candle timestamp to a small state file so the same candle is not processed twice accidentally:

```bash
python live_trading_bot.py --run-once --risk low --state-file /opt/btc_trade_bot/runtime/last_processed_bar.json
```

You can also override credentials directly:

```bash
python live_trading_bot.py --headless --risk high --api-key YOUR_KEY --api-secret YOUR_SECRET
```

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Recommended Environment Variables

Copy `deploy/systemd/btc_trade_bot.env.example` to `.env` and fill in your values.

Important variables:

- `BINANCE_DEMO_API_KEY`
- `BINANCE_DEMO_API_SECRET`
- `BTC_BOT_LEVERAGE`
- `BTC_BOT_ACCOUNT_RISK_PER_TRADE`
- `BTC_BOT_STOP_LOSS_FACTOR`

## Recommended For You: cron On A VM

After setup, run:

```bash
crontab -e
```

Then add this line:

```bash
1,31 * * * * cd /opt/btc_trade_bot/BTC_trade_bot && /opt/btc_trade_bot/venv/bin/python live_trading_bot.py --run-once --risk low --state-file /opt/btc_trade_bot/runtime/last_processed_bar.json >> /var/log/btc_trade_bot.log 2>&1
```

This means:

- at minute `01` and `31` of every hour
- the bot runs once
- it waits until just after the candle close
- logs go to `/var/log/btc_trade_bot.log`

This is usually better than exactly `00` and `30`, because exchanges may still be finalizing the just-closed candle for a few seconds.

Useful checks:

```bash
crontab -l
tail -f /var/log/btc_trade_bot.log
```

## systemd Example

If you still want a continuously running process, you can use `systemd`.

1. Copy the repo to the server, for example under `/opt/btc_trade_bot`.
2. Create `/opt/btc_trade_bot/.env` from `deploy/systemd/btc_trade_bot.env.example`.
3. Update paths and user in `deploy/systemd/btc_trade_bot.service.example`.
4. Copy it to `/etc/systemd/system/btc_trade_bot.service`.
5. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable btc_trade_bot
sudo systemctl start btc_trade_bot
```

Useful commands:

```bash
sudo systemctl status btc_trade_bot
sudo journalctl -u btc_trade_bot -f
tail -f /var/log/btc_trade_bot.log
```

## Cost Reality

This is not guaranteed to be completely free.

The closest thing to "almost free" is:

- a very small Compute Engine VM
- preferably one that fits inside Google Cloud free-tier rules if your region and machine type qualify

But note:

- Google free-tier rules can change
- eligible regions are limited
- if you pick a non-free machine or region, you pay
- disk, IP, network egress, and logs can also create cost

So the correct expectation is:

- **scheduled VM run is much cheaper than 24/7 runtime**
- **it may be near-zero cost**
- **it is not something I would promise as permanently 100% free**

## Billing And Invoice Control

You said you want to check from time to time whether you are going over the limit. Do these 3 things on day one:

### 1. Create a budget alert

In Google Cloud Console:

1. Open **Billing**
2. Open **Budgets & alerts**
3. Click **Create budget**
4. Scope it to the project that contains your VM
5. Set a low amount, for example:
   - budget: `5 USD`
6. Add alert thresholds:
   - `50%`
   - `90%`
   - `100%`
7. Enable email alerts
8. Save

Important:

- a budget does **not** automatically stop the VM
- it only warns you

### 2. Check the live billing reports

In Google Cloud Console:

1. Open **Billing**
2. Open **Reports**
3. Set the date range to **This month**
4. Filter to your project
5. Filter to **Compute Engine**

This page is useful to quickly see:

- whether cost is still near zero
- which service is generating cost
- whether the VM or disk is the source

### 3. Check the detailed cost table

In Google Cloud Console:

1. Open **Billing**
2. Open **Cost table**
3. Filter by your project
4. Group by:
   - `Project`
   - `Service`
   - `SKU`

This is the best place to answer:

- "Did the VM cost money?"
- "Did the disk cost money?"
- "Did network egress cost money?"

### Suggested routine

After the first deployment:

- check Billing -> Reports after 1 day
- check again after 3 days
- check again after 7 days

If the amount still looks near zero, then your setup is probably staying inside free-tier or near-free usage.

### If you see cost increasing

Do these checks immediately:

1. Verify the VM region is one of the free-tier eligible regions
2. Verify the machine type is still small enough
3. Check whether your persistent disk is larger than expected
4. Check whether you created a static external IP that is billed
5. Check whether network egress is growing

Useful official billing docs:

- budgets and alerts: https://cloud.google.com/billing/docs/how-to/budgets
- billing reports: https://cloud.google.com/billing/docs/how-to/reports
- cost table: https://cloud.google.com/billing/docs/how-to/cost-table

## UI Mode

For demos, the dashboard still exists:

```bash
python live_trading_bot.py --host 0.0.0.0 --port 8080 --no-browser
```

Then open the server IP with port `8080`.
