# Medguide AI

#### Setup 
```bash
cd ./backend

py -3.11 -m venv .venv

.\.venv\Scripts\activate

deactivate
```

### Deploy the site using gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app

sudo nano /etc/systemd/system/medguideai.service (paste code of server_setup/pysearch.service)

sudo systemctl daemon-reload

sudo systemctl restart medguideai

sudo systemctl enable medguideai

sudo systemctl status medguideai
```

### Installing SSL certificate
```bash
sudo apt install certbot python3-certbot-nginx

sudo certbot --nginx -d medguideai.mallickboy.com

sudo certbot certificates

sudo systemctl status certbot.timer
```

### Set-Up NGINX
```bash
sudo nano /etc/nginx/sites-available/medguideai.mallickboy.com   (paste code of server_setup/medguideai.mallickboy.com)
sudo nano /etc/nginx/sites-available/default  (paste code of default)

sudo ln -s /etc/nginx/sites-available/medguideai.mallickboy.com /etc/nginx/sites-enabled/

sudo nginx -t 

sudo systemctl reload nginx

sudo systemctl restart nginx
```