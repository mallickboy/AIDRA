server {                                        
    listen 80;
    server_name medguideai.mallickboy.com;
    return 301 https://$host$request_uri;      
}

server {                                        
    listen 443 ssl http2;
    server_name medguideai.mallickboy.com;

    ssl_certificate /etc/letsencrypt/live/medguideai.mallickboy.com/fullchain.pem;    
    ssl_certificate_key /etc/letsencrypt/live/medguideai.mallickboy.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;              
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    location / {                               
        proxy_pass http://localhost:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
    }
}