---
layout:     post
title:      "阿里云Ubuntu16.04 上利用gunicorn+supervisor+nginx搭建详解"
date:       2016-11-25 12:00:00
author:     "Bigzhao"
header-img: "img/post-bg-04.jpg"
---
#ubuntu上搭建gunicorn+supervisor+nginx
---

① pip安装gunicorn
```bash
pip install gunicorn
```
关于gunicorn的启动命令是：
```bash
gunicorn -w 4 -b 0.0.0.0:7000 myapp: app
```
② pip 安装supervisor
```bash
sudo pip install supervisor
```
注意：需要在安装在sudo下
配置方式：
```bash
echo_ supervisord_conf > supervisor.conf
vim supervisor.conf
```
* 接下来需要在conf最后添加自己的app项目
```bash
[program: myapp]
command=/你的venv路径/gunicorn -w4 -b 0.0.0.0:7000 manage:app
directory=/home/ubuntu/tdz/flask-projects-manage
startsecs=0
stopwaitsecs=0
autostart=false
autorestart=false
stdout_logfile=/home/ubuntu/tdz/flask-projects-manage/log/gunicorn.log
stderr_logfile=/home/ubuntu/tdz/flask-projects-manage/log/gunicorn.err
```
* 最好把管理界面打开，即直接用http可视化管理，省的输命令：
```Bash
[inet_http_server]         ; inet (TCP) server disabled by default
port=127.0.0.1:9001        ; (ip_address:port specifier, *:port for all iface)
username=user              ; (default is no username (open server))
password=123               ; (default is no password (open server))
```
* 还有这个：
```bash
[supervisorctl]
serverurl=unix:///tmp/supervisor.sock ; use a unix:// URL  for a unix socket
serverurl=http://0.0.0.0:9001 ; use an http:// url to specify an inet socket
username=user              ; should be same as http_username if set
password=123                ; should be same as http_password if set
;prompt=mysupervisor         ; cmd line prompt (default "supervisor")
;history_file=~/.sc_history  ; use readline history if available
```
ps:上面两个取消注释就好，改一下自己的用户名密码即可，这个用户名是用来登录可视化管理的

③ 安装nginx
```bash    
sudo apt-get install nginx
```
然后配置文件/etc/nginx/nginx.conf,主要是将你所要的服务server给include进去,例如
```bash
include /etc/nginx/sites-enabled/nginx_gunicorn.conf;
include /etc/nginx/sites-enabled/nginx_django_gunicorn.conf;
```
因为我有两个服务需要监听
其中一个conf内容是:
```bash
server{
    listen 0.0.0.0:8080;
    location /static/  {
        include /etc/nginx/mime.types;
        # Example:
        # root /full/path/to/application/static/file/dir;
        root /home/ubuntu/tdz/flask-projects-manage/app/;

        }
    location = /favicon.ico  {

        root /home/ubuntu/tdz/flask-projects-manage/app/static/f.ico;

    }

    location / {
        proxy_pass http://127.0.0.1:7000;
        proxy_redirect off;
        proxy_set_header Host $host:8080;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```
最后记得在supervisor.conf下加上nginx项目
```bash
 [program:nginx]
 command=/usr/sbin/nginx
 startsecs=0
 stopwaitsecs=0
 autostart=false
 autorestart=false
 stdout_logfile=/home/ubuntu/tdz/flask-projects-manage/log/nginx.log
 stderr_logfile=/home/ubuntu/tdz/flask-projects-manage/log/nginx.err
```
---
好了，部署完了，进入9001端口启动即可。

```
