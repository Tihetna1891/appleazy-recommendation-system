#!/bin/bash
cd /root/appleazy-recommendation-system
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
pm2 restart appleazy-recommendation-system
