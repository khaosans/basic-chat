FROM ghcr.io/khaosans/basic-chat-base:latest

WORKDIR /app
COPY . .

CMD ["python3"] 