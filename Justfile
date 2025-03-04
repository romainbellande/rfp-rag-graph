set dotenv-load := true
set dotenv-required := true
set quiet := true

up:
    docker compose up -d

down:
    docker compose down

studio:
    langgraph dev

deepseek:
    docker compose run ollama bash -c "ollama run deepseek:r1"