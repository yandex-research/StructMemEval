#!/usr/bin/env python3
"""
Интерактивный чат с моделью через терминал.
Конфигурация загружается из config.yaml (рядом со скриптом).
Поддерживает историю сообщений, команду /reset для сброса контекста и /exit для выхода.
"""

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from httpx import Client
from openai import OpenAI
from openai._base_client import DEFAULT_TIMEOUT, DEFAULT_CONNECTION_LIMITS


def load_config(config_path: str) -> dict:
    """Загружает YAML-конфиг с подстановкой переменных окружения."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_str = f.read()

    load_dotenv()  # загружаем переменные из .env, если есть
    config_str = os.path.expandvars(config_str)
    return yaml.safe_load(config_str)


def create_openai_client(llm_config: dict) -> OpenAI:
    """
    Создаёт клиент OpenAI с кастомным HTTPX-клиентом.
    Параметры verify=False, таймауты и лимиты берутся из примера.
    """
    http_client = Client(
        verify=False,
        timeout=DEFAULT_TIMEOUT,
        limits=DEFAULT_CONNECTION_LIMITS,
        follow_redirects=True
    )

    return OpenAI(
        api_key=llm_config['api_key'],
        base_url=llm_config.get('openrouter_base_url'),  # может отсутствовать
        http_client=http_client
    )


def main():
    # Определяем путь к конфигу (рядом со скриптом)
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"

    if not config_path.exists():
        print(f"Ошибка: файл конфигурации {config_path} не найден.")
        sys.exit(1)

    # Загружаем конфигурацию
    config = load_config(str(config_path))

    # Извлекаем настройки LLM (секция llm, как в benchmark)
    try:
        llm_config = config['mem0']['llm']
    except KeyError:
        print("Ошибка: в конфиге отсутствует секция 'llm'.")
        sys.exit(1)

    # Проверяем обязательные поля
    required = ['api_key', 'model']
    for field in required:
        if field not in llm_config:
            print(f"Ошибка: в секции llm отсутствует поле '{field}'.")
            sys.exit(1)

    # Создаём клиента
    client = create_openai_client(llm_config)

    # Системное сообщение (можно задать в конфиге или оставить пустым)
    system_message = llm_config.get('system_prompt', 'You are a helpful assistant.')

    # История сообщений для поддержания контекста
    messages = [{"role": "system", "content": system_message}]

    print("\n=== Интерактивный чат с моделью ===")
    print(f"Модель: {llm_config['model']}")
    print("Команды: /reset — сбросить историю, /exit — выйти\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not user_input:
            continue

        # Обработка команд
        if user_input.lower() == "/exit":
            print("Выход.")
            break
        elif user_input.lower() == "/reset":
            messages = [{"role": "system", "content": system_message}]
            print("История сброшена.\n")
            continue

        # Добавляем сообщение пользователя в историю
        messages.append({"role": "user", "content": user_input})

        try:
            # Отправляем запрос к модели
            response = client.chat.completions.create(
                model=llm_config['model'],
                messages=messages,
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 1000)
            )
        except Exception as e:
            print(f"Ошибка при обращении к API: {e}")
            # Убираем последнее сообщение пользователя, чтобы не нарушать историю
            messages.pop()
            continue

        # Извлекаем ответ ассистента
        assistant_reply = response.choices[0].message.content
        print(f"\n{assistant_reply}\n")

        # Добавляем ответ ассистента в историю
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
