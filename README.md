🛍️ Интернет-магазин на FastAPI
Описание проекта
Это backend-часть интернет-магазина одежды и аксессуаров, реализованная на FastAPI. Проект включает:

- Систему пользователей (регистрация, аутентификация)

- Каталог товаров с фильтрацией

- Корзину и оформление заказов

- Админ-панель для управления контентом

🛠 Технологии
- Backend: Python 3.10+, FastAPI

- База данных: SQLite/PostgreSQL (SQLAlchemy ORM)

- Аутентификация: JWT-токены

- Деплой: Docker

- Тестирование: pytest

⚙️ Установка и запуск
- Клонирование репозитория

git clone https://github.com/bladesadsad/summer2025


cd skyline-riot


3. Настройка окружения
Создайте файл .env в корне проекта:

ini


DATABASE_URL=sqlite:///./shop.db


SECRET_KEY=ваш-секретный-ключ


ALGORITHM=HS256


ACCESS_TOKEN_EXPIRE_MINUTES=30

3. Запуск через Docker (рекомендуется)

   
docker-compose up --build


5. Или локальный запуск

   
pip install -r requirements.txt


uvicorn main:app --reload



📚 Документация API


После запуска сервера документация будет доступна:

Swagger UI: http://localhost:8000/docs


МИИГАиК 2025
