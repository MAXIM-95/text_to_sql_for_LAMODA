import json #для сериализации примеров датасета в формат JSONL
import random
from pathlib import Path

#Описание схем таблиц (пригодится и для SFT, и для Schema Linking / промптов агента)
SCHEMAS = {
    "clickhouse": """
Таблица ClickHouse (база lamoda_analytics):
sales_fact (
    sale_id UInt32,
    order_id UInt32,
    user_id UInt32,
    sku String,
    product_name String,
    category LowCardinality(String), 
    price Float32,
    quantity UInt16,
    revenue Float32,
    warehouse_id UInt32,
    sale_date Date,
    tags Array(String)
)
""".strip(),
    "postgres": """
Таблицы PostgreSQL (база lamoda_db):
users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(50),
    registration_date DATE
)
warehouses (
    warehouse_id SERIAL PRIMARY KEY,
    city VARCHAR(50),
    address VARCHAR(200)
)
""".strip(),
    "hybrid": """
ClickHouse (lamoda_analytics):
sales_fact (sale_id, order_id, user_id, sku, product_name, category, price, quantity, revenue, warehouse_id, sale_date, tags)

PostgreSQL (lamoda_db):
users (user_id, name, city, registration_date)
warehouses (warehouse_id, city, address)
""".strip(),
}

# Базовые шаблоны пар (вопрос -> sql, диалект, схема)
BASE_SAMPLES = [
    # ClickHouse: базовые агрегации и категории
    (
        "clickhouse",
        "Покажи топ-3 категории товаров по общей выручке",
        "SELECT category, sum(revenue) AS total_revenue FROM sales_fact GROUP BY category ORDER BY total_revenue DESC LIMIT 3",
    ),
    (
        "clickhouse",
        "Какая общая выручка по категории Обувь?",
        "SELECT sum(revenue) AS total_revenue FROM sales_fact WHERE category = 'Обувь'",
    ),
    (
        "clickhouse",
        "Сколько всего единиц товаров продано по каждому SKU?",
        "SELECT sku, product_name, sum(quantity) AS total_units FROM sales_fact GROUP BY sku, product_name ORDER BY total_units DESC",
    ),
    (
        "clickhouse",
        "Посчитай количество уникальных заказов в таблице продаж",
        "SELECT uniqExact(order_id) AS unique_orders FROM sales_fact",
    ),
    (
        "clickhouse",
        "Покажи выручку по дням за последние 14 дней",
        "SELECT sale_date, sum(revenue) AS daily_revenue FROM sales_fact WHERE sale_date >= today() - 14 GROUP BY sale_date ORDER BY sale_date",
    ),
    # ClickHouse: специфика массивов Array(String)
    (
        "clickhouse",
        "Найди товары с тегом хит_продаж и посчитай их выручку по категориям",
        "SELECT category, sum(revenue) AS total_revenue FROM sales_fact WHERE has(tags, 'хит_продаж') GROUP BY category ORDER BY total_revenue DESC",
    ),
    (
        "clickhouse",
        "Разверни теги товаров и покажи выручку по каждому тегу",
        "SELECT tag, sum(revenue) AS total_revenue FROM sales_fact ARRAY JOIN tags AS tag GROUP BY tag ORDER BY total_revenue DESC",
    ),
    (
        "clickhouse",
        "Сколько продаж было с тегом скидка_20%?",
        "SELECT count() AS sales_count FROM sales_fact WHERE has(tags, 'скидка_20%')",
    ),
    # PostgreSQL: пользователи и склады
    (
        "postgres",
        "Сколько пользователей зарегистрировано в городе Москва?",
        "SELECT count(*) AS users_count FROM users WHERE city = 'Москва'",
    ),
    (
        "postgres",
        "Покажи список всех складов в городе Санкт-Петербург",
        "SELECT warehouse_id, city, address FROM warehouses WHERE city = 'Санкт-Петербург'",
    ),
    (
        "postgres",
        "Выведи 10 самых новых пользователей по дате регистрации",
        "SELECT user_id, name, city, registration_date FROM users ORDER BY registration_date DESC LIMIT 10",
    ),
    (
        "postgres",
        "Посчитай количество пользователей по каждому городу",
        "SELECT city, count(*) AS users_count FROM users GROUP BY city ORDER BY users_count DESC",
    ),
    # Смешанные / кросс-табличные сценарии
    (
        "hybrid",
        "Какую выручку принесли продажи со склада с ID 1?",
        "SELECT warehouse_id, sum(revenue) AS total_revenue FROM sales_fact WHERE warehouse_id = 1 GROUP BY warehouse_id",
    ),
    (
        "hybrid",
        "Покажи топ-5 пользователей по сумме покупок в ClickHouse",
        "SELECT user_id, sum(revenue) AS total_spent FROM sales_fact GROUP BY user_id ORDER BY total_spent DESC LIMIT 5",
    ),
]

# Вариации формулировок для расширения датасета без потери смысла
PARAPHRASE_PREFIXES = [
    "",
    "Подскажи, пожалуйста: ",
    "Сформируй запрос для аналитики: ",
    "Нужна выгрузка — ",
]


def build_dataset(num_samples: int = 80, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    dataset = []

    for i in range(num_samples):
        dialect, question, sql = BASE_SAMPLES[i % len(BASE_SAMPLES)]
        prefix = rng.choice(PARAPHRASE_PREFIXES)
        full_question = f"{prefix}{question[0].lower() + question[1:]}" if prefix else question

        schema_key = dialect if dialect in SCHEMAS else "hybrid"
        system_prompt = (
            "Ты — аналитический SQL-ассистент. Генерируй только валидный SQL-запрос "
            "под указанный диалект и схему без лишних пояснений и markdown-оформления."
        )
        user_prompt = (
            f"Диалект: {dialect.upper()}\n"
            f"Схема:\n{SCHEMAS[schema_key]}\n\n"
            f"Вопрос: {full_question}"
        )

        # Формат messages совместим с chat_template Qwen 2.5 и SFT-тренерами
        dataset.append({
            "id": i + 1,
            "dialect": dialect,
            "question": full_question,
            "sql": sql,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": sql},
            ],
        })

    rng.shuffle(dataset)
    return dataset


def main() -> None:
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sft_train.jsonl"

    samples = build_dataset(num_samples=80)
    with output_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Сгенерировано {len(samples)} примеров в {output_path}")


if __name__ == "__main__":
    main()