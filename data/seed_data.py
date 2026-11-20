import random
from datetime import datetime, timedelta
import psycopg2
import clickhouse_connect

#Настройки подключений
PG_CONFIG = {
    "dbname": "lamoda_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

CH_CONFIG = {
    "host": "localhost",
    "port": 8123,
    "username": "default",
    "password": "",
    "database": "lamoda_analytics",
}

#Справочники для генерации реалистичных данных Lamoda
CITIES = ["Москва", "Санкт-Петербург", "Краснодар", "Екатеринбург", "Казань", "Новосибирск"]

CATEGORIES = {
    "Одежда": ["Куртка утепленная", "Джинсы зауженные", "Худи оверсайз", "Платье миди", "Футболка базовая"],
    "Обувь": ["Кроссовки беговые", "Ботинки кожаные", "Кеды классические", "Лоферы замшевые"],
    "Аксессуары": ["Рюкзак городской", "Ремень кожаный", "Очки солнцезащитные", "Шарф кашемировый"]
}

TAGS_POOL = ["новинка", "скидка_20%", "премиум", "хит_продаж", "эко", "распродажа"]


def init_postgres():
    print("Инициализация PostgreSQL...")
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    #Создаем таблицы
    cur.execute("DROP TABLE IF EXISTS users CASCADE;")
    cur.execute("DROP TABLE IF EXISTS warehouses CASCADE;")

    cur.execute("""
        CREATE TABLE users (
            user_id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            city VARCHAR(50),
            registration_date DATE
        );
    """)

    cur.execute("""
        CREATE TABLE warehouses (
            warehouse_id SERIAL PRIMARY KEY,
            city VARCHAR(50),
            address VARCHAR(200)
        );
    """)

    #Генерируем склады
    warehouses_data = [
        ("Москва", "МКАД 19-й км, с1"),
        ("Санкт-Петербург", "Пулковское шоссе, 45"),
        ("Екатеринбург", "Уральская, 12"),
        ("Краснодар", "Уральская улица, 100")
    ]
    cur.executemany("INSERT INTO warehouses (city, address) VALUES (%s, %s);", warehouses_data)

    #Генерируем пользователей (50 штук)
    users_data = []
    start_date = datetime.now() - timedelta(days=365)
    for i in range(1, 51):
        name = f"Пользователь_{i}"
        city = random.choice(CITIES)
        reg_date = start_date + timedelta(days=random.randint(1, 350))
        users_data.append((name, city, reg_date.strftime("%Y-%m-%d")))

    cur.executemany("INSERT INTO users (name, city, registration_date) VALUES (%s, %s, %s);", users_data)

    conn.commit()
    cur.close()
    conn.close()
    print("PostgreSQL успешно заполнена тестовыми данными!")


def init_clickhouse():
    print("Инициализация ClickHouse...")
    client = clickhouse_connect.get_client(**CH_CONFIG)

    #Удаляем и создаем таблицу фактов продаж
    client.command("DROP TABLE IF EXISTS sales_fact;")
    
    client.command("""
        CREATE TABLE sales_fact (
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
        ) ENGINE = MergeTree()
        ORDER BY (sale_date, category, user_id);
    """)

    #Генерируем данные продаж (200 строк)
    sales_data = []
    start_date = datetime.now() - timedelta(days=60)
    
    sale_id = 1
    for order_id in range(1001, 1200):  # 200 заказов
        user_id = random.randint(1, 50)
        warehouse_id = random.randint(1, 4)
        sale_date = (start_date + timedelta(days=random.randint(0, 60))).date()

        # В одном заказе может быть от 1 до 3 товаров
        items_count = random.randint(1, 3)
        for _ in range(items_count):
            category = random.choice(list(CATEGORIES.keys()))
            product_name = random.choice(CATEGORIES[category])
            sku = f"SKU-{abs(hash(product_name)) % 10000:04d}"
            price = round(random.uniform(1500.0, 25000.0), 2)
            quantity = random.randint(1, 3)
            revenue = round(price * quantity, 2)
            tags = random.sample(TAGS_POOL, k=random.randint(0, 2))

            sales_data.append([
                sale_id, order_id, user_id, sku, product_name,
                category, price, quantity, revenue, warehouse_id,
                sale_date, tags
            ])
            sale_id += 1

    client.insert("sales_fact", sales_data, column_names=[
        "sale_id", "order_id", "user_id", "sku", "product_name", 
        "category", "price", "quantity", "revenue", "warehouse_id", 
        "sale_date", "tags"
    ])
    
    print("ClickHouse успешно заполнена тестовыми данными!")


if __name__ == "__main__":
    init_postgres()
    init_clickhouse()