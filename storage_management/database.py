import sqlite3
import files_management as fm


class Repository:
    def __init__(self):
        self.DB_PATH = "../cash/search_app.db"
        self.connection = sqlite3.connect(self.DB_PATH)
        self.cursor = self.connection.cursor()
        self.ensure_tables_exist()
        self.connection.close()

    def add_user(self, user, password):
        self.connection = sqlite3.connect(self.DB_PATH)
        self.cursor = self.connection.cursor()

        try:
            self.cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, password))
            self.connection.commit()
        except sqlite3.Error as err:
            print(f"Ошибка при добавлении пользователя {err}")
        finally:
            self.connection.close()

    def get_user(self, username, password):
        self.connection = sqlite3.connect(self.DB_PATH)
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute("SELECT password FROM users WHERE username = ? LIMIT 1", username)
            row = self.cursor.fetchall()
            if row:
                stored_password = row[0]
                return stored_password == password
            return False
        except sqlite3.Error as err:
            print(f"Ошибка при чтении пользователя {err}")
        finally:
            self.connection.close()

    def add_file_sector(self, index, fileSector):
        self.connection = sqlite3.connect(self.DB_PATH)
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute("INSERT INTO files (index_id, file_name, start_of_sector, end_of_sector) VALUES (?, ?, ?, ?)",
                           (index, fileSector.filename, fileSector.start, fileSector.end))
            self.connection.commit()
        except sqlite3.Error as err:
            print(f"Ошибка при добавлении файла {err}")
        finally:
            self.connection.close()



    def add_file_sectors(self, fileSectors):
        self.connection = sqlite3.connect(self.DB_PATH)
        self.cursor = self.connection.cursor()

        self.clear_index()

        files = [(i, fileSectors[i].filename, fileSectors[i].start, fileSectors[i].end) for i in range(len(fileSectors))]

        try:
            self.connection.executemany("""
            INSERT OR IGNORE INTO files (index_id, file_name, start_of_sector, end_of_sector)
            VALUES (?, ?, ?, ?)
        """, files)
            self.connection.commit()
        except sqlite3.Error as err:
            print(f"Ошибка при добавлении файлов {err}")
        finally:
            self.connection.close()

    def get_all_files(self) -> list[fm.TextSector]:
        self.connection = sqlite3.connect(self.DB_PATH)
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute("SELECT * FROM files")
            raw_data = self.cursor.fetchall()
            return [fm.TextSector(raw_data[i][2], "", raw_data[i][3], raw_data[i][4], raw_data[i][1]) for i in range(len(raw_data))]
        except sqlite3.Error as e:
            print(f"Ошибка при получении файлов: {e}")
            return []
        finally:
            self.connection.close()

    def clear_index(self):
        self.cursor.execute("DELETE FROM files")

    def ensure_tables_exist(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = set(name[0] for name in self.cursor.fetchall())

        # Проверяем и создаём таблицу users, если нужно
        if 'users' not in existing_tables:
            self.cursor.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            """)
            print("Создана таблица 'users'.")

        # Проверяем и создаём таблицу files, если нужно
        if 'files' not in existing_tables:
            self.cursor.execute("""
                CREATE TABLE files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    index_id TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    start_of_sector INTEGER NOT NULL,
                    end_of_sector INTEGER NOT NULL
                )
            """)
            print("Создана таблица 'files'.")

        self.connection.commit()