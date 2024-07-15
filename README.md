Эта библиотека предназначена для извлечения названий химических структур из текстов на русском и английском языках. В библиотеке три основные функции, которые могут быть использованы независимо друг от друга: извлечение названий химических структур из текстов на русском и английском языках, преобразования и сохранения их в MOL формате и стандартизации mol формата.
Для извлечения названий химических структур была обученя кроссязыковая модель на базе mBert. Для сохранения химических структур в MOL формате используется модифицированная онтология PubChem. Для стандартизации была обучена модель с применением графовых нейронных сетей.
# Системные требования к рабочей станции
- CPU не менее 8 ядер с частотой не менее 4Ггц
- RAM не менее 328 ГБ 
- SSD nvme не менее 1.5 ТБ
- GPU уровня 4060TI 16GB или мощнее
# Требования к ПО
- Debian 11 или новее
- Postgresql 15.7 или новее 
- Python 3.11.2 (можно новее, но работоспособность не гарантирована)
- CUDA 11 или новее

# Установка
Установить необходимые зависимости с помощью команнды:

```bash
sudo apt-get install postgresql git wget curl  postgresql-server-dev-<postgres versions> 
```
Далее установить pgvector выполнив поочередно комманды:
```bash
cd /tmp
git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install # may need sudo
```
Полную инструкцию по установке можете найти на <a href = "https://github.com/pgvector/pgvector?tab=readme-ov-file">в официальном гитхаб проекте</a>.

Скачиваете <a href="http://92.63.99.164/data/">Дамп базы данных молекул</a>.
после того как разархивируете его разверните его с помощью psql:
```bash
sudo su
su postgres
psql 
CREATE DATABASE <NAME>;
\c <NAME>;
CREATE EXTENSION vector;
\q
pg_restore -U postgres -W -Fc -d <NAME> -j <num of cores> -f <path/to/dump>
```
где: NAME - имя базы данных 

После того как развернули базу данных из бекапа необходимо ее проиндексировать:
```sql
SET maintenance_work_mem = '4GB';
SET max_parallel_maintenance_workers = 4;
CREATE INDEX ON public.compound_titles USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=64);
```
где параметры maintenance_work_mem и max_parallel_maintenance_workers выстовляются соответствующие вашей рабочей станции.

Далее заходим в корневой каталог вашего проекта и создаем виртуальное окружение:

```bash
mkdir my_python_project
cd my_python_project

python -m venv project_name_myproject
```
Замените project_name_myproject на желаемое название виртуальной среды.
Если у Вас возникает следующая ошибка:
```bash
The virtual environment was not created successfully because ensurepip is not
available. On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

apt-get install python3-venv

You may need to use sudo with that command. After installing the python3-venv
package, recreate your virtual environment.

Failing command: ['/home/osboxes/my_python_project/project_name_myproject/bin/python3', '-Im', 'ensurepip', '--upgrade', '--default-pip']
```
выполните комманду
```bash
sudo apt-get install python3-venv -y
``` 
После активируйте среду:
```bash
source project_name_myproject/bin/activate
```
Далее выполняете установку:
```bash
pip install psycopg2 langdetect torch torch_geometric

pip install git+<git link>
```

# Api reference:
[ChemStructers API reference](ChemStructures.md)
# Входные данные:
<p><b>chemical_structure_dict:</b> словарь, содержащий информацию о химической структуре.
<p><b>chemical_structures_list:</b> список словарей, содержащих информацию о химических cтруктурах. </p>
<p><b>path_to_file:</b> путь к файлу, содержащему химическую структуру в формате mol. </p>
<p><b>path_to_folder:</b> путь к папке, в которую будут сохранены mol файлы. </p>
<p><b>file_prefix:</b> префикс для имен файлов.</p>

# Выходные данные:
Inchi code: Inchi код химической структуры.
List of Inchi codes: список Inchi кодов химических структур.
Standardized mol file: mol файл, стандартизированный с помощью StandardizerTask.

# Функции
Извлекает названия химических структур из текста. Входные параметры - текст на русском или английском языке. Выходные параметры - массив токенов, характеризующих названия химиеских структур, и их координаты в тексте (начало и конец подстроки)

```python
GetChemicalStructuresFromText(text)
```

Получает Inchi для химической структуры.

```python
GetChemicalStructureInchi(chemical_structure_dict)
```
Получает список Inchi кодов для массива химических структур.

```python
GetChemicalStructuresInchi(chemical_structures_list)
```
Сохраняет химическую структуру в mol файл.

```python
SaveChemicalStructureToMol(chemical_structure_dict, path_to_file)
```
Стандартизирует mol файл.

```python
StandardizeMolFile(path_to_file)
```


# Пример использования:
```python
# Получение Inchi для химической структуры
inchi_code = GetChemicalStructureInchi({'language': 'en', 'name': 'Ethylbenzene'})

def main():
    # """
    # Основная функция.
    # """
    # Загрузка токенизатора и модели

    with ChemicalStructures("chem_db_vector", "postgres", "4847", "localhost", "5432") as chem_struct:
        with open("./d5b5469ffc744968b3d68fe80217b552.pdf.txt", "+r") as file:
            text = " ".join(file.readlines()).replace("\n", " ").replace("\r", " ")
            chem_struct.GetChemicalStructuresFromText(text)

        # chem_struct.test("пенициллинацилаза.mol", False, "sds.std.mol")
if __name__ == "__main__":
    main()
```


## Ошибки
В случае возникновения ошибки при выполнении функций будет выведено сообщение в STDOUT.

# Требования:
```
Python 3.6+
PyMySQL
rdkit
```
# Установка
Скачайте и распакуйте проект.
Введите в терминале pip install -r requirements.txt.
Запустите скрипт main.py.
Поддержка
Если у вас возникнут проблемы с использованием проекта, пожалуйста, свяжитесь со мной.

