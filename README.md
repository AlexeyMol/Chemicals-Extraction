# О проекте
Эта библиотека предназначена для извлечения названий химических структур из текстов на русском и английском языках. В библиотеке три основные функции, которые могут быть использованы независимо друг от друга: извлечение названий химических структур из текстов на русском и английском языках, преобразования и сохранения их в MOL формате и стандартизации mol формата.
Для извлечения названий химических структур была обучения кросс языковая модель на базе mBert. Для сохранения химических структур в MOL формате используется модифицированная онтология PubChem. Для стандартизации была обучена модель с применением графовых нейронных сетей.

Библеотека разработана для:
 - Выделение химических структур.
 - Стандартизация химических структур

Библеотека может быть использована для решения прикладных задач:
 - Проведение научных исследований в области в химии и извлечении информации из текстов.
 - Разботка специализированных поисковых машин.
 - Разработка систем агрегации данных из больших массивов текстов в области химии и смежных с ней областей.
 - Любые проекты которые требуют стандартизации химических структур в разных форматах.
 - Разработки в области создания специализированных поисковых индексов.

<hr></hr>

# Минимальные системные требования (Для работы стандартизатора и извлечения)
- CPU не менее 8 ядер с частотой не менее 4Ггц
- RAM не менее 64 ГБ
- SSD nvme не менее 1.5 ТБ
- GPU уровня 4060TI 16GB или мощнее

# Рекомендованные системные требования (Для работы стандартизатора  и извлечения)
- CPU не менее 12 ядер с частотой не менее 4Ггц
- RAM не менее 128 ГБ
- SSD nvme не менее 1.5 ТБ
- GPU уровня 4060TI 16GB или мощнее

# Системные требования к рабочей станции (для индексирования базы данных )
- CPU не менее 8 ядер с частотой не менее 4Ггц
- RAM не менее 328 ГБ (от 64 Гб - но чем меньше время индексирования растёт экспонинциально)
- SSD nvme не менее 1.5 ТБ
- GPU уровня 4060TI 16GB или мощнее

# Требования к ПО
- Debian 11 или новее
- Postgresql 15.7 или новее
- Python 3.11.2 (можно новее, но работоспособность не гарантирована)
- CUDA 11 или новее

<hr></hr>

# !!! Если вы установили пароль то в процессе работы с СУБД может потребоватся его ввести !!!
<hr></hr>

# Установка
Установить необходимые зависимости с помощью команды:

С помощью комманды 
```bash
make
```
Убедиться что пакет make установлен в случае если возникла ошибка 
```bash
make: command not found
```
Установить пакет make с помощью команды:
```bash
sudo apt-get install make
sudo apt-get install build-essential
```
Далее необходимо установить сопуствующие пакеты необходимые для установки различных компонентов системы:
```bash
sudo apt-get install git wget curl
```
Далее необходимо установить СУБД PostgreSQL версии 15.7 и новее:

```bash
sudo apt-get install postgresql postgresql-server-dev-<postgres versions>
```
Подробнее прочитать про версии и часто возникающие ошибки можно <a href="https://www.postgresql.org/download/linux/debian/">здесь</a> и <a href="https://www.postgresql.org/docs/15/">здесь</a>.

Далее установить необходимо установить расширение для pgvector выполнив поочередно команды:
```bash
cd /tmp
git clone --branch v0.7.2 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install # may need sudo
```
Полную инструкцию по установке можете найти на <a href = "https://github.com/pgvector/pgvector?tab=readme-ov-file">в официальном гитхаб проекте</a>.


Скачиваете <a href="http://92.63.99.164/data/">Дамп базы данных молекул</a>.
<hr></hr>

# !!! Перед выполнением комманд убедиться что сервер СУБД и CLUSTER запущены.!!!
Сделать можно с помощью команды:
```bash
sudo systemctl status postgresql
```
Подробнее можно ознакомиться <a href="https://webhostinggeeks.com/howto/how-to-check-the-status-of-postgresql-on-an-ubuntu-server/">в этой инструкции</a>.
<hr></hr>

После того как разархивируете его, необходимо  создать базу данных куда будут записаны данные и произвести операцию разворота дампа базы с помощью psql и pg_restore:

```bash
sudo su
su postgres
psql
CREATE DATABASE chem_db_vector;
\c chem_db_vector;
CREATE EXTENSION vector;
\q
```
где: chem_db_vector - имя базы данных

Перед запуском комманды:
```bash
pg_restore -U postgres -W -Fc -d chem_db_vector -j <num of cores> <path/to/dump>
```
убедитесь что вы используете пользователя postgres.

После того как развернули базу данных из бекапа необходимо ее проиндексировать (перед запуском убедитесь что выполняете комманды как пользователь postgres):
```sql
psql
\c chem_db_vector;

SET maintenance_work_mem = '19GB';
SET max_parallel_maintenance_workers = 12;
CREATE INDEX ON public.compound_titles USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=64);
```
где:
- maintenance_work_mem - размер единичного графа для индексирования. Чем больше вы выстовляете значение тем больше будет требоваться памяти, но тем быстрее будет идти  индексация.
- max_parallel_maintenance_workers - Сколько потоков будут зедействованы в индексировании + 1 главный.

Параметры maintenance_work_mem и max_parallel_maintenance_workers выставляются соответствующие вашей рабочей станции. <a href = "https://github.com/pgvector/pgvector?tab=readme-ov-file#index-build-time"> Подробнее можно ознакомиться здесь</a>.

<hr></hr>

# Настройка виртуального окружения

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
И повторите попытку создания виртуального окружения.

После активируйте среду:
```bash
source project_name_myproject/bin/activate
```
Далее выполняете установку:
```bash
pip install psycopg2 langdetect torch torch_geometric


pip install git+https://github.com/AlexeyMol/Chemicals-Extraction.git
```
<hr></hr>


# Описание основного класса:
[Описание класса ChemStructers](ChemStructures.md)

<hr></hr>

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
standartize_mol_file(path_to_file.mol)
```

<hr></hr>



# Пример использования:
Извлечение названий хисических структур,получение их inchi и их стандартизациия.

```python
from mol_analyzer  import ChemicalStructures as bt

def main():
    # """
    # Основная функция.
    # """
    # Загрузка токенизатора и модели

    with bt.ChemicalStructures("chem_db_vector", "postgres", "4847", "localhost", "5432") as chem_struct:
        with open("./d5b5469ffc744968b3d68fe80217b552.pdf.txt", "+r") as file:
            text = " ".join(file.readlines()).replace("\n", " ").replace("\r", " ")
            chem_struct.GetChemicalStructuresFromText(text)
            chem_struct.standartize_mol_file("name.mol")
if __name__ == "__main__":
    main()
```
Стандартизация существующего mol-файла. 
```python
from mol_analyzer  import ChemicalStructures as bt

def main():
    # """
    # Основная функция.
    # """
    # Загрузка токенизатора и модели

    with bt.ChemicalStructures("chem_db_vector", "postgres", "4847", "localhost", "5432") as chem_struct:
        with open("./niclosamide_noClNoNitro.mol", "+r") as file:
            moldata = file.readlines();
            chem_struct.standartize_mol_file("name.mol", raw_mol = moldata)
if __name__ == "__main__":
    main()
```

Стандартизация из SMILES и сохранение результатов в mol формате.
```python
from mol_analyzer  import ChemicalStructures as bt
from mol_analyzer  import StandardizerTaskType

def main():
    # """
    # Основная функция.
    # """
    # Загрузка токенизатора и модели

    with bt.ChemicalStructures("chem_db_vector", "postgres", "4847", "localhost", "5432") as chem_struct:
        smiles = 'Cc1ccccc1' # 
        chem_struct.standartize_mol_file("name.mol", smiles_text =smiles, type=StandardizerTaskType.SmilesText)
if __name__ == "__main__":
    main()
```
Извлечение названий хисических структур  и сохранение результатов в json формате.
```python
from mol_analyzer  import ChemicalStructures as bt

def main():
    # """
    # Основная функция.
    # """
    # Загрузка токенизатора и модели

    with bt.ChemicalStructures("chem_db_vector", "postgres", "4847", "localhost", "5432") as chem_struct:
        with open(".eng/3YpuBDzIIcFoDamn71g4tcrdZQpG3QspcG4AfjUTPpzkY0RygvlC_CbZ0PizNoss.txt", "+r") as file:
            text = " ".join(file.readlines()).replace("\n", " ").replace("\r", " ")
            dictionary =json.dumps(chem_struct.extract_chem_data_from_text(text))
            with open("./data.json", w+) as file2:
                file2.writelines(dictionary)
                file2.close()
if __name__ == "__main__":
    main()

```
Результат выполнения: Содержимое файла data.json
```json
{
    "secoisolariciresinol": {
        "1": [
            2993,
            3013
        ],
        "2": [
            39908,
            39928
        ],
        "3": [
            40788,
            40808
        ],
        "4": [
            41807,
            41827
        ],
        "5": [
            43000,
            43020
        ],
        "6": [
            43112,
            43132
        ]
    },
    "dihydroquercetin": {
        "1": [
            3018,
            3034
        ],
        "2": [
            39933,
            39949
        ],
        "3": [
            40813,
            40829
        ],
        "4": [
            41832,
            41848
        ],
        "5": [
            43024,
            43040
        ],
        "6": [
            43137,
            43153
        ]
    },
    "lignan": {
        "1": [
            3851,
            3857
        ],
        "2": [
            12342,
            12348
        ],
        "3": [
            16904,
            16910
        ]
    },
    "flavonoid": {
        "1": [
            3862,
            3871
        ],
        "2": [
            12353,
            12362
        ]
    },
    "Dihydroquercetin": {
        "1": [
            4267,
            4283
        ]
    },
    "DHQ": {
        "1": [
            4358,
            4361
        ],
        "2": [
            8423,
            8426
        ],
        "3": [
            12780,
            12783
        ],
        "4": [
            16086,
            16089
        ],
        "5": [
            19504,
            19507
        ],
        "6": [
            20099,
            20102
        ],
        "7": [
            26988,
            26991
        ],
        "8": [
            28916,
            28919
        ],
        "9": [
            29188,
            29191
        ],
        "10": [
            30937,
            30940
        ],
        "11": [
            31212,
            31215
        ],
        "12": [
            32266,
            32269
        ],
        "13": [
            32671,
            32674
        ],
        "14": [
            33139,
            33142
        ]
    },
    "isolated": {
        "1": [
            4871,
            4879
        ]
    },
    "Larixin": {
        "1": [
            4968,
            4975
        ]
    },
    "SECO": {
        "1": [
            5035,
            5039
        ],
        "2": [
            13301,
            13305
        ],
        "3": [
            17200,
            17204
        ],
        "4": [
            19512,
            19516
        ],
        "5": [
            26979,
            26983
        ],
        "6": [
            28907,
            28911
        ],
        "7": [
            29179,
            29183
        ],
        "8": [
            30928,
            30932
        ],
        "9": [
            31203,
            31207
        ],
        "10": [
            32257,
            32261
        ],
        "11": [
            33494,
            33498
        ],
        "12": [
            33926,
            33930
        ],
        "13": [
            37982,
            37986
        ],
        "14": [
            39063,
            39067
        ]
    },
    "phenol": {
        "1": [
            6305,
            6311
        ],
        "2": [
            6463,
            6469
        ],
        "3": [
            8885,
            8891
        ]
    },
    "hydroxymatairesinol": {
        "1": [
            6869,
            6888
        ]
    },
    "compounds": {
        "1": [
            6982,
            6991
        ]
    },
    "class": {
        "1": [
            7131,
            7136
        ]
    },
    "to": {
        "1": [
            7841,
            7843
        ]
    },
    "extraction": {
        "1": [
            8506,
            8516
        ]
    },
    "Acetone": {
        "1": [
            11641,
            11648
        ]
    },
    "lignans": {
        "1": [
            15370,
            15377
        ]
    },
    "acetone": {
        "1": [
            21289,
            21296
        ],
        "2": [
            21433,
            21440
        ],
        "3": [
            24889,
            24896
        ]
    },
    "propanol": {
        "1": [
            22938,
            22946
        ]
    },
    "isolation": {
        "1": [
            24214,
            24223
        ]
    },
    "acetate": {
        "1": [
            26310,
            26317
        ]
    },
    "petroleum": {
        "1": [
            26320,
            26329
        ]
    },
    "Mixture": {
        "1": [
            27431,
            27438
        ],
        "2": [
            29347,
            29354
        ]
    },
    "ether": {
        "1": [
            28857,
            28862
        ]
    },
    "mg": {
        "1": [
            31867,
            31869
        ]
    },
    "sample": {
        "1": [
            32838,
            32844
        ]
    },
    "preparations": {
        "1": [
            38484,
            38496
        ]
    },
    "Coulter": {
        "1": [
            38719,
            38726
        ]
    },
    "ethanol": {
        "1": [
            39241,
            39248
        ]
    },
    "ranges": {
        "1": [
            40727,
            40733
        ],
        "2": [
            41745,
            41751
        ],
        "3": [
            42810,
            42816
        ],
        "4": [
            43291,
            43297
        ],
        "5": [
            43415,
            43421
        ],
        "6": [
            43539,
            43545
        ],
        "7": [
            43664,
            43670
        ],
        "8": [
            43789,
            43795
        ],
        "9": [
            43914,
            43920
        ],
        "10": [
            44164,
            44170
        ],
        "11": [
            44289,
            44295
        ]
    },
    "treating": {
        "1": [
            41062,
            41070
        ]
    }
}

```

<hr></hr>

## Ошибки
В случае возникновения ошибки при выполнении функций будет выведено сообщение в STDOUT. При создании issue на гитхаб просьба прикладывать errorlog и error traceback
<hr></hr>

# Благодарности
Библиотека создана при поддержке Фонда содействия инновациям (Договор №37ГУКодИИС12-D7/86408 о предоставлении гранта на выполнение проекта открытых библиотек от от 09.08.2023).





