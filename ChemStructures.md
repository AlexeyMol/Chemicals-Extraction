## Документация проекта

Этот проект представляет собой класс `ChemicalStructures`, который предназначен для анализа и обработки химических и биохимических текстов. Класс предоставляет методы для выделения химических структур из текстов на русском и английском языках, стандартизации химических структур и взаимодействия с базой данных.

### Основные методы класса:

1. `VectorizeData`: Векторизация данных.
2. `GetChemicalStructuresNames`: Получение названий химических структур из базы данных.
3. `GetChemicalStructuresByPubChemIndex`: Получение названий химических структур из базы данных по индексу PubChem.
4. `GetChemicalStructuresFromText`: Получение названий химических структур из базы данных по тексту.
5. `extract_chem_data_from_text`: Извлечение химических структур из текста.
6. `SaveSingleChemStructerToMolWithoutStandartization`: Сохранение одной химической структуры в mol формате без стандартизации.
7. `GetCemicalStructureIchi`: Получение inchi для химической структуры.
8. `GetCemicalStructuresIchi`: Получение графовых структур по inchi для массива химических структур.
9. `StandartizeMolFile`: Стандартизация mol файла.
10. `InchiToStantartisedMolFile`: Стандартизация графовой структуры.

### Дополнительные методы:

1. `GetVectorRowByCosine`: Получение N ближайших записей по косинусному расстоянию векторов.
2. `GetVectorRowByL1`: Получение N ближайших записей по расстоянию L1 для векторов.
3. `GetVectorRowByL2`: Получение N ближайших записей по расстоянию L2 для векторов.
4. `InchiToStantartisedMolFiles`: Стандартизация нескольких mol файлов.
5. `TokenizeText`: Токенизация текста.
6. `CleanText`: Очистка текста от лишних символов и стоп-слов.

### Пример использования класса:


```python
# Создание объекта ChemicalStructures
cs = ChemicalStructures('db_name', 'user', 'password', 'host', 'port')

# Векторизация данных
cs.VectorizeData()

# Получение названий химических структур из базы данных
names = cs.GetChemicalStructuresNames('vector')

# Сохранение одной химической структуры в mol формате без стандартизации
cs.SaveSingleChemStructerToMolWithoutStandartization({'name': 'name', 'language': 'ru'}, 'path_to_file')

# Стандартизация mol файла
cs.StandartizeMolFile('path_to_file')

# Закрытие соединения с базой данных
cs._close_connection()
```
### Примечания:

- Класс ChemicalStructures использует библиотеки psycopg2, nltk, re, bs4, numpy, langdetect, pickle, rdkit и vectorization.
- Для работы с базой данных требуется установленный PostgreSQL сервер.
- Класс предполагает, что в базе данных есть таблица compound_titles с колонками title, vector и pubchem_index.
- Методы VectorizeData, GetChemicalStructuresNames, GetChemicalStructuresByPubChemIndex и GetChemicalStructuresFromText предполагают, что в таблице compound_titles уже есть векторные представления химических структур.
- Методы SaveSingleChemStructerToMolWithoutStandartization, GetCemicalStructureIchi, GetCemicalStructuresIchi, StandartizeMolFile, InchiToStantartisedMolFile и InchiToStantartisedMolFiles предполагают, что в базе данных есть таблица pubchem_database с колонкой inchi.