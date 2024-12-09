import sys
import re
from types import TracebackType
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from string import digits
import psycopg2
import psycopg2.extras
from rdkit import Chem
import os
from .vectorization import vectorize_text
from tqdm import tqdm
import nltk
from enum import Enum
from .GNNT.task import *
from .extraction import Extractor
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch import cuda
import time
import .StandardizerTaskType

class ChemicalStructures():
    """
    Основной класс для анализа обработки химических и биохимических текстов.
    Этот класс предназначен для выделения химических структур из текстов на русском и английсков языках
    и их стандартизации.
    """

    extract_model = None
    standartisation_model = None
    conn = None
    curs = None
    empty_db = False
    def __init__(self, db_name=None, user=None, password=None, host=None, port=None):
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Создание объекта ChemicalStructures")
        if(db_name is None or user is None or password is None or host is None or port is None):
            self.empty_db = True
        if(not self.empty_db):
            self._get_db_connection(db_name, user, password, host, port)
            if not self.curs:
                self._get_db_cursor()
        print("Загрузка моделей")
        self.tokenizer = BertTokenizer.from_pretrained('AlexeyMol/mBERT_chemical_ner')
        self.model = BertModel.from_pretrained('AlexeyMol/mBERT_chemical_ner')
        # Определение устройства для работы модели
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        # device = "cpu"
        self.model.to(self.device)
    
    def create_db_connection(self, db_name, user, password, host, port)->None:
        """
        Создание подключения к базе данных.
        Args:
            db_name (str): Имя базы данных.
            user (str): Имя пользователя.
            password (str): Пароль.
            host (str): Хост.
            port (str): Порт.
        """
        self._get_db_connection(db_name, user, password, host, port)
        if not self.curs:
            self._get_db_cursor()
        self.empty_db = False
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None,) -> None:
        if(not self.empty_db):
            self._close_connection()

    def _get_db_connection(self, db_name, user, password, host, port):
        try:
            self.conn = psycopg2.connect(
                'postgresql://' + user + ':' + password + '@' + host + ':' + port + '/' + db_name)
            self.curs = self.conn.cursor()
        except:
            print('Can`t establish connection to database')
            print("Unexpected error:", sys.exc_info())
            return

    def _get_db_cursor(self):

        try:
            self.curs = self.conn.cursor()
        except:
            print('Can`t establish connection to database')
            print("Unexpected error:", sys.exc_info())

    def _close_connection(self):
        self.conn.close()

    def write_translate(self)-> None:
        t = pd.read_csv("./translated_titles.csv")
        data = []
        for _, row in t.iterrows():
            data.append([ row["title_rus"], row["pubchem_index"]])
        self.update(data)

    def update(self, values) -> None:

        """
        Обновление данных в базе данных.

        Args:
            values (list): Значения для обновления.
        """
        if(self.empty_db):
            print("Database connection empty. First create it by create_db_connection method!")
            return
        strsd = """UPDATE public.compound_titles as t SET title_rus = %s WHERE t.pubchem_index = %s"""
        psycopg2.extras.execute_batch(self.curs, strsd, values)
        self.conn.commit()

    def VectorizeData(self, model=None, tokenizer=None, device=None):
        """
        Векторизация данных.

        Args:
            model (BertModel): Загруженная модель.
            tokenizer (BertTokenizer): Загруженный токенизатор.
            device (str): Устройство для работы модели.

        Returns:
            list: Векторизованные данные.
        """
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return
        data = []
        while True:
            rows = self.curs.execute(
                f"select * from public.compound_titles where vector is NULL LIMIT 1000")
            if not rows:
                return
            for row in tqdm(rows, total=len(rows)):
                data.append([vectorize_text(
                    row[1], True, tokenizer=tokenizer, device=device, model=model), row[0]])
            self.update(data)
            data.clear()

    def GetChemicalStructuresNames(self, vector, limit=2):
        """
        Функция получает названия химических структур из базы данных.
        Args:
            vector -- вектор, который используется для поиска в базе данных.
            limit -- количество результатов, которые нужно вернуть. По умолчанию равно 2.
        Returns:
            Список названий химических структур. Если ничего не найдено, возвращает пустой список.
        """
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return        
        rows = self.curs.execute(
            f"SELECT title FROM public.compound_titles ORDER BY vector <=> '{vector}' LIMIT {limit}")
        if not rows:
            return []
        return [row for row in rows]

    def GetChemicalStructuresByPubChemIndex(self, pubchemindex, limit=2):
        """
        Функция получает названия химических структур из базы данных по индексу PubChem.
        Args:
            pubchemindex -- индекс PubChem, который используется для поиска в базе данных.
            limit -- количество результатов, которые нужно вернуть. По умолчанию равно 2.
        Returns:
            Список названий химических структур. Если ничего не найдено, возвращает пустой список.
        """
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return        
        rows = self.curs.execute(
            f"SELECT title FROM public.compound_titles where pubchem_index = {pubchemindex} LIMIT {limit}")
        if not rows:
            return []
        return [row for row in rows]

    def GetChemicalStructuresFromText(self, text, limit=2):
        """
        Функция получает названия химических структур из базы данных по тексту.
        Args:
            text -- текст, который используется для поиска в базе данных.
            limit -- количество результатов, которые нужно вернуть. По умолчанию равно 2.
        Returns:
            Список названий химических структур. Если ничего не найдено, возвращает пустой список.
        """
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return        
        rows = self.curs.execute(
            f"SELECT title FROM public.compound_titles where title = '{text}' LIMIT {limit}")
        if not rows:
            return []
        return [row for row in rows]
    

    def extract_chem_data_from_text(self, text):
        with Extractor() as extract:
            return extract.extract_chemicals_from_text(text)

    # получение химических структур из текста
    def GetChemicalStructuresFromText(self, text, type_search="cosine", save_file=True, benchmark=False):
        """
        Функция для извлечения химических структур из текста.

        Аргументы:
        text (str): Текст, из которого нужно извлечь химические структуры.
        type_search (str, optional): Метод сравнения векторов. Допустимые значения: "cosine", "L1", "L2". По умолчанию "cosine".
        save_file (bool, optional): Если True, то химические структуры сохраняются в файл. Если False, то возвращается молекулярная структура. По умолчанию True.

        Возвращает:
        Mol или None: Молекулярную структуру или None, если save_file равен False.

        Исключения:
        Unexpected error: В случае возникновения непредвиденной ошибки.
        """
        try:
            c_text = self.clean_text(text)
            data = self.extract_chem_data_from_text(c_text)
            chem_structures = list(data.keys())
            for text in chem_structures:
                vector = vectorize_text(text, True, tokenizer=self.tokenizer, device=self.device, model=self.model)
                pubchem_index=None
                if (benchmark):continue
                if type_search == "cosine":
                    pubchem_index = self.get_vector_row_by_cosine(vector,1)[0]
                elif type_search == "L1":
                    pubchem_index = self.get_vector_row_by_L1(vector, 1)[0]
                elif type_search == "L2":
                    pubchem_index = self.get_vector_row_by_L2(vector, 1)[0]
                inchi = self.get_inchi_by_pubchem_index(pubchem_index)
                raw_mol = Chem.MolFromInchi(inchi[0])
                if save_file:
                    self._save_to_file(raw_mol, ".", text)
                else:
                    return raw_mol
        except:
            print("Unexpected error:", sys.exc_info())

    def _save_to_file(self, data:Mol, path:str =".",name:str="default") -> None:
        """
        Сохранение молекулы в файл.
        """
        try:
            with open(f"{path}/{name}.mol", "w") as f:
                    b_mol = Chem.MolToMolBlock(data)
                    f.write(b_mol)
                    f.close()  
        except:
            print("Unexpected error:", sys.exc_info())

    def get_vector_row_by_cosine(self, vector: str, limit_row: int = 5) -> Union[list,None]:
        '''
        Получение N ближайших записей по косинусному расстоянию векторов.
        '''
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return None
        self.curs.execute(
                f"SELECT pubchem_index FROM public.compound_titles ORDER BY vector <=> '{vector}' LIMIT {limit_row}")
        return self.curs.fetchall()[0]

    def get_vector_row_by_L1(self, vector: str, limit_row: int = 5) -> list:
        '''
        Получение N ближайших записей по расстоянию L1 для векторов.
        '''
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return None
        self.curs.execute(
            f"SELECT * FROM public.compound_titles ORDER BY vector <+> '{vector}' LIMIT {limit_row}")
        t = self.curs.fetchall()
        return t
    
    def get_vector_row_by_L2(self, vector: str, limit_row: int = 5) -> list:
        '''
        Получение N ближайших записей по расстоянию L2 для векторов.
        '''
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM public.compound_titles ORDER BY vector <-> '{vector}' LIMIT {limit_row}")
            return cursor.fetchall()
        
    def standartize_mol_file(self, data: str, type:StandardizerTaskType=StandardizerTaskType.MolFile,raw_mol=None,  smiles_text:str=None, inchi_code:str=None, file_name:str=None, save_path:str=None) -> None:
        """
        Функция принимает на вход данные в формате строки (data) и тип стандартизации (type).
        Если тип стандартизации равен StandardizerTaskType.MolFile, то функция вызывает метод __StandardizeMolFile__ с параметром data.
        Если тип стандартизации равен StandardizerTaskType.SmilesText, то функция проверяет наличие параметра smiles_text. Если он отсутствует, то вызывается исключение ValueError с сообщением об ошибке. Затем функция вызывает метод __StandardizeSmilesText__ с параметром smiles_text.
        Если тип стандартизации равен StandardizerTaskType.InchiCode, то функция проверяет наличие параметра inchi_code. Если он отсутствует, то вызывается исключение ValueError с сообщением об ошибке. Затем функция вызывает метод __StandardizeInchiCode__ с параметром inchi_code.
        :param data: данные в формате строки
        :param type: тип стандартизации
        :param smiles_text: текст в формате SMILES
        :param inchi_code: код InChI
        :param file_name: имя файла
        :param save_path: путь к папке для сохранения
        :return: None
        """
        if(type==StandardizerTaskType.MolFile):
            return self.__StandardizeMolFile__(data, raw_mol)
        if(type == StandardizerTaskType.SmilesText):
            if not smiles_text: raise ValueError(f"The smiles_text argument is missing.You cannot use type {type} without smiles text.")
            return self.__StandardizeSmilesText__(save_path, file_name, smiles_text)
        if(type == StandardizerTaskType.InchiCode):
            if not inchi_code: raise ValueError(f"The inchi_code argument is missing. You cannot use type {type} without inchi.")
            return self.__StandardizeInchiCode__(inchi_code,save_path, file_name)
   

    def get_inchi_by_pubchem_index(self, index):
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return None
        self.curs.execute(f"SELECT inchi FROM public.pubchem_database WHERE pubchem_index = '{index}';")
        return self.curs.fetchone()
    
    def get_inchi_list_by_pubchem_indexes(self, indexes):
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return
        self.curs.execute(f"SELECT inchi FROM public.pubchem_database WHERE pubchem_index in '{indexes}';")
        return self.curs.fetchall()

    def benchmark(self, text, type_search="cosine", save_file=True, benchmark=False):
        t1 = time.time()
        r = self.GetChemicalStructuresFromText(text,type_search, save_file, benchmark=True)
        t2 = time.time()
        print("Function=%s, Time=%s" % (self.GetChemicalStructuresFromText.__name__, t2 - t1))
        print(len(text))

    # сохранение выделенных в тексте химических структур в mol формате без стандартизации
    def ChemicalStructuresToMol(self, chemical_structures_list, path_to_folder, file_prefix):
        """
        This function saves a list of chemical structures to mol files without standardizing them.

        :param self: The instance of the class.
        :param chemical_structures_list: A list of dictionaries containing the chemical structure information.
        :param path_to_folder: The path to the folder where the mol files should be saved.
        :param file_prefix: The prefix to be added to the file names.
        :return: A list of the results of the SaveSingleChemStructerToMolWithoutStandartization function for each chemical structure.
        """

        try:
            result = list(map(lambda i: self.SaveSingleChemStructerToMolWithoutStandartization(
                chemical_structures_list[i], path_to_folder + "/" + file_prefix + "_" + str(i) + ".mol"), range(0, len(chemical_structures_list))))
            return result
        except Exception as e:
            # в случае сбоя будет выведено сообщение в STDOUT
            print("Unexpected error:", e)

    def SaveSingleChemStructerToMolWithoutStandartization(self, chemical_structure_dict, path_to_folder, file_prefix):
        """
        This function saves a single chemical structure to a mol file without standardizing it.
        :param self: The instance of the class.
        :param chemical_structure_dict: A dictionary containing the chemical structure information.
        :param path_to_folder: The path to the folder where the mol file should be saved.
        :param file_prefix: The prefix to be added to the file name.
        :return: The result of the SaveMolFile function.
        """
        try:
            return self.__save_Mol_File__(chemical_structure_dict, path_to_folder, file_prefix)
        except Exception as e:
            print("Unexpected error:", e)
            return None

    def __save_Mol_File__(self, chemical_structure_dict, path_to_folder, file_prefix):
        """
        This function saves a mol file.
        :param self: The instance of the class.
        :param chemical_structure_dict: A dictionary containing the chemical structure information.
        :param path_to_folder: The path to the folder where the mol file should be saved.
        :param file_prefix: The prefix to be added to the file name.
        :return: The result of the SaveMolFile function.
        """
        try:
            with open(path_to_folder + "/" + file_prefix + ".mol", "w") as f:
                f.write(Chem.MolToMolBlock(chemical_structure_dict['mol']))
            return True
        except Exception as e:
            print("Unexpected error:", e)
            
    def GetChemicalStructureInchi(self, chemical_structure_dict):
        """
        This function gets the Inchi code for a chemical structure.

        :param self: The instance of the class.
        :param chemical_structure_dict: A dictionary containing the chemical structure information.
        :return: The Inchi code for the chemical structure.
        """
        if(self.empty_db):
            print("Database connection empty. First create it by Create_DB method!")
            return None
        try:
            with self.conn.cursor() as curs:

                row = curs.execute('SELECT inchi FROM pubchem_database WHERE pubchem_index IN (SELECT puchem_index FROM compound_titles WHERE rus_title = %s)', (chemical_structure_dict['name'],))
                if not row:
                    row = curs.execute('SELECT inchi FROM pubchem_database WHERE pubchem_index IN (SELECT puchem_index FROM compound_titles WHERE title = %s)', (chemical_structure_dict['name'],))
                if not row:
                    print("Empty result")
                    return None
                inchicode = curs.fetchone()[0]
                mol = Chem.MolFromInchi(inchicode)
                return mol
        except Exception as e:
            # в случае сбоя будет выведено сообщение в STDOUT
            print("Unexpected error:", e)

    # получение графовых структур по inchi для массива химических структур
    def GetChemicalStructuresInchi(self, chemical_structures_list):
        """
        This function gets the Inchi codes for a list of chemical structures.

        :param self: The instance of the class.
        :param chemical_structures_list: A list of dictionaries containing the chemical structure information.
        :return: A list of the Inchi codes for the chemical structures.
        """
        try:
            result = list(map(lambda i: self.GetChemicalStructureInchi(
                chemical_structures_list[i]), range(0, len(chemical_structures_list))))
            return result
        except Exception as e:
            # в случае сбоя будет выведено сообщение в STDOUT
            print("Unexpected error:", e)

    def __StandardizeSmilesText__(self, save_path, filename, smiles = None):
        """
        This function standardizes a mol file.
        :save_path: The path to the folder where the mol file should be saved.
        :param filename: The name of the file to be saved.
        :param self: The instance of the class.
        :param smiles: The smiles to be standardized.If not specified, the smiles will be taken from the file.
        """
        try:
            st_file_path = save_path+"/" +filename+ "_standardized.mol"
            if not smiles:
                print("Smiles was not specified")
                return; 
            smiles_raw = Chem.MolFromSmiles(smiles)
            stdrz = StandardizerTask()
            s_mol = stdrz.predict_mol(smiles_raw, False)
            with open(st_file_path, 'w') as f:
                b_mol = Chem.MolToMolBlock(s_mol)
                f.write(b_mol)
                f.close()
        except Exception as e:
            # в случае сбоя будет выведено сообщение в STDOUT
            print("Unexpected error:", e)

    def __StandardizeInchiCode__(self, inchi, file_path, filename):
        """
        This function standardizes an Inchi code.
        :param self: The instance of the class.
        :param inchi: The Inchi code to be standardized.
        :param file_path: The path to the file where the standardized Inchi code should be saved.
        :param filename: The name of the file to be saved.
        """
        try:
            mol = Chem.MolFromInchi(inchi)
            stdrz = StandardizerTask()
            s_mol = stdrz.predict_mol(mol, False)
            with open(file_path, 'w') as f:
                b_mol = Chem.MolToMolBlock(s_mol)
                f.write(b_mol)
                f.close()
        except Exception as e:
            print("Unexpected error:", e)
        
    def __StandardizeMolFile__(self, path_to_file, mol = None):
        """
        This function standardizes a mol file.

        :param self: The instance of the class.
        :param path_to_file: The path to the mol file to be standardized.
        """
        try:
            f_file_name = os.path.basename(path_to_file)
            file_name = f_file_name.split(".mol")[0]
            st_file_path = path_to_file.split(
                f_file_name)[0] + file_name + "_standardized.mol"
            if not mol:
                mol = Chem.MolFromMolFile(path_to_file)
            stdrz = StandardizerTask()
            s_mol = stdrz.predict_mol(mol, False)
            with open(st_file_path, 'w') as f:
                b_mol = Chem.MolToMolBlock(s_mol)
                f.write(b_mol)
                f.close()
        except Exception as e:
            # в случае сбоя будет выведено сообщение в STDOUT
            print("Unexpected error:", e)      
    def __SmilesToMolStandartize__ (self, SMILES, file_path=None):
        mol = Chem.MolFromSmiles(SMILES)
        stdrz = StandardizerTask()
        s_mol = stdrz.predict_mol(mol, False)
        if file_path:
            with open(file_path, 'w') as f:
                b_mol = Chem.MolToMolBlock(s_mol)
                f.write(b_mol)
                f.close()
        else:
            return Chem.MolToMolBlock(s_mol)

    def __InchiToMolStandartize (self, inchi, file_path=None):
        mol = Chem.InchiToMol(inchi)
        stdrz = StandardizerTask()
        s_mol = stdrz.predict_mol(mol, False)
        if file_path:
            with open(file_path, 'w') as f:
                b_mol = Chem.MolToMolBlock(s_mol)
                f.write(b_mol)
                f.close()
        else:
            return Chem.MolToMolBlock(s_mol)


    def InchiToStantartisedMolFile(self, path_to_file, mol):
        try:
            # тут будет часть кода для стандартизации
            s_mol = self.__InchiToMolStandartize(path_to_file, mol)
            with open(path_to_file, 'w') as f:
                b_mol = Chem.MolToMolBlock(s_mol)
                f.write(b_mol)
        except:
            print("Unexpected error:", sys.exc_info())

    # стандартизация нескольких mol файлов

    def InchiToStantartisedMolFiles(self, mol_list, path_to_folder, file_name_prefix):
        try:
            r = list(map(lambda i: self.InchiToStantartisedMolFile(path_to_folder + "/" +
                     file_name_prefix + "_" + i + "_stantartised.mol", mol_list[i]), range(0, len(mol_list))))
        except:
            # в случае сбоя будет выведено сообщение в STDOUT
            print("Unexpected error:", sys.exc_info())

    def tokenize_text(slef, text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 3:
                    continue
                tokens.append(word)
        return tokens

    def clean_text(self, text):
        """
            text: a string

            return: modified initial string
        """
        REPLACE_BY_SPACE_RE = re.compile('[/{}\[\]\|@,;]')
        STOPWORDS = set(stopwords.words('russian'))

        res_text = BeautifulSoup(text, "html.parser").text  # HTML decoding
        res_text = res_text.lower()  # lowercase text
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        res_text = REPLACE_BY_SPACE_RE.sub(' ', res_text)
        res_text = ' '.join(word for word in res_text.split(
        ) if word not in STOPWORDS)  # delete stopwors from text

        return res_text
