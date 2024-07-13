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

    