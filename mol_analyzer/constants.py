#make constant static class
class Constants:
    NOISE = [',', '%', 'в', 'и', 'или', 'а', 'с', 'за',
            'к', ':', ';', '.', '+', '!', '>', '<', '=',
            '[', ']', ' ', '/', '', '""', '"', '(', ')',
            'как', 'на', 'но', 'of', 'is', 'are', 'a', 'and',
            'or', 'then', 'when', 'for', 'it', 'with', 'that',
            'are', 'at', 'we', 'be', 'was', 'the', 'do', 'does',
            'did', 'as', 'so', 'in', 'were', 'use', 'often', 'may',
            'will', 'can', 'on', 'has', 'had', 'have', 'into', '{', '}',
            'ил', 'из', 'ка', 'ра', 'да', 'для', 'не', 'по', 'which',
            'that', 'that', "'", '"', '""', "''", '−', '-', '[UNK]']

    LABEL_TO_ID = {'B-chem': 1, 'I-chem': 2, 'L-chem': 3, 'S-chem': 4,  'O': 0}
    ID_TO_LABEL = {0: 'O', 1: 'B-chem', 2: 'I-chem', 3: 'L-chem', 4: 'S-chem'}
    