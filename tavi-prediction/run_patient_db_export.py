import csv
import logging as log
import os
import sqlite3


DATA_PATH = '/data/calcium_processed/'
DB_NAME = 'site.db'
PATIENT_TABLE = 'patient'
RESULTS_METADATA_SUBFOLDER = 'calcium'


if __name__ == '__main__':
    log.basicConfig(level='INFO', format='%(asctime)s [%(levelname)s]: %(message)s')
    
    if not os.path.exists(f'metadata/{RESULTS_METADATA_SUBFOLDER}'):
        os.makedirs(f'metadata/{RESULTS_METADATA_SUBFOLDER}', exist_ok=True)
    
    db_path = os.path.join(DATA_PATH, DB_NAME)
    log.info(f'Connecting to DB {db_path}')
    db_connection = sqlite3.connect(db_path)
    cursor = db_connection.cursor()
    cursor.execute(f'SELECT * FROM {PATIENT_TABLE};')
    
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    
    log.info(f'Extracted {len(rows)} rows from DB')
    log.info(f'Table {PATIENT_TABLE} structure: {columns}')
    
    csv_file_path = os.path.join('metadata', RESULTS_METADATA_SUBFOLDER, 'database.csv')
    with open(csv_file_path, 'w') as file:
        csv_writer = csv.writer(file, dialect='excel')
        csv_writer.writerow(columns + ['ct', 'xray'])
        
        for row in rows:
            ct_path = os.path.join(DATA_PATH, row[0], 'ct')
            has_ct = os.path.exists(ct_path) and os.listdir(ct_path) != []
            xray_path = os.path.join(DATA_PATH, row[0], 'rx')
            has_xray = os.path.exists(xray_path) and os.listdir(xray_path) != []
            csv_writer.writerow(row + (has_ct, has_xray))
            
    log.info(f'Data exported to {csv_file_path}')
