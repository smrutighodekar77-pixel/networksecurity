import os
import sys
import json
import pandas as pd
import pymongo
import certifi

from dotenv import load_dotenv
load_dotenv()

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            self.mongo_url = MONGO_DB_URL
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            logging.info(f"Converted CSV to JSON records: {len(records)}")
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            client = pymongo.MongoClient(
                self.mongo_url,
                tls=True,
                tlsCAFile=ca,
                serverSelectionTimeoutMS=30000
            )

            db = client[database]
            col = db[collection]

            result = col.insert_many(records)
            client.close()

            logging.info(f"Inserted {len(result.inserted_ids)} records into MongoDB")
            return len(result.inserted_ids)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")
    DATABASE = "SMRUTI"
    COLLECTION = "NetworkData"

    networkobj = NetworkDataExtract()

    records = networkobj.csv_to_json_convertor(FILE_PATH)
    print(f"Records converted: {len(records)}")

    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, COLLECTION)
    print(f"Records inserted: {no_of_records}")
