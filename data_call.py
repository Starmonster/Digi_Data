from sshtunnel import SSHTunnelForwarder
import pymongo
import pandas as pd
import datetime

def data_call():
    try:
        print("Calling data store...")
        MONGO_HOST = '35.178.138.67'
        SERVER_USER   = 'ubuntu'
        PRIVATE_KEY ='/Users/jamesmoulds/Desktop/projects/digi/awsdigikey_int_01.pem'

        MONGO_DB = "digi-account"

        # define ssh tunnel
        server = SSHTunnelForwarder(
            MONGO_HOST,
            ssh_username=SERVER_USER,
            ssh_pkey=PRIVATE_KEY,
            remote_bind_address=('127.0.0.1', 27118)
        )

        # start ssh tunnel
        server.start()

        connection = pymongo.MongoClient('127.0.0.1', server.local_bind_port)
        db = connection[MONGO_DB]
        print("Connection acquired...")
        print("Accessing data....")

        family = pd.DataFrame(list(db["user-family"].find()))
        account = pd.DataFrame(list(db["user-account"].find()))
        profiles = pd.DataFrame(list(db["user-profile"].find()))
        # Save to csv
        print("Data collected...")
        print("Saving latest data...")
        profiles.to_csv("app_data/profiles.csv", index=False)
        account.to_csv("app_data/account.csv", index=False)
        family.to_csv("app_data/family.csv", index=False)

        last_call_date_df = pd.DataFrame([datetime.datetime.now()], columns=["last_call"])
        last_call_date_df.to_csv("app_data/last_call_date.csv", index=False)

        print("Data call successful!")
    except:
        print("call failed- more error logging to be introduced here")


    # data = pd.DataFrame(list(db['user-profile'].find())).drop(columns=['_id'])
    # print("PRINTING......")
    # print(data.shape)
    return None

# data_call()
