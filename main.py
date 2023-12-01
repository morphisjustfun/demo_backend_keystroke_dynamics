from flask import Flask, request
import os
import json
import numpy as np
import pandas as pd
import hashlib
import datetime
import sqlalchemy
from flask_cors import CORS


def get_db(client_id):
    if client_id == '1234-abcd':
        return sqlalchemy.create_engine(
            'postgresql://default:7ouirQM1OEUZ@ep-floral-morning-923422.us-east-1.postgres.vercel-storage.com:5432/verceldb'
        )

    return None


class KeystrokeQuery:
    current_user = None
    chracter_time = None
    client_id: str
    df = None

    def __init__(self, client_id, min_records=20, threshold=0.2):
        self.client_id = client_id
        self.MIN_RECORDS = min_records
        self.THRESHOLD = threshold

    @staticmethod
    def getPathModel(client_id, username):
        return f'models/{client_id}/{username}.parquet'

    def dimension_from_username(self):
        return len(self.current_user) * 2 - 1

    def load_df(self):
        if os.path.exists(KeystrokeQuery.getPathModel(self.client_id, self.current_user)):
            self.df = pd.read_parquet(KeystrokeQuery.getPathModel(self.client_id, self.current_user))
        else:
            self.df = pd.DataFrame(columns=range(self.dimension_from_username()))

    def get_vector(self):
        keystrokes = []
        for i, keystroke in enumerate(self.chracter_time):
            holdtime = keystroke['endTime'] - keystroke['beginTime']
            keystrokes.append(holdtime)
            if i < len(self.chracter_time) - 1:
                next_keystroke = self.chracter_time[i + 1]
                digraph = next_keystroke['beginTime'] - keystroke['beginTime']
                keystrokes.append(digraph)
        return keystrokes

    def nearest_neighbour(self, keystrokes):
        df = self.df

        distances = np.sqrt(np.sum(np.square(df - keystrokes), axis=1))
        distance = distances.min() / distances.max()

        return 1 - distance

    def query(self, json_data):
        self.current_user = json_data['username']
        self.chracter_time = json_data['characterTime']
        self.chracter_time = list(filter(lambda x: x['character'] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áéíóúÁÉÍÓÚ', self.chracter_time))
        self.load_df()
        keystrokes = self.get_vector()
        if self.df.shape[0] < self.MIN_RECORDS:
            # self.df.loc[len(self.df)] = keystrokes
            # self.df.to_parquet(KeystrokeQuery.getPathModel(self.client_id, self.current_user))
            # print(f"Not enough records for user {self.current_user}. Record added")
            # print(f'User {self.current_user} is authenticated')
            return True
        else:
            distance = self.nearest_neighbour(keystrokes)
            print(f'Nearest neighbour distance: {distance}')

            if distance > self.THRESHOLD:
                print(f'User {self.current_user} is authenticated')
                self.df.loc[len(self.df)] = keystrokes
                self.df.to_parquet(KeystrokeQuery.getPathModel(self.client_id, self.current_user))
                return True
            else:
                print(f'User {self.current_user} is not authenticated')
                return False

    def save(self, json_data):
        self.current_user = json_data['username']
        self.chracter_time = json_data['characterTime']
        # delete from chracter_time every record with character not in [a-zA-Z0-9] including tildes
        self.chracter_time = list(filter(lambda x: x['character'] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áéíóúÁÉÍÓÚ', self.chracter_time))
        self.load_df()
        keystrokes = self.get_vector()
        self.df.loc[len(self.df)] = keystrokes
        self.df.to_parquet(KeystrokeQuery.getPathModel(self.client_id, self.current_user))


app = Flask(__name__)
CORS(app)


def exists_folder(folder_name):
    return os.path.isdir(folder_name)


@app.route("/auth", methods=['POST'])
def auth():
    body = request.get_json()
    client_id = body['client_id']
    username = body['username']

    if not exists_folder(f'models/{client_id}'):
        os.mkdir(f'models/{client_id}')

    aka = KeystrokeQuery(client_id, min_records=20, threshold=0.68)
    result = aka.query(body)

    timestamp = datetime.datetime.now().timestamp()
    toSendUser = [client_id, timestamp, username, result]
    toSendUserPayload = hashlib.sha256(str(toSendUser).encode()).hexdigest()

    if result:
        toSendServer = [client_id, timestamp, username, True]
        toSendServerPayload = hashlib.sha256(str(toSendServer).encode()).hexdigest()
        # send to server
        db = get_db(client_id)

        if db is not None:
            with db.connect() as conn:
                data = {
                    'hash': toSendServerPayload,
                }
                sql = sqlalchemy.text(
                    'INSERT INTO public."Tries" (hash) VALUES (:hash)'
                )
                conn.execute(sql, data)
                conn.commit()
            return json.dumps({'payload': toSendUserPayload})

        return json.dumps({'payload': toSendUserPayload})
    else:
        return json.dumps({'payload': toSendUserPayload})


@app.route("/train", methods=['POST'])
def train():
    body = request.get_json()
    username = body['username']
    characterTime = body['characterTime']
    client_id = body['client_id']

    aka = KeystrokeQuery(client_id, min_records=20, threshold=0.7)
    aka.save(body)
    return json.dumps({'status': 'ok'})


if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
