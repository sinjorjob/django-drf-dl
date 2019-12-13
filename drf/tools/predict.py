import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import json

def predict(input_text):
    URL = "http://127.0.0.1:8000/api/v1/predict/"
    values = {
        "format": "json",
        "input_text": input_text,
            }
    data = urllib.parse.urlencode({'input_text': input_text}).encode('utf-8')
    request = urllib.request.Request(URL, data)
    response = urllib.request.urlopen(request)
    result= json.loads(response.read())
    return result['neg_pos']

if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    print('load csv file ....')
    df = pd.read_csv("test.csv", engine="python", encoding="utf-8-sig")
    df["PREDICT"] = np.nan   #予測列を追加
    print('Getting prediction results ....')
    for index, row in df.iterrows():
        df.at[index, "PREDICT"] = predict(row['INPUT'])
    print('save results to csv file')
    df.to_csv("predicted_test .csv", encoding="utf-8-sig", index=False)
    print('Processing terminated normally.')
