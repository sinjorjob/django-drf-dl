from rest_framework import serializers
from appv1.config import *  
from appv1.predict import predict2
from appv1.bert import get_config, BertModel,BertForchABSA  
import torch
import os


class BertPredictSerializer(serializers.Serializer):
    """BERTのネガポジ分類結果を得るシリアライザ"""

    input_text = serializers.CharField()
    neg_pos = serializers.SerializerMethodField()

    def get_neg_pos(self, obj):
        config = get_config(file_path=BERT_CONFIG)  #bertコンフィグファイルのロード
        net_bert = BertModel(config)  #BERTモデルの生成
        net_trained = BertForchABSA(net_bert) # #BERTモデルにネガポジ用分類機を結合
        net_trained.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))  #学習済みの重みをロード
        net_trained.eval()  #推論モードにセット
        label = predict2(obj['input_text'], net_trained).numpy()[0]  #推論結果を取得
        return label
