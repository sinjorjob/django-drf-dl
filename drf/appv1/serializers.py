from rest_framework import serializers
from appv1.config import *	
from appv1.predict import predict2, create_vocab_text, build_bert_model	
from IPython.display import HTML, display	
from appv1.bert import get_config, BertModel,BertForchABSA, set_learned_params	
	
import torch	
import os	
from django.conf import settings	


class BertPredictSerializer(serializers.Serializer):
    """BERTのネガポジ分類結果を得るシリアライザ"""

    input_text = serializers.CharField()
    neg_pos = serializers.SerializerMethodField()

    def get_neg_pos(self, obj):
        config_path = os.path.join(settings.BASE_DIR, 'appv1/weights/bert_config.json')	
        config = get_config(file_path=config_path)
        net_bert = BertModel(config)
        net_trained = BertForchABSA(net_bert) # 学習モデルのロード
        net_trained.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
        net_trained.eval()
        print("obj=",obj)
        input_text = "㈱東急コミュニティーにおいて管理ストックがマンション、ビルともに拡大し増収増益となりました"
        label = predict2(input_text, net_trained).numpy()[0]
        return input_text, label
