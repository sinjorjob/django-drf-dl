from django.shortcuts import render
from rest_framework import generics, status, views
from rest_framework.response import Response
from appv1.serializers import BertPredictSerializer
from rest_framework.decorators import api_view
import requests
from rest_framework.views import APIView
from rest_framework import mixins



class BertPredictAPIView(mixins.ListModelMixin,mixins.CreateModelMixin,generics.GenericAPIView):
    """BERTネガポジ分類予測クラス"""
    serializer_class = BertPredictSerializer

    def post(self, request, *args, **kwargs):     

        serializer = self.get_serializer(request.data)
        print("post!!!")
        return Response(serializer.data, status=status.HTTP_200_OK)


 
@api_view(['GET'])
def bert_predict(request, *args, **kwargs):
    if request.method == 'GET':
        contents = { str(key):val for key, val in kwargs.items() }
        print("request.data=", request.data)        
        serializer = BertPredictSerializer(data=request.data)
        if serializer.is_valid():
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        


