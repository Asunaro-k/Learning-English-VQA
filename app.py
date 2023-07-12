#必要なライブラリをインポート
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import datasets
import nltk
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor,AutoTokenizer
from transformers import pipeline
from transformers.tools import image_captioning
from pathlib import Path
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig
from transformers import ViltProcessor, ViltForQuestionAnswering

#1回だけ呼び出すもの
def load_image_captioner():
    return pipeline("image-to-text", model="./image-captioning-output/lastcheckpoint/")

def load_model():
    model_folder = './model/vqg_model1/'  # モデルが保存されているフォルダのパス

    # モデルを読み込む
    model = ProphetNetForConditionalGeneration.from_pretrained(model_folder)
    tokenizer = ProphetNetTokenizer.from_pretrained(model_folder)

    return model, tokenizer

# モデルの読み込みを行う関数
def load_vilt_model():
    model_folder1 = './model/vqa_model/'  # モデルが保存されているフォルダのパス

    # モデルを読み込む
    processor = ViltProcessor.from_pretrained(model_folder1)
    model = ViltForQuestionAnswering.from_pretrained(model_folder1)

    return processor, model
 

#ロードするモデルの定義
def imgcap_model(image_captioner,image_path):
    #imagecaption = image_path
    #temp_image = np.array(Image.open(imagecaption))
    result = image_captioner(image_path)
    return result[0]['generated_text']


def main():

    st.set_page_config(layout="wide")
    #タイトルの表示
    st.title("VQA Englsih")
    #制作者の表示
    st.text("Created by Kibune Sohta")
    #アプリの説明の表示
    st.markdown("sample")

    #サイドバーの表示
    image_path = st.sidebar.file_uploader("画像をアップロードしてください", type=['jpg','jpeg', 'png'])
    #st.text(image_path)
    #サンプル画像を使用する場合
    use_sample = st.sidebar.checkbox("サンプル画像を使用する")
    if use_sample:
        image_path = "3189251454_03b76c2e92.jpg"

    #start
    os.environ["WANDB_DISABLED"] = "true"
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)

    #保存済みのモデルをロード
    if "image_captioner" not in st.session_state:
        st.session_state.image_captioner = load_image_captioner()

    # モデルの読み込み
    if "vqa_model" not in st.session_state:
        st.session_state.vqg_model, st.session_state.vqg_tokenizer = load_model()

    # モデルの読み込み
    if "vilt_processor" not in st.session_state:
        st.session_state.vilt_processor, st.session_state.vilt_model = load_vilt_model()


    #画像ファイルが読み込まれた後，実行
    if image_path != None:
        
        #画像の読み込み
        image = np.array(Image.open(image_path))
        pil_image = Image.fromarray(image) 
        #画像から検出を行う
        result = imgcap_model(st.session_state.image_captioner,pil_image)
        #検出を行った結果を表示
        st.image(image,use_column_width=True)
        st.text(result)
    

        # 機能の追加
    if st.button("Generate Question"):
        vqg_tokenizer = st.session_state.vqg_tokenizer
        inputs = vqg_tokenizer([result], return_tensors='pt')
        vqg_model = st.session_state.vqg_model

        # 質問の生成
        question_ids = vqg_model.generate(inputs['input_ids'], num_beams=5, early_stopping=True)
        result2 = vqg_tokenizer.batch_decode(question_ids, skip_special_tokens=True)

        st.text("Question:"+result2[0])
    
    if st.button("Generate Answer"):
        processor = st.session_state.vilt_processor
        model = st.session_state.vilt_model

        # 画像の読み込みと前処理
        encoding = processor(pil_image, result2[0], return_tensors="pt")

        # 質問応答の実行
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        result3 = model.config.id2label[idx]

        st.text("Predicted answer:"+ result3)


if __name__ == "__main__":
    #main関数の呼び出し
    main()