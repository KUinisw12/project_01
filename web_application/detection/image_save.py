from requests.compat import urlparse, urlunparse, urljoin
from bs4 import BeautifulSoup
import re
import requests
from requests import Session, request, get
import urllib.request
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import os
from PIL import Image


def load_image(goods_id):
    header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.62'}
    web_page = "https://www.musinsa.com/app/goods/"+str(goods_id)
    res = get(web_page, headers=header)
    soup = BeautifulSoup(res.text, "html.parser")
    image = soup.find("div", attrs={"class":"product-img"})
    link = "http:"+image.img["src"] 
    image_ =  str(goods_id)+'.jpg'
    urllib.request.urlretrieve(link,  'image/' + image_)
    return image_

# Image.open("product.jpg")