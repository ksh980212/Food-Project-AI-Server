from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import pandas as pd
from flask import jsonify

def insta_crawler(instagramID):
    options = Options()
    options.add_argument('--no-proxy-server')
    options.add_argument("--proxy-server='direct://'");
    options.add_argument("--proxy-bypass-list=*");
    options.add_argument('headless')
    driver = webdriver.Chrome(executable_path='./chromedriver/chromedriver', options=options)
    driver.get('http://instagram.com/'+str(instagramID))
    try:
        driver.find_element_by_css_selector('.error-container')
        driver.quit()
        print("존재하지 않는 계정입니다")
        return 'Page Not Found'
    except:
        pass
    try:
        driver.find_element_by_css_selector('.rkEop')
        driver.quit()
        print("비공개 계정입니다")
        return 'Private page'
    except:
        pass
    article = driver.find_elements_by_css_selector('.v1Nh3 > a')[:12]
    url = [i.get_attribute('href') for i in article]
    text = ""
    for i in url:
        driver.get(i)
        try:
            text += driver.find_element_by_css_selector('.C4VMK').text
        except:
            continue
    df = pd.read_csv('./data/food_name_data_utf8.csv')
    food_index = [df[df['food_name']==i].index.tolist()[0] for i in df.food_name if i in text]
    if (len(food_index) < 5):
        driver.quit()
        print("데이터가 불충분한 계정입니다")
        return 'Insufficient data'
    driver.quit()
    print("크롤링 결과 인덱스 :",food_index)
    return food_index
