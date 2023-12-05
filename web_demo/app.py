from flask import Flask, render_template, request
import os
import argparse
import pandas as pd

app = Flask(__name__)

csv_file_path = os.path.join('static', 'test_prediction.csv') # output.csv是sort過的csv
img = os.path.join('static', 'test_full_images')

# 全域變數
df = pd.read_csv(csv_file_path)
df = df.sort_values(by=['file', 'tooth'])
df['Check_data'] = None 
image = 0

@app.route('/', methods=['POST', 'GET'])
def show():
    global df, image

    def trans_type(num):
        if (num == '0'):
            return '複雜拔牙'
        elif (num == '1'):
            return '單純齒切'
        elif (num == '2'):
            return '複雜齒切'
        else:
            return '有問題'

    #抽出有image的row
    image_list = []
    for i in range(len(df)):
        if df.iloc[i, 0] != df.iloc[i-1, 0]:
            image_list.append(i)

    if request.method == 'POST':
        # 處理上下張
        bt_p = request.values.get("previous")
        bt_n = request.values.get("next")
        if (bt_p == 'previous'):
            image -= 1
        if (bt_n == 'next'):
            image += 1

        # 處理醫師判斷的結果
        if 'check' in request.form:
            Answer = request.form.getlist('check') #所有 name='check'的checkbox
            #---------------------------------------------------------------------
            # 有幾顆智齒 就只跑幾輪
            if image == len(image_list)-1: #(-5) % 4 = (-2 × 4 + 3) % 4 = 3.
                teeth_number = len(df) - image_list[image] #最後一顆牙 = len(df)-1
            else:
                teeth_number = image_list[image+1]-image_list[image]
            #---------------------------------------------------------------------
            for i in range(teeth_number):
                temp = str(df.iloc[image_list[image]+i,1])
                if temp in Answer: df.iloc[image_list[image]+i,3] =  True
                else: df.iloc[image_list[image]+i,3] =  False #寫錯可以重填一次進去 洗掉原本資料
            print(Answer)
            # 預設切到下一張
            image += 1
    else:
        image = 0

    # 處理顯示資料
    image = image%len(image_list)
    file_name = str(df.iloc[image_list[image], 0])
    template_data = {
        'file_name': file_name,
        'image': os.path.join(img, file_name),
        'image_variant' : str(image+1),
    }

    # 確認是否檢查過
    red_check = False
    if df.iloc[image_list[image], 3] != None:
        red_check = True
    template_data[f'red_check'] = red_check

    # 有幾顆智齒 就只跑幾輪
    if image == len(image_list)-1: #(-5) % 4 = (-2 × 4 + 3) % 4 = 3.
        teeth_number = len(df) - image_list[image] #最後一顆牙 = len(df)-1
    else:
        teeth_number = image_list[image+1]-image_list[image]

    teeth_place = []
    for i in range (teeth_number):
        template_data[f'place_type_{i}'] = str(df.iloc[image_list[image]+i, 1]) + ':' + trans_type(str(df.iloc[image_list[image]+i, 2]))
        teeth_place.append(str(df.iloc[image_list[image]+i, 1]))
    
    for i in range (teeth_number, 4):
        template_data[f'place_type_{i}'] = ''

    # 選擇是否顯示checkbox
    check_18 = False
    check_28 = False
    check_38 = False
    check_48 = False
    for i in range(len(teeth_place)):
        if teeth_place[i] == '18': check_18 = True
        elif teeth_place[i] == '28': check_28 = True
        elif teeth_place[i] == '38': check_38 = True
        elif teeth_place[i] == '48': check_48 = True
    template_data[f'check_18'] = check_18
    template_data[f'check_28'] = check_28
    template_data[f'check_38'] = check_38
    template_data[f'check_48'] = check_48

    # 輸出結果
    df.to_csv('./static/result.csv', index=False)

    return render_template('index.html', **template_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="the website's local port")
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)