from flask import Flask, render_template, request
import os
import pandas as pd
import argparse
import subprocess

# 先把兩個資料組合起來
# file_path = './pre-processing.py'
# try:
#     subprocess.run(['python', file_path], check=True, capture_output=True)
#     print("Pre-processing script executed successfully.")
# except subprocess.CalledProcessError as e:
#     print("Error in pre-processing script:")
#     print(e.stderr.decode('utf-8'))
#     exit()  # 或者適當處理錯誤

app = Flask(__name__)

# 選取所需檔案
csv_file_path = os.path.join('static', 'val_prediction_final.csv') # sort過的csv
img = os.path.join('static', 'val_full_images')

# 全域變數
df = pd.read_csv(csv_file_path)
df = df.sort_values(by=['file', 'tooth'])
df['Model_acc'] = None 
df['Result'] = None
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
        
        if 'save' in request.form:
            select_18 = request.form.get('select_18')
            select_28 = request.form.get('select_28')
            select_38 = request.form.get('select_38')
            select_48 = request.form.get('select_48')
            Answer = [select_18, select_28, select_38, select_48]
            #---------------------------------------------------------------------
            # 有幾顆智齒 就只跑幾輪
            if image == len(image_list)-1: #(-5) % 4 = (-2 × 4 + 3) % 4 = 3.
                teeth_number = len(df) - image_list[image] #最後一顆牙 = len(df)-1
            else:
                teeth_number = image_list[image+1]-image_list[image]
            #---------------------------------------------------------------------
            for i in range(teeth_number):
                temp = int(df.iloc[image_list[image]+i,1])/10 - 1
                if Answer[int(temp)] == '': 
                    df.iloc[image_list[image]+i,4] =  True
                    df.iloc[image_list[image]+i,5] =  df.iloc[image_list[image]+i,2]
                else: 
                    print(Answer[int(temp)])
                    if str(Answer[int(temp)]) == str(df.iloc[image_list[image]+i,2]):
                        df.iloc[image_list[image]+i,4] =  True 
                    else:
                        df.iloc[image_list[image]+i,4] =  False #寫錯可以重填一次進去 洗掉原本資料
                    df.iloc[image_list[image]+i,5] =  Answer[int(temp)]
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
    if df.iloc[image_list[image], 4] != None:
        red_check = True
    template_data[f'red_check'] = red_check

    # 有幾顆智齒 就只跑幾輪
    if image == len(image_list)-1: #(-5) % 4 = (-2 × 4 + 3) % 4 = 3.
        teeth_number = len(df) - image_list[image] #最後一顆牙 = len(df)-1
    else:
        teeth_number = image_list[image+1]-image_list[image]

    teeth_place = []
    
    for i in range (teeth_number):
        template_data[f'place_{i}'] = str(df.iloc[image_list[image]+i, 1]) + ':' 
        template_data[f'type_model_{i}'] = trans_type(str(df.iloc[image_list[image]+i, 2]))
        template_data[f'type_DR_{i}'] = trans_type(str(df.iloc[image_list[image]+i, 3]))
        teeth_place.append(str(df.iloc[image_list[image]+i, 1]))
        if trans_type(str(df.iloc[image_list[image]+i, 2])) != trans_type(str(df.iloc[image_list[image]+i, 3])):
            template_data[f'red_check_{i}'] = True
        else:
            template_data[f'red_check_{i}'] = False

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
    # df.to_csv('./static/result.csv', index=False)

    return render_template('index.html', **template_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="the website's local port")
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)