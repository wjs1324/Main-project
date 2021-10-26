import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from sub import input_image_trim, main_run, result_crop


app = Flask(__name__)

# 메인페이지
@app.route('/', methods=['GET'])
def index():
    for root, dirs, files in os.walk('C:/dev/AtomWorkspace/ggulfit/static/img/uploads/'):
        if files:
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".jpeg"):
                    os.remove(os.path.join(root, file))
                    print('{} 삭제'.format(os.path.join(root, file)), file=sys.stderr)
    for root, dirs, files in os.walk('C:/dev/AtomWorkspace/ggulfit/static/img/assets/representative/custom/female/'):
        if files:
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".jpeg"):
                    os.remove(os.path.join(root, file))
                    print('{} 삭제'.format(os.path.join(root, file)), file=sys.stderr)
    for root, dirs, files in os.walk('C:/dev/AtomWorkspace/ggulfit/static/img/assets/representative/celeba_hq/src/female'):
        if files:
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".jpeg"):
                    os.remove(os.path.join(root, file))
                    print('{} 삭제'.format(os.path.join(root, file)), file=sys.stderr)
    for root, dirs, files in os.walk('C:/dev/AtomWorkspace/ggulfit/static/img/expr/results/celeba_hq/'):
        if files:
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".jpeg"):
                    os.remove(os.path.join(root, file))
                    print('{} 삭제'.format(os.path.join(root, file)), file=sys.stderr)
    return render_template("index.html", image=False)

@app.route('/', methods=['POST'])
def image_save():
   file = request.files['file']
   if file:
       filename = secure_filename(file.filename)

       # 파일 저장
       fileroute = 'static/img/uploads/' + filename
       file.save(fileroute)

       # 이미지 crop
       img = Image.open(fileroute).convert('RGB')
       h_size = int(256 * (img.size[1] / img.size[0]))
       resizedImg = img.resize((256, h_size))
       croppedImg = resizedImg.crop((0,0,256,256))
       croproute = 'static/img/assets/representative/custom/female/' + filename
       croppedImg.save(croproute)
       # stargan모델 align mode (모델에 적합한 이미지로 변형)
       input_image_trim()

       # 인공지능 모델 실행
       result = hair_recommend(croproute, filename)
   return render_template("index.html", image=True, fname=filename, result=result)

def hair_recommend(croproute, filename):
   model = load_model('new_face_classifier.h5')
   model.load_weights('new_model_weights.h5')
   test_image = Image.open(croproute)
   prediction = model(preprocess_input(np.expand_dims(np.array(test_image.resize((224,224)))[:, :, 0:3], axis=0)))
   class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
   faceshape = class_names[np.argmax(prediction)]
   main_run()

   with open('faceshape.json', encoding="utf-8") as json_file:
       faceshape_js = json.load(json_file, strict=False)
   with open('hairstyle.json', encoding="utf-8") as json_file:
       hairstyle_js = json.load(json_file, strict=False)

   result = {'faceshape':[], 'womanhair':[]}
   result['faceshape'] = faceshape_js[faceshape]
   for i in hairstyle_js[faceshape][0]:
       hair = faceshape + i + '.jpg'
       name = hairstyle_js[faceshape][0][i]
       simul = result_crop(i, filename)
       result['womanhair'].append([hair, name, simul])

   return result


# 로그인, 로그아웃 페이지
@app.route('/login', methods=['POST', 'GET'])
def login():
    id = ''
    id_err = ''
    pw_err = ''
    error = None
    if request.method == "POST":
        id = request.form['id']
        pw = request.form['pw']

        # 아아디, 비밀번호 칸이 비어있는지 확인
        if not id: id_err = '아이디를 입력해주세요.'
        elif not pw: pw_err = '비밀번호를 입력해주세요.'
        else:
            # 안 비어있으면 다음 단계(아이디, 비밀번호가 알맞은지 확인)
            conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
            cur = conn.cursor()
            id_test = cur.execute('SELECT * FROM member WHERE id="' + id + '"')
            pw_test = cur.fetchone()
            conn.close()
            # 먼저 아이디가 존재하는 아이디인지 확인
            if id_test == 0:
                error = '존재하지 않는 아이디입니다.'
            # 아이다가 존재하면 이제 비밀번호가 일치하는지 확인
            elif not check_password_hash(pw_test[1], pw):
                error = '비밀번호가 일치하지 않습니다.'
            else:
                # 모두 통과됐을 경우, 로그인 완료하면서 메인 페이지로 이동
                session.clear()
                session['user_id'] = id
                return redirect('/')
    return render_template('login.html', id=id, id_err=id_err, pw_err=pw_err, error=error)

@app.route('/logout', methods=['POST', 'GET'])
def logout():
    session.pop('user_id')
    return redirect('/')


# 회원가입 페이지
@app.route('/register', methods=['GET'])
def register():
    return render_template("register.html")

@app.route('/register', methods=['POST'])
def post_register():
    arr = request.get_json()
    print('[DEBUG] {}'.format(arr), file=sys.stderr)
    if arr['err'].count('') == 6 and arr['info'].count('') == 0:
        arr['info'][1] = generate_password_hash(arr['info'][1])
        conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
        cur = conn.cursor()
        cur.execute('INSERT INTO member VALUES(%s, %s, %s, %s, %s)', arr['info'])
        conn.commit()
        conn.close()
        return str(1)
    return str(0)

@app.route('/register/chk', methods=['POST'])
def idchk():
    idChk = request.get_json()
    if idChk:
        conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
        cur = conn.cursor()
        data = cur.execute('SELECT * FROM member WHERE id="' + idChk + '"')
        conn.close()
        return str(data)
    return str(-1)


# 마이페이지
@app.route('/mypage', methods=['GET'])
def mypage():
    return render_template('mypage_check.html')

@app.route('/mypage', methods=['POST'])
def mypage_check():
    conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
    cur = conn.cursor()
    cur.execute('SELECT * FROM member WHERE id="' + session['user_id'] + '"')
    info = cur.fetchone()
    conn.close()

    pw = request.form['password_check']
    if check_password_hash(info[1], pw):
        return render_template('mypage_modify.html', info=info)
    return render_template('mypage_check.html')

@app.route('/modify', methods=['POST'])
def modify():
    conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
    cur = conn.cursor()
    cur.execute('SELECT * FROM member WHERE id="' + session['user_id'] + '"')
    info = cur.fetchone()

    arr = request.get_json()
    print('[DEBUG] {}'.format(arr), file=sys.stderr)

    if arr:
        if arr['err'].count('') == 4 and ((arr['pw'] and arr['pw_chk']) or arr['name'] != info[2] or arr['pnum'] != info[4]):
            if arr['pw']:
                if arr['pw'] == arr['pw_chk']:
                    arr['pw'] = generate_password_hash(arr['pw'])
                    cur.execute('UPDATE member SET password="' + arr['pw'] + '" WHERE id="' + session['user_id'] + '"')
                    print('[DEBUG] 비밀번호를 {}로 수정'.format(arr['pw']), file=sys.stderr)
                else:
                    return str(0)

            if arr['name'] and arr['name'] != info[2]:
                cur.execute('UPDATE member SET name="' + arr['name'] + '" WHERE id="' + session['user_id'] + '"')
                print('[DEBUG] 이름을 {}로 수정 '.format(arr['name']), file=sys.stderr)

            if arr['pnum'] and arr['pnum'] != info[4]:
                cur.execute('UPDATE member SET contact="' + arr['pnum'] + '" WHERE id="' + session['user_id'] + '"')
                print('[DEBUG] 전화번호를 {}로 수정'.format(arr['pnum']), file=sys.stderr)

            conn.commit()
            return str(1)
        return str(0)
    return render_template('mypage_modify.html', info=info)

@app.route('/reservation', methods=['POST'])
def reservation():
    conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
    cur = conn.cursor()
    cur.execute('SELECT * FROM reservation WHERE member_id="' + session['user_id'] + '" ORDER BY res_date DESC')
    reservation_list = cur.fetchall()

    present = []
    past = []
    today = datetime.now()
    for i in reservation_list:
        cur.execute('SELECT * FROM hair_shop WHERE num="' + str(i[2]) + '"')
        shop = cur.fetchone()

        if i[3] < today:
            past.append([shop[1], shop[3], datetime.date(i[3]), datetime.time(i[3]).strftime("%H:%M")])
        else:
            present = [i[0], shop[1], shop[3], datetime.date(i[3]), datetime.time(i[3]).strftime("%H:%M")]
    conn.close()

    return render_template('mypage_reservation.html', past=past, present=present)

@app.route('/reservation/cancel', methods=['POST'])
def cancellation():
    num = request.form.get('reservation_num')
    print('[DEBUG] {}'.format(num), file=sys.stderr)
    if num:
        conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
        cur = conn.cursor()
        cur.execute('DELETE FROM reservation WHERE num="' + num + '"')
        conn.commit()
        conn.close()
    return reservation()

@app.route('/unregister', methods=['POST'])
def unregister():
    reasons = request.get_json()
    if reasons:
        conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
        cur = conn.cursor()
        for reason in reasons:
            cur.execute('INSERT INTO withdrawal(reason) VALUES(%s)', reason)
        cur.execute('DELETE FROM reservation WHERE member_id="' + session['user_id'] + '"')
        cur.execute('DELETE FROM member WHERE id="' + session['user_id'] + '"')
        conn.commit()
        conn.close()
        return str(1)
    return render_template('mypage_delete.html')


# 서브 페이지
@app.route('/intro', methods=['GET'])
def intro():
    return render_template('introduce.html')

@app.route('/feature', methods=['GET'])
def feature():
    return render_template('face_feature.html')

@app.route('/thesis', methods=['GET'])
def thesis():
    return render_template('thesis.html')


# 미용실 추천 페이지
@app.route('/shop', methods=['GET'])
def shop():
    return render_template('reserve_shop.html')

@app.route('/shop', methods=['POST'])
def shop_info():
    shop = request.form.get('shop')
    date = request.form.get('date')

    today = str(datetime.now().date())
    current_hour = int(datetime.now().strftime("%H"))
    reserved_time = []
    if date == today and current_hour >= 10:
        for i in range(10, current_hour+1):
            reserved_time.append(str(i))

    conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
    cur = conn.cursor()
    reserved = cur.execute('SELECT res_date FROM reservation WHERE hair_shop_num=(SELECT num FROM hair_shop WHERE name="' + shop + '") and DATE(res_date) = "' + date + '"')
    if reserved > 0:
        for i in cur.fetchall():
            print('[DEBUG] {}'.format(i), file=sys.stderr)

            time = str(datetime.time(i[0]).strftime("%H"))
            reserved_time.append(time)
        conn.close()
    reserved_time = json.dumps(list(set(reserved_time)), ensure_ascii=False)
    print('[DEBUG] {}'.format(reserved_time), file=sys.stderr)
    return reserved_time

@app.route('/shop/reserve', methods=['POST'])
def shop_reserve():
    arr = request.get_json()
    if arr:
        conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
        cur = conn.cursor()
        cur.execute('SELECT num FROM hair_shop WHERE name="' + arr[1] + '"')
        arr[0] = session['user_id']
        arr[1] = cur.fetchone()[0]
        arr[2] = " ".join(arr[2])
        print('[DEBUG] {}'.format(arr), file=sys.stderr)

        cur.execute('INSERT INTO reservation(member_id, hair_shop_num, res_date) VALUES(%s, %s, %s)', arr)
        conn.commit()
        conn.close()
    return str(1)


# 미용실 정보들을 DB에 저장
# @app.route('/api/db', methods=['POST'])
# def shop_info_save():
#     # arr = request.form.get('arr')
#     arr = request.get_json()
#     print('[DEBUG] {}'.format(arr), file=sys.stderr)
#     count = 0
#     for i in arr:
#         conn = pymysql.connect(user='ggulfit', passwd='1234', db='hairdb')
#         cur = conn.cursor()
#         test = cur.execute('SELECT * FROM hair_shop WHERE Name="' + i[0] + '"')
#         if test == 0:
#             if not i[1][0]:
#                 i[1] = i[1][1]
#                 cur.execute('INSERT INTO hair_shop(Name, Address, Contact) VALUES(%s, %s, %s)', i)
#             else:
#                 print('[DEBUG] {}'.format(1), file=sys.stderr)
#                 i[1] = i[1][0]
#                 print('[DEBUG] {}'.format(i), file=sys.stderr)
#                 cur.execute('INSERT INTO hair_shop(Name, Address, Contact) VALUES(%s, %s, %s)', i)
#                 print('[DEBUG] {}'.format(2), file=sys.stderr)
#             conn.commit()
#             count += 1
#             conn.close()
#     return str(count)


if __name__ == '__main__':
    app.secret_key = 'ggulfit'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(host='0.0.0.0', port=80, debug=True)
