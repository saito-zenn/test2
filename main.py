import streamlit as st

# タイトルを設定
st.title("入力値による画面切り替え")

# テキスト入力欄
input_text = st.text_input("数字を入力してください:")

# ボタン
if st.button("送信"):
    if input_text == "1234":
        # 入力値が1234の場合、別の画面を表示する
        st.write("入力値が1234なので、この画面が表示されます")
    else:
        st.write("入力値が異なります")
