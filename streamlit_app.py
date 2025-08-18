import streamlit as st
import requests
import duckdb
import json
import pandas as pd

db = duckdb.connect("scripts/proto.duckdb")

query = """
SELECT 
    b.Title, 
    b.authors, 
    b.categories, 
    r.Id, 
    r.User_id, 
    r."review/score",
    b.ratingsCount
FROM books b
JOIN ratings r ON b.Title = r.Title
ORDER BY r.User_id, r.Id, b.categories, b.authors;
"""

df = db.execute(query).fetchdf()

# Creating the book dict for decoding

book_df = df[['Id', 'Title']]

book_dict = dict(zip(book_df['Id'], book_df['Title']))

user_list = df['User_id'].unique().tolist()
    
st.title("Recommendation System Test")

users_id = st.selectbox(
    "Choose a user",
    user_list[15:25]
)



st.write("You selected:", users_id)

def create_payload(user_ids):
        return {
            "users": [{"id": user_ids}]
        }


if st.button("Predict"):
    with st.spinner("Thinking..."):
            
            payload = create_payload(user_ids=users_id)

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )        
            predict = response.content

            s = predict.decode('utf-8')  # decodifica bytes para string

            s_dict = json.loads(s)

            st.success(f"Sucess!")

            name_books = []

            for book in s_dict['predictions']:
                   name = book_dict[book]
                   name_books.append(name)

            st.text(f'Recommended books:')

            df = pd.DataFrame(name_books, columns=['Books'], index=list(range(0, len(name_books))))

            st.table(df)
    
