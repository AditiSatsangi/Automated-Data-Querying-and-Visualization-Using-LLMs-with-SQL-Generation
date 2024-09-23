import streamlit as st
from sqlalchemy import create_engine, Table, MetaData, text
from sqlalchemy.orm import sessionmaker
import openai
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
# Connect to the MySQL database
engine = create_engine('mysql+pymysql://root:aditi@localhost:3306/sport', echo=False)
metadata = MetaData()
football = Table('football', metadata, autoload_with=engine)

# Function to generate SQL query using OpenAI
def generate_sql_query(user_input):
    # Describe the columns in the 'football' table
    football_columns = ', '.join([col.name for col in football.columns])
    
    messages = [
        {"role": "system", "content": f"The columns for the football table are: {football_columns}."},
        {"role": "user", "content": f"Convert the following user request to an SQL query according to the columns of the 'football' table. User request: {user_input}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )
    
    generated_query = response.choices[0].message['content'].strip()
    keywords = ['SELECT', 'FROM']
    is_valid_query = all(keyword in generated_query.upper() for keyword in keywords)
    
    return generated_query, is_valid_query

# Function to execute the generated SQL query
Session = sessionmaker(bind=engine)
session = Session()

def execute_query(sql_query):
    try:
        result = session.execute(text(sql_query))
        session.commit()
        return result.fetchall(), result.keys()
    except Exception as e:
        session.rollback()
        return str(e), []
    finally:
        session.close()

# Streamlit application
st.title('SQL Query Generation and Visualization')

user_input = st.text_input("Enter your query:")
if st.button("Generate SQL and Execute"):
    sql_query, is_valid_query = generate_sql_query(user_input)
    st.write(f"SQL query generated: {sql_query}")
    
    if is_valid_query:
        results, columns = execute_query(sql_query)
        if isinstance(results, str):
            st.error("The query could not be executed. It may not be related to the database schema.")
            st.error(f"Error details: {results}")
        elif results:
            st.success("Query executed successfully. Here are the results:")
            df = pd.DataFrame(results, columns=columns)
            st.dataframe(df)
            
            st.write("### Select Visualization")
            visualization_type = st.selectbox("Choose a visualization type", ["Bar Graph", "Scatter Plot", "Line Plot"])
            
            if len(df.columns) >= 2:
                x_col = st.selectbox("Select the X-axis column", df.columns, key='xcol')
                y_col = st.selectbox("Select the Y-axis column", df.columns, key='ycol')
                
                if x_col and y_col:
                    plt.figure(figsize=(10, 5))
                    
                    if visualization_type == "Bar Graph":
                        plt.bar(df[x_col], df[y_col])
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title(f'{y_col} vs {x_col}')
                    
                    elif visualization_type == "Scatter Plot":
                        plt.scatter(df[x_col], df[y_col])
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title(f'{y_col} vs {x_col}')
                    
                    elif visualization_type == "Line Plot":
                        plt.plot(df[x_col], df[y_col])
                        plt.xlabel(x_col)
                        plt.ylabel(y_col)
                        plt.title(f'{y_col} vs {x_col}')
                    
                    st.pyplot(plt)
            else:
                st.warning("Not enough columns to create a graph.")
        else:
            st.info("No results found.")
    else:
        st.error("The input is not related to the database schema or generated an invalid query.")
        
        
