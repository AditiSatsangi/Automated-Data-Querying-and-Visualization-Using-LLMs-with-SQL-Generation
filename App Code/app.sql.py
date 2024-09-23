import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import openai
from openai import Completion
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns



# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("API key not found.")
openai.api_key = openai_api_key

# Create database connection
engine = create_engine('mysql+mysqlconnector://sql12711462:gpUZSD8cd5@sql12.freesqldatabase.com/sql12711462')
Session = sessionmaker(bind=engine)
session = Session()

# Mapping of SQL data types to Python data types
sql_to_python_types = {
    'INTEGER': 'int',
    'VARCHAR': 'str',
    'TEXT': 'str',
    'DECIMAL': 'float',
    'DATE': 'str'
}

# Function to convert SQL data type to Python data type
def convert_sql_to_python_type(sql_type):
    for sql_key, py_value in sql_to_python_types.items():
        if sql_key in sql_type:
            return py_value
    return sql_type  # If no conversion found, return the original type

# Retrieve and format table information
inspector = inspect(engine)
tables = inspector.get_table_names()
table_info = {}

for table in tables:
    columns = inspector.get_columns(table)
    column_info = [(col['name'], convert_sql_to_python_type(str(col['type']))) for col in columns]
    table_info[table] = column_info

table_names = list(table_info.keys())
table_data = "\n\n".join(
    f"Table '{table}' has columns:\n  " + "\n  ".join(f"{col[0]} ({col[1]})" for col in table_info[table])
    for table in table_names
)

st.sidebar.code(table_data,language="sql")











#########################################################################################################################################################################
#####QUERIES#####
#########################################################################################################################################################################
def clean_query(q):
    # Split the query into individual queries based on semicolons
    queries = [query.strip() + ';' for query in q.split(';') if query.strip()]

    # Define the keywords that should be on new lines
    keywords = ['SELECT', 'FROM', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'FULL OUTER JOIN', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING']

    # Clean and format each query
    cleaned_queries = []
    for query in queries:
        lines = query.split('\n')
        cleaned_lines = []
        for line in lines:
            words = line.split()
            new_line = []
            for word in words:
                if word.upper() in keywords:
                    if new_line:
                        cleaned_lines.append(' '.join(new_line))
                    cleaned_lines.append(word)
                    new_line = []
                else:
                    new_line.append(word)
            if new_line:
                cleaned_lines.append(' '.join(new_line))
        cleaned_query = '\n'.join(cleaned_lines)
        cleaned_queries.append(cleaned_query)

    # Join the cleaned queries into a single string with each query on a new line
    cleaned_query = '\n\n'.join(cleaned_queries)

    return cleaned_query

def generate_sql_query(user_input, table_data):
    prompt = f"""
    You are a professional SQL developer. Convert the given text prompt to a valid MySQL query using the provided database schema: {table_data}. 

    Convert the user input "{user_input}" to an equivalent MySQL query.

    - Use tables and columns as defined in the database schema.
    - If the user input is unrelated to the database (e.g., "what can you do?", "is chatgpt free?", "give me python commands"), respond with "no valid sql query generated".    
    - For correlation, correlation matrix, or percentile calculations involving string variables, convert strings to integers using one-hot encoding.

    Only provide the correct MySQL query without comments or descriptions.
    """

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )

    return response.choices[0].text.strip()

def execute_query(sql_query):
    
    manipulation_keywords = ['update', 'insert', 'delete', 'truncate', 'drop']
    if any(keyword in sql_query.lower() for keyword in manipulation_keywords):
        error_of_manipulation_keywords = f"User has read access and cannot manipulate data."
        return None , error_of_manipulation_keywords

    try:
        df = pd.read_sql_query(sql_query, engine)
    except Exception as e:
        error_pandas = f"Error executing SQL query: {e}"
        return None , error_pandas

    if df.empty:
        error_df_empty = ("No records in table match requested query.")
        return None , error_df_empty

    return df , None







#########################################################################################################################################################################
#####DESIGNING CHARTS#####
#########################################################################################################################################################################

def get_bar_chart_params(df: pd.DataFrame, table_data: str, chart_info: str) -> tuple[str, str]:
    try:
        prompt = f"Analyze the dataframe structure and content: {df.to_csv()} \n\n" \
                  f"Table data: {table_data} \n\n" \
                  f"Chart info: {chart_info} \n\n" \
                  f"for bar chart humans have easy understanding if x parameter is string and y parameter is numeric\n\n"\
                  f"to plot two column of dataframe as a bar chart, determine which column should be x and y parameters for bar chart and return only two column names separated by space and no description and no comment"

        analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        #st.write("-------------------------")
        #st.write(analysis)
        result = analysis.choices[0].message.content
        #st.write(result)

        x, y = result.split()
        # Check that the columns exist in the dataframe
        if x not in df.columns:
            raise ValueError(f"Column '{x}' does not exist in the dataframe")
        if y and "count(" in y:
            y_column = y.split("(")[1].split(")")[0]
            if y_column not in df.columns:
                raise ValueError(f"Column '{y_column}' does not exist in the dataframe")
        elif y and y not in df.columns:
            raise ValueError(f"Column '{y}' does not exist in the dataframe")

        return x, y
    
    except openai.error.APIError as e:
        st.error(f"API error: {e}")
        raise e
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        raise e
    except Exception as e:
        st.error(f"Unknown error: {e}")
        raise e

def get_line_chart_params(df, user_prompt_input):
    # Determine the x, y columns for a line chart
    x = df.columns[0]
    y = df.columns[1]
    return x, y

def get_pie_chart_params(df, user_prompt_input):
    # Determine the x, y columns for a pie chart
    x = df.columns[0]
    y = None
    
    return x, y

def get_scatter_chart_params(df, user_prompt_input):
    # Determine the x, y columns for a scatter chart
    x = df.columns[0]
    y = df.columns[1]
    
    return x, y

def get_swarm_chart_params(df, user_prompt_input):
    # Determine the x, y columns for a swarm chart
    x = df.columns[0]
    y = df.columns[1]
    
    return x, y

def get_box_chart_params(df, user_prompt_input):
    # Determine the x, y columns for a box chart
    x = df.columns[0]
    y = df.columns[1]
    
    return x, y




def determine_parameters(df, chart_info, table_data, chart_type, user_prompt_input):
    if "bar" in chart_type.lower():
        x, y = get_bar_chart_params(df,table_data,chart_info)
    elif "line" in chart_type.lower():
        x, y = get_line_chart_params(df, user_prompt_input)
    elif "pie" in chart_type.lower():
        x, y = get_pie_chart_params(df, user_prompt_input)
    elif "scatter" in chart_type.lower():
        x, y = get_scatter_chart_params(df, user_prompt_input)
    elif "swarm" in chart_type.lower():
        x, y = get_swarm_chart_params(df, user_prompt_input)
    elif "box" in chart_type.lower():
        x, y = get_box_chart_params(df, user_prompt_input)
    else:
        st.write("Invalid chart type")
        return None, None

    return x, y

def preferable_chart(df, chart_info, table_data):
    if "bar" in chart_info.lower():
        return "bar"
    elif "line" in chart_info.lower():
        return "line"
    elif "pie" in chart_info.lower():
        return "pie"
    elif "scatter" in chart_info.lower():
        return "scatter"
    elif "swarm" in chart_info.lower():
        return "swarm"
    elif "box" in chart_info.lower():
        return "box"
    else:
        return None

def build_logic_for_charts(df, chart_info, table_data):
    session_all_result = []
    chart_type = preferable_chart(df, chart_info, table_data)

    if chart_type:
        x, y = determine_parameters(df, chart_info, table_data, chart_type, user_input)
        title = f"""{chart_type} chart for {x} and {y} """
        session_all_result.append({"question": user_input, "chart_recommendation": chart_type, "x_recommendation": x, "y_recommendation": y, "title_recommendation": title, "hide_graph": False})
    return session_all_result






#########################################################################################################################################################################
#####PLOTTING CHARTS#####
#########################################################################################################################################################################

def create_bar_chart(df, x, y, title):  
    fig, ax = plt.subplots()
    sns.barplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def create_line_chart(df, x, y, title):
    fig, ax = plt.subplots()
    sns.lineplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def create_pie_chart(df, x, y, title):
    fig, ax = plt.subplots()
    plt.pie(df[x].value_counts(), labels=df[x].unique(), autopct='%1.1f%%')
    ax.set_title(title)
    st.pyplot(fig)

def create_scatter_chart(df, x, y, title):
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def create_swarm_chart(df, x, y, title):
    fig, ax = plt.subplots()
    sns.swarmplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def create_box_chart(df, x, y, title):
    fig, ax = plt.subplots()
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
















#########################################################################################################################################################################
#####STREAMLIT APP#####
#########################################################################################################################################################################

st.title("SQL Query Generator and Data Visualizer")

user_input = st.text_input("Enter your query or prompt:")
if user_input:
    sql_query = generate_sql_query(user_input, table_data)
    st.write("Generated SQL Query:")
    st.code(sql_query,language="sql")

    df, error = execute_query(sql_query)
    
    if error:
        st.error(error)
    else:
        st.write("Dataframe:")
        st.write(df)

        #also make sure if user changes x_var or y_var then page shoukd no refresh as it will clear output
        analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Analyze the dataframe structure and content: {df.to_csv()} \n\nalso mention which one chart from ( bar chart , line chart , pie chart, scatter chart, swarm chart ,box chart ) can best describe mentioned dataframe"},
            ]
        )

        chart_info = analysis.choices[0].message.content
        st.write("Chart Information:")
        st.write(chart_info)

        session_all_result = build_logic_for_charts(df, chart_info, table_data)
        #st.write("Session All Result:", session_all_result)        


        for recommendation in session_all_result:
            if recommendation['hide_graph'] == False:
                question = recommendation['question']
                chart_recommendation = recommendation['chart_recommendation']
                x_recommendation = recommendation['x_recommendation']
                y_recommendation = recommendation['y_recommendation']
                title_recommendation = recommendation['title_recommendation']

                st.write(f"Chart for {question}:")
                if "bar" in chart_recommendation.lower():
                    create_bar_chart(df, x_recommendation, y_recommendation, title_recommendation)
                elif "line" in chart_recommendation.lower():
                    create_line_chart(df, x_recommendation, y_recommendation, title_recommendation)
                elif "pie" in chart_recommendation.lower():
                    create_pie_chart(df, x_recommendation, y_recommendation, title_recommendation)
                elif "scatter" in chart_recommendation.lower():
                    create_scatter_chart(df, x_recommendation, y_recommendation, title_recommendation)
                elif 'swarm' in chart_recommendation.lower():
                    create_swarm_chart(df, x_recommendation, y_recommendation, title_recommendation)
                elif 'box' in chart_recommendation.lower():
                    create_box_chart(df, x_recommendation, y_recommendation, title_recommendation)

