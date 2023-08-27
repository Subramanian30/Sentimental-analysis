try:
    from datetime import timedelta
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
    from datetime import datetime
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from sqlalchemy import create_engine
    import sqlite3
    import re
    import math
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    print("Sentiment is running")

except Exception as e:
    print("Error  {} ".format(e))


default_args = {
    'owner': 'Subramanian',
    'start_date': datetime(2023, 8, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# Define your DAG with the provided arguments
dag = DAG(
    'sentiment_analysis',
    default_args=default_args,
    max_active_runs=1,
    description='Sentiment Analysis of one plus nord ce 2 lite',
    schedule_interval="@daily",
    
)

def extract():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    product_links = []
    short_review=[]
    long_review=[]
    for x in range(1, 501):
        link = f'https://www.amazon.in/OnePlus-Nord-Black-128GB-Storage/product-reviews/B09WQY65HN/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={x}&pageNumber={x+1}'
        product_links.append(link)
        
    for link in product_links:
        r = requests.get(link, headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        reviews_title = soup.find_all('a', class_='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold')
        reviews_long = soup.find_all('span', class_='a-size-base review-text review-text-content')
        try:
            for title in reviews_title:
                short_review.append(title.text.strip())
        except:
            short_review.append("No reviews found")
        try:
            for long_text in reviews_long:
                long_review.append(long_text.text.strip())
        except:
            long_review.append("No reviews found")
    print(len(short_review))
    return short_review,long_review


extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract,
    dag=dag,
)


def split_short(**context):
    ti = context['task_instance']
    extracted_lists = ti.xcom_pull(task_ids='extract_task')
    short,long=extracted_lists
    df = pd.DataFrame()
   
    df['short_review'] = short
    df['long_review']= long
    print(len(df.index))
    print(df.head())


    df[['rating', 'short_review']] = df['short_review'].str.split('\n', 1, expand=True)
    df['rating'] = df['rating'].str.replace(' out of 5 stars', '')

    return df


rating = PythonOperator(
    task_id='rating',
    python_callable= split_short,
    provide_context = True,
    dag=dag,
)

def remove_nulls(**context):
    ti = context['task_instance']
    df = ti.xcom_pull(task_ids='rating')
    print(df.head())
    df = df.fillna('None')
    null_counts = df.isnull().sum()
    print(null_counts)
    return df

null_removal = PythonOperator(
    task_id='null_removal',
    python_callable = remove_nulls,
    provide_context=True,
    dag=dag,
)

def cleaning_data(**context):
    ti = context['task_instance']
    df = ti.xcom_pull(task_ids='null_removal')
    df['short_review'] = df['short_review'].apply(lambda x: re.sub("[^a-zA-Z0-9, ']", "", x))
    df['long_review'] = df['long_review'].apply(lambda x: re.sub("[^a-zA-Z0-9, ']", "", x))
    return df

cleaning = PythonOperator(
    task_id='cleaning',
    python_callable=cleaning_data,
    provide_context=True,
    dag=dag,
)

def transform_reviews(**context):

    ti = context['task_instance']
    df = ti.xcom_pull(task_ids='cleaning')

    df['preprocessed_short_review'] = df['short_review'].str.lower()
    df['preprocessed_short_review'] = df['preprocessed_short_review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    df['short_review_tokens'] = df['preprocessed_short_review'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    df['filtered_short_review_tokens'] = df['short_review_tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # Preprocess long reviews
    df['preprocessed_long_review'] = df['long_review'].str.lower()
    df['preprocessed_long_review'] = df['preprocessed_long_review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    df['long_review_tokens'] = df['preprocessed_long_review'].apply(word_tokenize)
    df['filtered_long_review_tokens'] = df['long_review_tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # Select relevant columns
    df = df[['preprocessed_short_review', 'filtered_short_review_tokens', 'preprocessed_long_review', 'filtered_long_review_tokens']]
    df = df[(df['filtered_short_review_tokens'].apply(len) > 0) & (df['filtered_long_review_tokens'].apply(len) > 0)]
    print(df.head())
    return df

transform = PythonOperator(
    task_id='transform',
    python_callable=transform_reviews,
    provide_context=True,
    dag=dag,
)

def sentiment_analysis(**context):
    ti = context['task_instance']
    python_df = ti.xcom_pull(task_ids='transform')
    sid = SentimentIntensityAnalyzer()

    python_df['short_sentiment_score'] = python_df['preprocessed_short_review'].apply(lambda x: sid.polarity_scores(x)['compound'])

    python_df['long_sentiment_score'] = python_df['preprocessed_long_review'].apply(lambda x: sid.polarity_scores(x)['compound'])

    python_df['combined_sentiment_score'] = (python_df['short_sentiment_score'] + python_df['long_sentiment_score']) / 2

    python_df['sentiment'] = python_df['combined_sentiment_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

    sid = SentimentIntensityAnalyzer()
    python_df['combined_filtered_tokens'] = python_df['filtered_short_review_tokens'] + python_df['filtered_long_review_tokens']
    def calculate_average_sentiment(row):
        scores = [sid.polarity_scores(token)['compound'] for token in row['combined_filtered_tokens']]
        return math.fsum(scores)/len(scores)

    python_df['average_sentiment_score'] = python_df.apply(calculate_average_sentiment, axis=1).astype(float)

    python_df['word_sentiment'] = python_df['average_sentiment_score'].apply(lambda score: 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral')
    return python_df

sentiment = PythonOperator(
    task_id='sentiment',
    python_callable=sentiment_analysis,
    provide_context=True,
    dag=dag,
)

def ml(**context):
    ti = context['task_instance']
    python_df = ti.xcom_pull(task_ids='sentiment')
    X = python_df['preprocessed_short_review']
    y = python_df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train)
    y_pred = nb_model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

machine_learning = PythonOperator(
    task_id='machine_learning',
    python_callable=ml,
    provide_context=True,
    dag=dag,
)

def load_function(**context):
    ti = context['task_instance']
    df_1 = ti.xcom_pull(task_ids='sentiment')
    df_2 = ti.xcom_pull(task_ids='rating')
    df = df_2.join(df_1)
    print(df.head())
    conn = sqlite3.connect('sentiment_final.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            short_review VARCHAR(8000),
            long_review VARCHAR(8000),
            rating REAL,
            sentiment VARCHAR(100)
            )
        ''')

    for index, row in df.iterrows():
        cursor.execute('''
            INSERT INTO reviews (
                short_review, long_review, rating,sentiment
            )
            VALUES (?, ?, ?, ?)
            ''', (
                row['short_review'], row['long_review'],row['rating'], row['sentiment']
            ))

    
    conn.commit()
    conn.close()


load_task = PythonOperator(
    task_id='load_task',
    python_callable=load_function,
    provide_context=True,
    dag=dag,
)

extract_task >>rating >> null_removal >> cleaning >>transform>>sentiment>> load_task
sentiment>>machine_learning