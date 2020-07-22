from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objects as go
import tweepy
import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import GetOldTweets3 as got
import time
import requests
import json
import nltk
from nltk.corpus import stopwords
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
nltk.download('stopwords')

def search_for_hashtags(fname, hashtag_phrase):
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''
    # create authentication for accessing Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # initialize Tweepy API
    api = tweepy.API(auth)
    # open the spreadsheet we will write to
    with open('hastag%s.csv' % (fname), 'w+', encoding='utf-8') as file:
        w = csv.writer(file)

        # write header row to spreadsheet
        w.writerow(['timestamp', 'tweet_text', 'username', 'all_hashtags', 'followers_count', 'likes_count'])

        # for each tweet matching our hashtags, write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search, q=hashtag_phrase + ' -filter:retweets', lang="en", tweet_mode='extended',
                                   since="2019-12-03").items(75):
            w.writerow([tweet.created_at, tweet.full_text.replace('\n', ' ').encode('utf-8'),
                        tweet.user.screen_name.encode('utf-8'),
                        [e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count,
                        tweet.favorite_count])


def create_dataset(new):
    dataset = pd.read_csv('{}.csv'.format(new))
    dataset = dataset.dropna()
    os.remove('{}.csv'.format(new))
    index = dataset.index
    if dataset.empty:
        return dataset
    df = pd.DataFrame(dataset.tweet_text)
    pattern = re.compile(r'[\\]+[\w\w\w]+')
    t = []
    for i in df['tweet_text']:
        i = re.sub(pattern, '', i)
        t.append(i[2:-2])
    df4 = pd.DataFrame(t)
    df4.columns = ["tweet_text"]
    df4['timestamp'] = dataset['timestamp']
    return (df4)

def sentiment_analyzer_scores(text):
    score = analyser.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.25:
        return 1
    elif (lb > -0.25) and (lb < 0.25):
        return 0
    else:
        return -1

def histogram(df4):
    df = df4['sentiment']
    fig = px.histogram(df, x="sentiment", nbins=10)

    fig.update_layout(
        title_text="Sentiment Breakdown",
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

def pie_plot(train):
    c = 0
    d = 0
    e = 0
    for i in train['score']:
        if i == 0:
            c += 1
        if i == 1:
            d += 1
        if i == -1:
            e += 1
    colors = ['royalblue', 'lightgreen', 'red']

    fig = go.Figure(
        data=[go.Pie(
            labels=['Neutral', 'Positive', 'Negative'],
            values=[c, d, e])
        ])
    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        textfont_size=20,
        marker=dict(colors=colors,line=dict(color='#000000', width=2))
    )
    fig.update_layout(
        title_text="Distribution Of Sentiments",
        )

    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div

def word_cloud(wd_list,st,stoplist,w):
    stopwords = set(stoplist)
    wordcloud = WordCloud(background_color = 'white', stopwords = stopwords, colormap='jet',max_words=150, width = 1500, height = 800).generate(str(wd_list))
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.title('Most Popular '+st+' Words', fontsize = 30)
    plt.axis('off')
    plt.imshow(wordcloud)
    wordcloud.to_file("home/static/home/img/{}.png".format(st+"_"+w))
    image = "/static/home/img/{}.png".format(st+"_"+w)

    return image

def top_pos_sentiments(train):
    x = train['score'] == 1
    new = train[x]
    new = new.nlargest(10,'sentiment')
    df = pd.DataFrame(new['Tweet_final'])
    pos = []
    for i in df['Tweet_final']:
        i = i.replace('&amp;', '&')
        pos.append(i)
    if len(pos) == 0:
        pos.append("No positive tweets found!")
    return pos

def top_neg_sentiments(train):
    y = train['score'] == -1
    new1 = train[y]
    new1 = new1.nsmallest(10,'sentiment')
    df = pd.DataFrame(new1['Tweet_final'])
    neg = []
    for i in df['Tweet_final']:
        i = i.replace('&amp;', '&')
        neg.append(i)
    if len(neg) == 0:
        neg.append("No negative tweets found!")
    return neg

def handle_tweet(result):
    return result*100

def get_intent(results):
    avg = 0
    c = 0
    for res in results:
        avg = avg + handle_tweet(res)
        c = c+1
    avg = avg/c# 100 is the number of tweets being fetched
    return avg

def compound_score(df4):
    sents = []
    for col in df4['Tweet_final']:
        val = analyser.polarity_scores(col)
        sents.append(val['compound'])
    df4['sentiment'] = sents
    return df4


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst

def show_trends(new_hash):
    since = ["2019-10-01", "2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
             "2020-05-01", "2020-06-01"]
    until = ["2019-10-31", "2019-11-30", "2019-12-31", "2020-01-31", "2020-02-28", "2020-03-30", "2020-04-30",
             "2020-05-30", "2020-06-30"]
    df1 = pd.DataFrame(columns=['tweet', 'date'])
    for i in range(9):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(new_hash) \
            .setSince(since[i]) \
            .setUntil(until[i]) \
            .setLang('en') \
            .setMaxTweets(30)
        # Creation of list that contains all tweets
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        # Creating list of chosen tweet data
        text_tweets = [tweet.text for tweet in tweets]
        text_date = [str(tweet.date)[:-14] for tweet in tweets]
        d = {
            'tweet': text_tweets,
            'date': text_date
         }
        df2=pd.DataFrame(d)
        df1=pd.concat([df1, df2])
    df1['Tweet_final'] = clean_tweets(df1['tweet'])
    df1=df1.drop(["tweet"], axis=1)
    df1=compound_score(df1)
    df1=df1.drop(["Tweet_final"], axis=1)
    df1['y'] = df1['sentiment'].apply(lambda x: handle_tweet(x))
    df1=df1.drop(["sentiment"], axis=1)
    df1 = df1.set_index(pd.to_datetime(df1.iloc[:, 0])).drop('date', axis=1)
    df_year = pd.DataFrame({"y": df1.y.resample("M").sum()})
    df_year['mva'] = df_year.y.rolling(center=True, window=2).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=since, y=df_year['mva'], name='Monthly Average',
                         line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=since, y=df_year['y'], name='Actual',
                         line=dict(color='royalblue', width=4)))
    fig.update_layout(title='Trends in Historic Tweets',
                   xaxis_title ='Month',
                   yaxis_title ='Sentiment Score')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div


def current_graph(df4):
    fig = go.Figure([go.Scatter(x=df4['timestamp'], y=df4['sentiment'])])
    fig.update_layout(
        title_text="Sentiments score against time for current tweet",
    )
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div


def news(request):
    API_Token = '29f170ee129de334f76cc958a89e4252'
    url = 'https://gnews.io/api/v3/search?q='
    url = url + 'covid19' + '&' + 'country=in' + '&' + 'max=9' + '&' + 'token=' + API_Token
    r = requests.get(url).json()

    arr = []
    for i in r['articles']:
        arr.append({'title': i['title'], 'description': i['description'], 'image': i['image'],
                    'name': i['source']['name'], 'url': i['source']['url']})

    arr1 = arr[:3]
    arr2 = arr[3:6]
    arr3 = arr[6:]
    context = {
        'news1': arr1,
        'news2': arr2,
        'news3': arr3,
    }
    return render(request, 'home/news.html', context)

def world():
    url = 'https://covid2019-api.herokuapp.com/v2/country/USa'
    r = requests.get(url).json()

    arr = []
    for i in r['data']:
        arr.append([i['location'], i['confirmed'], i['recovered'], i['deaths'], i['active']])
    df = pd.DataFrame(data=arr)
    df.columns = ['country', 'confirmed', 'recovered', 'deaths', 'active']
    with open('home/static/home/isofile/data.json', 'rb') as fp:
        data = json.load(fp)

    df['iso_location'] = df['country'].apply(lambda x: data[x])


    fig = px.choropleth(df, locations='iso_location', color='active',
                        range_color=(0, 10000),
                        hover_data=['country', 'confirmed', 'recovered', 'deaths'],
                        projection='orthographic',
                        title='Covid19 Worldwide Stats',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        )
    fig.update_geos(
        showocean=True, oceancolor="LightBlue",
    )
    fig.update_layout(margin={"r": 60, "t": 50, "l": 60, "b": 50})

    plot_div = plot(fig, output_type='div', include_plotlyjs=False)

    df2 = df.iloc[:20]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=df2['country'],
        x=df2['active'],
        name='active',
        width=0.8,
        orientation='h',
        marker=dict(
            color='rgba(11, 48, 100,0.6)',
            line=dict(color='rgb(11, 48, 100)', width=3)
        )
    ))
    fig1.add_trace(go.Bar(
        y=df2['country'],
        x=df2['recovered'],
        name='recovered',
        width=0.8,
        orientation='h',
        marker=dict(
            color='rgba(43, 95, 6,0.6)',
            line=dict(color='rgb(43, 95, 6)', width=3)
        )
    ))
    fig1.add_trace(go.Bar(
        y=df2['country'],
        x=df2['deaths'],
        name='deaths',
        width=0.8,
        orientation='h',
        marker=dict(
            color='rgba(141, 27, 16,0.6)',
            line=dict(color='rgb(141, 27, 16)', width=3)
        )
    ))

    fig1.update_layout(
        title='Comparison of reported covid19 cases',
        xaxis_title='Covid19 Cases',
        yaxis_title='Country',
        height=900,
        barmode='stack'),
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    return plot_div, plot_div1, r['dt']

def statistic(request):
    world_graph, bargraph, current_date = world()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    url = 'https://covid2019-api.herokuapp.com/v2/total'
    r = requests.get(url).json()

    info = r['data']

    context = {
        "world": world_graph,
        'bar': bargraph,
        'date': current_date,
        'time': current_time,
        'info': info,
    }
    return render(request, 'home/world.html', context)

def oldtweets(request):
    if 'oldtweets' in request.POST:
        trend = "#"+request.POST.get("oldtweets")
        old = show_trends(trend)
        context ={
            'old': old,
        }
    return render(request, 'home/oldtweets.html', context)


def home1(request):
    return render(request, 'home/home.html')

def home(request):
    if "hashphrase" in request.POST:
        w = request.POST.get("hashphrase")
        if w == "":
            context = {
                'error': "Please input a hashphrase.",
            }

            return render(request, 'home/home.html', context)

        elif w in ['#', '@', '!', '$', '%', '^', '&', '*', '(', ')']:
            context = {
                'error': "Please input a valid hashphrase.",
            }
            return render(request, 'home/home.html', context)

        elif w in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            context = {
                'error': "Please input a valid hashphrase.",
            }
            return render(request, 'home/home.html', context)

        elif w[0] == "#":
            context = {
                'error': "Please input hashphrase without '#'.",
            }

            return render(request, 'home/home.html', context)

        w1 = w.split()
        folder = "home/static/home/img"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        arr = []
        for i in w1:
            if i != 'AND' and i != 'OR':
                arr.append("#" + i)
            else:
                arr.append(i)
        new_hash = " ".join(arr)
        fname = '_'.join(re.findall(r'#(\w+)', new_hash))
        search_for_hashtags(fname, new_hash)
        new = 'hastag' + fname
        df4 = create_dataset(new)
        if df4.empty:
            context = {
                'error': "Dataset Empty Please Try Again!!"
            }
            return render(request, 'home/home.html', context)
        df4['Tweet_final'] = clean_tweets(df4['tweet_text'])
        df4 = compound_score(df4)
        curr = current_graph(df4)
        hist = histogram(df4)
        df4['score'] = df4['Tweet_final'].apply(lambda x: sentiment_analyzer_scores(x))
        # nltk.download('stopwords')
        stoplist = stopwords.words('english')
        i = ['Name', 'dtype', 'object', 'Length', 'Tweet_final', 'amp']
        stoplist.extend(i)
        tws_pos = df4['Tweet_final'][df4['score'] == 1]
        st = "positive"
        pos_image = word_cloud(tws_pos, st, stoplist, w)
        tws_neg = df4['Tweet_final'][df4['score'] == -1]
        st = "negative"
        neg_image = word_cloud(tws_neg, st, stoplist, w)
        pie = pie_plot(df4)
        pos = top_pos_sentiments(df4)
        neg = top_neg_sentiments(df4)

        context = {
            'plot2': pie,
            'plot3': hist,
            'pos': pos_image,
            'neg': neg_image,
            'positive': pos,
            'negative': neg,
            # 'old': oldtweets,
            'curr': curr,
        }

    return render(request, 'home/welcome.html', context)
