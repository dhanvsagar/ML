import tweepy
from textblob import TextBlob

consumer_key    = ''
consumer_secret = ''

access_tocken   = '-'
access_secret   =  ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_tocken, access_secret)

api = tweepy.API(auth)

all_tweets = api.search('Kerala')


for tweet in all_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
