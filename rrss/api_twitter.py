# Se utiliza la libreria tweepy. Pip install tweey

import tweepy

consumer_key = 'VYZUlMi43ZvZJ8kK45S9I1izB'
consumer_secret = 'sSid17RVgLq5UK7msGHIPo69GJJY0dkEGmkQFXjqaH6fuBLbfH'
access_token = '1624951805124673537-VA7YKinlGoD4LtonRHfAWjZihMJrvE'
access_token_secret = 'lZiPf99fAODthyMnpjgvezqpROvjFMZEuR0ecSdIdounf'

# Autenticación con OAuth
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Crear una instancia de la API de Tweepy
api = tweepy.API(auth, wait_on_rate_limit=True)

search_query = "HudsonKitchen_"
tweets = tweepy.Cursor(api.search_tweets, q=search_query).items(10)

for tweet in tweets:
    print(tweet.text) 

# Error generado por limitaciones de la Api gratuita: raise Forbidden(resp)
#tweepy.errors.Forbidden: 403 Forbidden
#453 - You currently have access to a subset of Twitter API v2 endpoints and limited v1.1 endpoints (e.g. media post, oauth) only. If you need access to this endpoint, you may need a different access level. You can learn more here: https://developer.twitter.com/en/portal/product