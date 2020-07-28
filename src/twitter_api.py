import tweepy

def tweet(text):
    with open('../twitter-api', 'r') as f:
        api_key = f.readline().strip()
        api_key_secret = f.readline().strip()
        access_token = f.readline().strip()
        access_token_secret = f.readline().strip()
        bearer_token = f.readline().strip()

        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(access_token, access_token_secret)

        # Create API object
        api = tweepy.API(auth)

        # Create a tweet
        api.update_status(text)